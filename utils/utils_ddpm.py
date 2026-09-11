"""
utils_ddpm.py
Clean DDPM utilities for controlled feature-space augmentation.

Main use:
- EfficientNet-B3 features (1536D) are scaled and reduced by PCA inside each CV fold.
- Conditional DDPM is trained in PCA/latent feature space.
- Synthetic features are generated moderately and filtered by classifier confidence + nearest-neighbor distance.
- Low-data ablation compares RealOnly vs Real+DDPM for classical ML classifiers.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    XGBClassifier = None
    HAS_XGB = False


def set_seed(seed: int = 1337) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


@dataclass
class DiffusionSchedule:
    T: int = 100
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def tensors(self):
        betas = torch.linspace(self.beta_start, self.beta_end, self.T, device=self.device)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        return betas, alphas, alphas_bar


def timestep_embedding(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class EpsModel(nn.Module):
    def __init__(self, dim: int, n_classes: int, width: int = 512, tdim: int = 128, cdim: int = 64):
        super().__init__()
        self.class_emb = nn.Embedding(n_classes, cdim)
        self.fc_t = nn.Sequential(nn.Linear(tdim, cdim), nn.SiLU())
        self.net = nn.Sequential(
            nn.Linear(dim + cdim + cdim, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        te = self.fc_t(timestep_embedding(t, 128))
        ce = self.class_emb(y)
        return self.net(torch.cat([x, te, ce], dim=1))


def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, alphas_bar: torch.Tensor) -> torch.Tensor:
    a = alphas_bar[t][:, None]
    return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise


@torch.no_grad()
def p_sample_loop(model: nn.Module, n: int, y: torch.Tensor, schedule: DiffusionSchedule, dim: int) -> torch.Tensor:
    device = schedule.device
    betas, alphas, alphas_bar = schedule.tensors()
    x = torch.randn(n, dim, device=device)
    for t in reversed(range(schedule.T)):
        tt = torch.full((n,), t, dtype=torch.long, device=device)
        eps = model(x, tt, y)
        a, ab = alphas[t], alphas_bar[t]
        mean = (1 / torch.sqrt(a)) * (x - ((1 - a) / torch.sqrt(1 - ab)) * eps)
        x = mean + torch.sqrt(betas[t]) * torch.randn_like(x) if t > 0 else mean
    return x


def train_ddpm(
    Xtr_latent: np.ndarray,
    ytr: np.ndarray,
    n_classes: int,
    steps: int = 4000,
    batch_size: int = 128,
    lr: float = 2e-4,
    schedule: Optional[DiffusionSchedule] = None,
    width: int = 512,
    verbose: bool = True,
    log_every: int = 500,
) -> nn.Module:
    """Train conditional DDPM on latent/PCA features."""
    schedule = schedule or DiffusionSchedule()
    device = schedule.device
    _, _, alphas_bar = schedule.tensors()

    X_t = torch.tensor(Xtr_latent, dtype=torch.float32, device=device)
    y_t = torch.tensor(ytr, dtype=torch.long, device=device)
    model = EpsModel(dim=X_t.shape[1], n_classes=n_classes, width=width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for step in range(steps):
        idx = torch.randint(0, len(X_t), (batch_size,), device=device)
        x0, yy = X_t[idx], y_t[idx]
        tt = torch.randint(0, schedule.T, (batch_size,), device=device)
        noise = torch.randn_like(x0)
        xt = q_sample(x0, tt, noise, alphas_bar)
        pred_noise = model(xt, tt, yy)
        loss = F.mse_loss(pred_noise, noise)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if verbose and log_every and (step + 1) % log_every == 0:
            print(f"[DDPM] step {step + 1}/{steps} | loss={loss.item():.4f}")

    model.eval()
    return model


def _allocate_synthetic_per_class(ytr: np.ndarray, n_total: int, n_classes: int) -> np.ndarray:
    n_total = int(max(0, n_total))
    base = n_total // n_classes
    rem = n_total % n_classes
    out = np.full(n_classes, base, dtype=int)
    if rem > 0:
        counts = np.bincount(ytr, minlength=n_classes)
        for c in np.argsort(counts)[:rem]:
            out[c] += 1
    return out


def _fit_qc_tools(
    X_real: np.ndarray,
    y_real: np.ndarray,
    n_classes: int,
    seed: int = 1337,
    nn_percentile: float = 95,
    nn_multiplier: float = 1.25,
):
    qc_clf = LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced", random_state=seed)
    qc_clf.fit(X_real, y_real)

    nn_models: Dict[int, Optional[NearestNeighbors]] = {}
    nn_thresholds: Dict[int, float] = {}
    for c in range(n_classes):
        Xc = X_real[y_real == c]
        if len(Xc) == 0:
            nn_models[c] = None
            nn_thresholds[c] = np.inf
            continue
        nn_models[c] = NearestNeighbors(n_neighbors=1).fit(Xc)
        if len(Xc) >= 2:
            nn2 = NearestNeighbors(n_neighbors=2).fit(Xc)
            d2, _ = nn2.kneighbors(Xc)
            same_class_dist = d2[:, 1]
            thr = float(np.percentile(same_class_dist, nn_percentile) * nn_multiplier)
            nn_thresholds[c] = thr if np.isfinite(thr) and thr > 0 else np.inf
        else:
            nn_thresholds[c] = np.inf
    return qc_clf, nn_models, nn_thresholds


def _select_synthetic_for_class(
    Xcand: np.ndarray,
    class_id: int,
    n_target: int,
    qc_clf,
    nn_models: Dict[int, Optional[NearestNeighbors]],
    nn_thresholds: Dict[int, float],
    filter_synthetic: bool = True,
    confidence_threshold: float = 0.85,
    allow_filter_fallback: bool = False,
) -> np.ndarray:
    if n_target <= 0 or len(Xcand) == 0:
        return np.empty((0, Xcand.shape[1]), dtype=np.float32)
    if not filter_synthetic:
        return Xcand[:n_target].astype(np.float32)

    proba = qc_clf.predict_proba(Xcand)
    pred = qc_clf.classes_[np.argmax(proba, axis=1)]
    ci = int(np.where(qc_clf.classes_ == class_id)[0][0]) if class_id in qc_clf.classes_ else None
    class_conf = proba[:, ci] if ci is not None else np.zeros(len(Xcand), dtype=float)
    mask_conf = (pred == class_id) & (class_conf >= confidence_threshold)

    nn = nn_models.get(class_id)
    thr = nn_thresholds.get(class_id, np.inf)
    if nn is not None:
        dist = nn.kneighbors(Xcand, return_distance=True)[0].ravel()
        mask_nn = dist <= thr
        norm_dist = dist / (thr + 1e-8) if np.isfinite(thr) and thr > 0 else np.zeros_like(dist)
    else:
        mask_nn = np.ones(len(Xcand), dtype=bool)
        norm_dist = np.zeros(len(Xcand), dtype=float)

    score = class_conf - 0.25 * norm_dist
    hard_idx = np.where(mask_conf & mask_nn)[0]
    hard_idx = hard_idx[np.argsort(-score[hard_idx])]

    if allow_filter_fallback and len(hard_idx) < n_target:
        fallback = np.argsort(-score)
        selected = list(hard_idx)
        for idx in fallback:
            if idx not in selected:
                selected.append(int(idx))
            if len(selected) >= n_target:
                break
        hard_idx = np.array(selected, dtype=int)

    return Xcand[hard_idx[:n_target]].astype(np.float32)


@torch.no_grad()
def generate_controlled_ddpm_features(
    ddpm: nn.Module,
    X_real_latent: np.ndarray,
    y_real: np.ndarray,
    n_classes: int,
    n_total: int,
    schedule: DiffusionSchedule,
    seed: int = 1337,
    filter_synthetic: bool = True,
    confidence_threshold: float = 0.85,
    nn_percentile: float = 95,
    nn_multiplier: float = 1.25,
    oversample_factor: float = 4.0,
    allow_filter_fallback: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate class-conditional DDPM features with confidence + NN filtering."""
    if n_total <= 0:
        return np.empty((0, X_real_latent.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)

    set_seed(seed)
    n_per_class = _allocate_synthetic_per_class(y_real, n_total, n_classes)
    qc_clf, nn_models, nn_thresholds = _fit_qc_tools(
        X_real_latent, y_real, n_classes=n_classes, seed=seed,
        nn_percentile=nn_percentile, nn_multiplier=nn_multiplier,
    )

    dim = X_real_latent.shape[1]
    gen_x, gen_y = [], []
    for c, n_keep in enumerate(n_per_class):
        if n_keep <= 0:
            continue
        n_cand = max(n_keep, int(np.ceil(n_keep * oversample_factor)))
        yy = torch.full((n_cand,), c, dtype=torch.long, device=schedule.device)
        Xcand = p_sample_loop(ddpm, n_cand, yy, schedule=schedule, dim=dim).cpu().numpy()
        Xsel = _select_synthetic_for_class(
            Xcand=Xcand,
            class_id=c,
            n_target=n_keep,
            qc_clf=qc_clf,
            nn_models=nn_models,
            nn_thresholds=nn_thresholds,
            filter_synthetic=filter_synthetic,
            confidence_threshold=confidence_threshold,
            allow_filter_fallback=allow_filter_fallback,
        )
        if verbose:
            print(f"    [QC class {c}] candidates={n_cand} | target={n_keep} | kept={len(Xsel)}")
        gen_x.append(Xsel)
        gen_y.append(np.full(len(Xsel), c, dtype=np.int64))

    if not gen_x:
        return np.empty((0, X_real_latent.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.vstack(gen_x).astype(np.float32), np.concatenate(gen_y).astype(np.int64)


# Backward-compatible alias for old notebook cells.
_generate_controlled_ddpm_features = generate_controlled_ddpm_features


def get_classical_models(seed: int = 1337, include_ann: bool = True, include_xgb: bool = True):
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel="rbf", C=2.0, gamma="scale", probability=True),
        "LR": LogisticRegression(max_iter=3000, C=2.0, random_state=seed),
        "RF": RandomForestClassifier(n_estimators=400, max_features="sqrt", random_state=seed, n_jobs=-1),
    }
    if include_ann:
        models["ANN"] = MLPClassifier(hidden_layer_sizes=(512, 256), batch_size=64, max_iter=600, random_state=seed)
    if include_xgb and HAS_XGB:
        models["XGB"] = XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.8,
            eval_metric="mlogloss", tree_method="hist", random_state=seed,
        )
    return models


def evaluate_models(models: dict, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray):
    rows = []
    for model_name, base_model in models.items():
        clf = clone(base_model) if hasattr(base_model, "get_params") else copy.deepcopy(base_model)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xva)
        acc = accuracy_score(yva, pred)
        p, r, f1, _ = precision_recall_fscore_support(yva, pred, average="macro", zero_division=0)
        rows.append({"Model": model_name, "Accuracy": acc, "Precision": p, "Recall": r, "F1": f1})
    return rows


def summarize_results(df_raw: pd.DataFrame) -> pd.DataFrame:
    return (
        df_raw.groupby(["Train_fraction", "Setting", "Model"], as_index=False)
        .agg(
            Accuracy=("Accuracy", "mean"),
            F1=("F1", "mean"),
            Precision=("Precision", "mean"),
            Recall=("Recall", "mean"),
            Acc_std=("Accuracy", "std"),
            F1_std=("F1", "std"),
            N_real_train_mean=("N_real_train", "mean"),
            N_synthetic_mean=("N_synthetic", "mean"),
            PCA_var_mean=("PCA_var", "mean"),
        )
        .sort_values(["Train_fraction", "Model", "Setting"])
    )


def make_delta_table(df_summary: pd.DataFrame) -> pd.DataFrame:
    pivot = df_summary.pivot_table(
        index=["Train_fraction", "Model"],
        columns="Setting",
        values=["Accuracy", "F1", "Acc_std", "F1_std", "N_real_train_mean", "N_synthetic_mean"],
    )
    out = pd.DataFrame(index=pivot.index)
    out["RealOnly_Acc"] = pivot[("Accuracy", "RealOnly")] * 100
    out["DDPM_Acc"] = pivot[("Accuracy", "Real+DDPM")] * 100
    out["Delta_Acc"] = out["DDPM_Acc"] - out["RealOnly_Acc"]
    out["RealOnly_F1"] = pivot[("F1", "RealOnly")] * 100
    out["DDPM_F1"] = pivot[("F1", "Real+DDPM")] * 100
    out["Delta_F1"] = out["DDPM_F1"] - out["RealOnly_F1"]
    out["RealOnly_Acc_std"] = pivot[("Acc_std", "RealOnly")] * 100
    out["DDPM_Acc_std"] = pivot[("Acc_std", "Real+DDPM")] * 100
    out["N_real_train"] = pivot[("N_real_train_mean", "RealOnly")]
    out["N_synthetic"] = pivot[("N_synthetic_mean", "Real+DDPM")]
    out = out.reset_index().sort_values(["Train_fraction", "Delta_F1"], ascending=[True, False])
    return out.round(3)


def print_delta_table(df_delta: pd.DataFrame, train_fraction: Optional[float] = None) -> None:
    df = df_delta.copy()
    if train_fraction is not None:
        df = df[np.isclose(df["Train_fraction"], train_fraction)]
    print(df.to_string(index=False))


def run_lowdata_ddpm_ablation(
    root,
    feat_name: str = "features_b3_1536.npy",
    label_name: str = "labels_b3.npy",
    class_name: str = "classes.json",
    out_csv_name: str = "table3_lowdata_ddpm_ablation_10fold.csv",
    train_fracs: Sequence[float] = (0.25, 0.50, 0.75, 1.00),
    n_splits: int = 10,
    seed: int = 1337,
    pca_dim: int = 64,
    diffusion_steps: int = 100,
    ddpm_train_steps: int = 4000,
    ddpm_batch_size: int = 128,
    ddpm_lr: float = 2e-4,
    ddpm_width: int = 512,
    synth_ratio: float = 1.0,
    max_synthetic_total: int = 500,
    confidence_threshold: float = 0.85,
    nn_percentile: float = 95,
    nn_multiplier: float = 1.25,
    oversample_factor: float = 4.0,
    allow_filter_fallback: bool = False,
    include_ann: bool = True,
    include_xgb: bool = True,
    verbose: bool = True,
    ddpm_log_every: int = 500,
):
    """Run low-data ablation: RealOnly vs Real+DDPM."""
    root = Path(root)
    X = np.load(root / feat_name).astype(np.float32)
    y = np.load(root / label_name).astype(np.int64).ravel()

    import json
    with open(root / class_name, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    n_classes = len(class_names)

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    schedule = DiffusionSchedule(T=diffusion_steps, device=device)
    models = get_classical_models(seed=seed, include_ann=include_ann, include_xgb=include_xgb)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    if verbose:
        print("Device:", device, "| X:", X.shape, "| y:", y.shape, "| classes:", class_names)
        print("Train fractions:", train_fracs, "| n_splits:", n_splits, "| DDPM steps:", ddpm_train_steps)

    rows = []
    for frac in train_fracs:
        if verbose:
            print("\n" + "=" * 70)
            print(f"LOW-DATA FRACTION = {frac:.2f}")
            print("=" * 70)

        for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
            Xtr_full, ytr_full = X[tr_idx], y[tr_idx]
            Xva_raw, yva = X[va_idx], y[va_idx]

            if frac < 1.0:
                Xtr_raw, _, ytr, _ = train_test_split(
                    Xtr_full, ytr_full, train_size=frac, stratify=ytr_full, random_state=seed + fold
                )
            else:
                Xtr_raw, ytr = Xtr_full, ytr_full

            scaler = StandardScaler()
            Xtr_scaled = scaler.fit_transform(Xtr_raw)
            Xva_scaled = scaler.transform(Xva_raw)
            actual_pca_dim = int(min(pca_dim, Xtr_scaled.shape[0] - 1, Xtr_scaled.shape[1]))
            pca = PCA(n_components=actual_pca_dim, random_state=seed)
            Xtr_lat = pca.fit_transform(Xtr_scaled).astype(np.float32)
            Xva_lat = pca.transform(Xva_scaled).astype(np.float32)
            pca_var = float(pca.explained_variance_ratio_.sum())

            if verbose:
                print(
                    f"\n[frac={frac:.2f} | fold={fold:02d}] "
                    f"train_real={len(ytr)}, val={len(yva)}, pca_dim={actual_pca_dim}, pca_var={pca_var:.4f}"
                )

            for rec in evaluate_models(models, Xtr_lat, ytr, Xva_lat, yva):
                rows.append({
                    "Train_fraction": frac, "Setting": "RealOnly", "Fold": fold,
                    "N_real_train": len(ytr), "N_synthetic": 0,
                    "PCA_dim": actual_pca_dim, "PCA_var": pca_var, **rec,
                })

            if verbose:
                print(f"[frac={frac:.2f} | fold={fold:02d}] Train DDPM...")
            set_seed(seed + fold)
            ddpm = train_ddpm(
                Xtr_lat, ytr, n_classes=n_classes,
                steps=ddpm_train_steps, batch_size=ddpm_batch_size, lr=ddpm_lr,
                schedule=schedule, width=ddpm_width, verbose=verbose, log_every=ddpm_log_every,
            )

            n_syn_target = min(int(round(len(ytr) * synth_ratio)), int(max_synthetic_total))
            Xg_lat, yg = generate_controlled_ddpm_features(
                ddpm=ddpm, X_real_latent=Xtr_lat, y_real=ytr, n_classes=n_classes,
                n_total=n_syn_target, schedule=schedule, seed=seed + fold,
                filter_synthetic=True, confidence_threshold=confidence_threshold,
                nn_percentile=nn_percentile, nn_multiplier=nn_multiplier,
                oversample_factor=oversample_factor, allow_filter_fallback=allow_filter_fallback,
                verbose=verbose,
            )

            Xtr_aug = np.vstack([Xtr_lat, Xg_lat]).astype(np.float32)
            ytr_aug = np.concatenate([ytr, yg]).astype(np.int64)
            if verbose:
                print(
                    f"[frac={frac:.2f} | fold={fold:02d}] "
                    f"target_syn={n_syn_target} | kept_syn={len(yg)} | aug_train={len(ytr_aug)}"
                )

            for rec in evaluate_models(models, Xtr_aug, ytr_aug, Xva_lat, yva):
                rows.append({
                    "Train_fraction": frac, "Setting": "Real+DDPM", "Fold": fold,
                    "N_real_train": len(ytr), "N_synthetic": len(yg),
                    "PCA_dim": actual_pca_dim, "PCA_var": pca_var, **rec,
                })

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df_raw = pd.DataFrame(rows)
    df_summary = summarize_results(df_raw)
    df_delta = make_delta_table(df_summary)

    out_path = root / out_csv_name
    raw_path = out_path.with_name(out_path.stem + "_raw" + out_path.suffix)
    delta_path = out_path.with_name(out_path.stem + "_delta" + out_path.suffix)
    df_summary.to_csv(out_path, index=False)
    df_raw.to_csv(raw_path, index=False)
    df_delta.to_csv(delta_path, index=False)

    if verbose:
        print("\nSaved summary ->", out_path)
        print("Saved raw folds ->", raw_path)
        print("Saved delta table ->", delta_path)
        print("\n===== DDPM vs RealOnly comparison (percent) =====")
        print(df_delta.to_string(index=False))

    return df_summary, df_raw, df_delta

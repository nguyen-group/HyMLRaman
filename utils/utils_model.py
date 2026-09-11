from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from torchvision import models


def build_cnn_model(
    model_name: str,
    num_classes: int,
    dropout: float = 0.6,
    pretrained: bool = True,
):
    """Build supported image classifiers with a replaced classification head."""
    name = model_name.lower().replace("-", "_")

    if name in {"efficientnet_b0", "effnet_b0", "b0"}:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))

    elif name in {"efficientnet_b3", "effnet_b3", "b3"}:
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b3(weights=weights)
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))

    elif name in {"regnet_y_400mf", "regnety400mf", "regnety"}:
        weights = models.RegNet_Y_400MF_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.regnet_y_400mf(weights=weights)
        in_f = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))

    elif name in {"vit_b16", "vit_b_16", "vit"}:
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        in_f = model.heads.head.in_features
        model.heads.head = nn.Sequential(nn.Dropout(min(dropout, 0.3)), nn.Linear(in_f, num_classes))

    elif name in {"resnet50", "resnet_50"}:
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        in_f = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return model


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, label_smoothing: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization optimizer wrapper."""
    def __init__(self, params, base_optimizer, rho: float = 0.05, adaptive: bool = False, **kwargs):
        if rho < 0.0:
            raise ValueError("Invalid rho; should be non-negative.")
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group.get("adaptive", False) else 1.0) * p.grad * scale
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = (torch.abs(p) if group.get("adaptive", False) else 1.0) * p.grad
                norms.append(grad.norm(p=2).to(shared_device))
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    def step(self, closure=None):
        raise NotImplementedError("SAM requires first_step and second_step.")


class SpectraCNN(nn.Module):
    """1D CNN classifier for 1536-D feature vectors."""
    def __init__(self, input_dim: int = 1536, n_classes: int = 6):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.res1x1 = nn.Conv1d(32, 64, kernel_size=1)
        self.se_fc1 = nn.Linear(64, 16)
        self.se_fc2 = nn.Linear(16, 64)
        self.pool = nn.AdaptiveMaxPool1d(16)
        self.fc1 = nn.Linear(64 * 16, 256)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(x1))
        out = F.relu(out + self.res1x1(x1))
        w = F.adaptive_avg_pool1d(out, 1).squeeze(-1)
        w = F.relu(self.se_fc1(w))
        w = torch.sigmoid(self.se_fc2(w)).unsqueeze(-1)
        out = out * w
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.drop(out)
        return self.fc2(out)


class TorchCNNWrapper(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper for SpectraCNN."""
    def __init__(
        self,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 64,
        n_classes: int = 6,
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.model = None

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        y_tensor = torch.tensor(y, dtype=torch.long)
        ds = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.model = SpectraCNN(input_dim=X_tensor.shape[-1], n_classes=self.n_classes).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            total, correct, loss_sum = 0, 0, 0.0
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                loss = criterion(logits, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.item()) * xb.size(0)
                total += yb.size(0)
                correct += (logits.argmax(1) == yb).sum().item()
            if self.verbose and (epoch == 1 or epoch == self.epochs or epoch % 5 == 0):
                print(f"[CNN] epoch {epoch:03d}/{self.epochs} | loss={loss_sum/total:.4f} | acc={correct/total:.4f}")
        return self

    @torch.no_grad()
    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
        logits = self.model(X_tensor)
        return logits.argmax(1).cpu().numpy()


def run_7model_confusion_matrices(
    root,
    feat_name="features_b3_1536.npy",
    label_name="labels_b3.npy",
    class_name="classes.json",
    run_models=("CNN", "ANN", "RF", "SVM", "KNN", "LR", "XGB"),
    n_folds=10,
    seed=1337,
    cnn_epochs=30,
    cnn_batch_size=64,
    cnn_lr=1e-3,
    save_csv=True,
    verbose=True,
):
    """
    Run 10-fold CV confusion matrices for 7 models and save per-model CM CSV files.

    Models:
        CNN, ANN, RF, SVM, KNN, LR, XGB

    Expected files under root:
        - features_b3_1536.npy
        - labels_b3.npy
        - classes.json

    Outputs
    -------
    cm_cnn_fold10.csv
    cm_ann_fold10.csv
    cm_rf_fold10.csv
    cm_svm_fold10.csv
    cm_knn_fold10.csv
    cm_lr_fold10.csv
    cm_xgb_fold10.csv

    Returns
    -------
    avg_cms : dict
        Mean normalized confusion matrices, values in 0-1.
    summary_df : pandas.DataFrame
        Accuracy, precision, recall per model.
    """

    import json
    import random
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch

    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    root = Path(root)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = np.load(root / feat_name).astype(np.float32)
    y = np.load(root / label_name).astype(int).ravel()

    with open(root / class_name, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    n_classes = len(class_names)

    filename_map = {
        "CNN": "cm_cnn_fold10.csv",
        "ANN": "cm_ann_fold10.csv",
        "RF": "cm_rf_fold10.csv",
        "SVM": "cm_svm_fold10.csv",
        "KNN": "cm_knn_fold10.csv",
        "LR": "cm_lr_fold10.csv",
        "XGB": "cm_xgb_fold10.csv",
    }

    def _save_cm_csv(model_name, cm_norm):
        out_csv = root / filename_map[model_name]
        pd.DataFrame(
            cm_norm * 100.0,
            index=class_names,
            columns=class_names,
        ).to_csv(out_csv, encoding="utf-8-sig", float_format="%.1f")

        if verbose:
            print(f"Saved CM CSV -> {out_csv}")

    def _run_sklearn_cv(model):
        cms, accs, pres, recs = [], [], [], []
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])

        for tr, va in skf.split(X, y):
            pipe.fit(X[tr], y[tr])
            pred = pipe.predict(X[va])

            cms.append(confusion_matrix(y[va], pred, labels=np.arange(n_classes), normalize="true"))
            accs.append(accuracy_score(y[va], pred))
            pres.append(precision_score(y[va], pred, average="macro", zero_division=0))
            recs.append(recall_score(y[va], pred, average="macro", zero_division=0))

        return np.mean(np.stack(cms), axis=0), np.mean(accs), np.mean(pres), np.mean(recs)

    def _run_cnn_cv():
        cms, accs, pres, recs = [], [], [], []
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for tr, va in skf.split(X, y):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X[tr])
            Xva = scaler.transform(X[va])

            clf = TorchCNNWrapper(
                epochs=cnn_epochs,
                lr=cnn_lr,
                batch_size=cnn_batch_size,
                n_classes=n_classes,
                device=str(device),
                verbose=False,
            )

            clf.fit(Xtr, y[tr])
            pred = clf.predict(Xva)

            cms.append(confusion_matrix(y[va], pred, labels=np.arange(n_classes), normalize="true"))
            accs.append(accuracy_score(y[va], pred))
            pres.append(precision_score(y[va], pred, average="macro", zero_division=0))
            recs.append(recall_score(y[va], pred, average="macro", zero_division=0))

        return np.mean(np.stack(cms), axis=0), np.mean(accs), np.mean(pres), np.mean(recs)

    model_bank = {
        "ANN": lambda: MLPClassifier(
            hidden_layer_sizes=(512, 256),
            batch_size=64,
            max_iter=600,
            random_state=42,
        ),
        "RF": lambda: RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "SVM": lambda: SVC(
            kernel="rbf",
            C=2.0,
            gamma="scale",
            probability=True,
            random_state=42,
        ),
        "KNN": lambda: KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1,
        ),
        "LR": lambda: LogisticRegression(
            max_iter=5000,
            C=2.0,
            solver="lbfgs",
            multi_class="auto",
        ),
        "XGB": lambda: XGBClassifier(
            max_depth=7,
            n_estimators=400,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        ),
    }

    avg_cms = {}
    rows = []

    for model_name in run_models:
        model_name = str(model_name).upper()

        if verbose:
            print(f"\n========== {model_name} ==========")

        if model_name == "CNN":
            cm, acc, pre, rec = _run_cnn_cv()
        elif model_name in model_bank:
            cm, acc, pre, rec = _run_sklearn_cv(model_bank[model_name]())
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        avg_cms[model_name] = cm

        if save_csv:
            _save_cm_csv(model_name, cm)

        rows.append({
            "Method": model_name,
            "Accuracy": float(acc),
            "Precision": float(pre),
            "Recall": float(rec),
        })

        if verbose:
            print(f"{model_name:4s} | Acc={acc:.4f} | Prec={pre:.4f} | Rec={rec:.4f}")

    summary_df = pd.DataFrame(rows)

    if verbose:
        print("\n=== Summary (10-fold CV) ===")
        print(summary_df.to_string(index=False))

    return avg_cms, summary_df


def run_table2_ml_benchmark(
    feat_path,
    label_path,
    out_csv=None,
    methods=("RF", "LR", "XGB", "SVM", "ANN", "KNN"),
    n_splits=10,
    use_scaler=True,
    use_pca=False,
    n_pca=256,
    random_state=1337,
    verbose=True,
):
    """
    Run Table 2 machine-learning benchmark on saved CNN embeddings.

    Returns a DataFrame with:
    Method, Accuracy, Precision, Recall, F1, Acc_std, F1_std
    """

    from pathlib import Path

    import numpy as np
    import pandas as pd

    from sklearn.base import clone
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from xgboost import XGBClassifier

    feat_path = Path(feat_path)
    label_path = Path(label_path)

    if not feat_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feat_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    X = np.load(feat_path).astype(np.float32)
    y = np.load(label_path).astype(np.int64).ravel()

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X/y size mismatch: X={X.shape}, y={y.shape}")

    model_bank = {
        "RF": RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        ),
        "LR": LogisticRegression(
            max_iter=5000,
            C=2.0,
            solver="lbfgs",
            multi_class="auto",
        ),
        "XGB": XGBClassifier(
            max_depth=7,
            n_estimators=400,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
        ),
        "SVM": SVC(
            kernel="rbf",
            C=2.0,
            gamma="scale",
            probability=True,
            random_state=random_state,
        ),
        "ANN": MLPClassifier(
            hidden_layer_sizes=(512, 256),
            batch_size=64,
            max_iter=600,
            random_state=random_state,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1,
        ),
    }

    methods = [str(m).upper().replace("XGBOOST", "XGB") for m in methods]
    unknown = [m for m in methods if m not in model_bank]
    if unknown:
        raise ValueError(f"Unsupported methods: {unknown}. Supported: {list(model_bank)}")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []

    if verbose:
        print("X:", X.shape, "| y:", y.shape)
        print("Methods:", methods)

    for method in methods:
        accs, pres, recs, f1s = [], [], [], []

        if verbose:
            print(f"\n========== {method} ==========")

        for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
            steps = []

            if use_scaler:
                steps.append(("scaler", StandardScaler()))

            if use_pca:
                steps.append(("pca", PCA(n_components=n_pca, random_state=random_state)))

            steps.append(("clf", clone(model_bank[method])))
            pipe = Pipeline(steps)

            pipe.fit(X[tr], y[tr])
            pred = pipe.predict(X[va])

            acc = accuracy_score(y[va], pred)
            pre, rec, f1, _ = precision_recall_fscore_support(
                y[va],
                pred,
                average="macro",
                zero_division=0,
            )

            accs.append(acc)
            pres.append(pre)
            recs.append(rec)
            f1s.append(f1)

            if verbose:
                print(f"[{method} fold {fold:02d}] Acc={acc*100:.2f}% | F1={f1*100:.2f}%")

        rows.append({
            "Method": method,
            "Accuracy": float(np.mean(accs)),
            "Precision": float(np.mean(pres)),
            "Recall": float(np.mean(recs)),
            "F1": float(np.mean(f1s)),
            "Acc_std": float(np.std(accs, ddof=1)),
            "F1_std": float(np.std(f1s, ddof=1)),
        })

    df = pd.DataFrame(rows)

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

        if verbose:
            print("\nSaved Table 2 CSV ->", out_csv)

    if verbose:
        print("\n====== TABLE 2 ML BENCHMARK ======")
        print(df.to_string(index=False))

    return df


# ---------------------------------------------------------------------
# Table 2 original benchmark with CNN + classical ML models
# ---------------------------------------------------------------------
def run_table2_original_with_cnn(
    root,
    feat_name="features_b3_1536.npy",
    label_name="labels_b3.npy",
    class_name="classes.json",
    fallback_class_name="class_to_idx.json",
    out_csv_name="table2_feats_b3_orig.csv",
    methods=("CNN", "ANN", "RF", "SVM", "KNN", "LR", "XGB"),
    n_splits=10,
    seed=1337,
    use_scaler=True,
    use_pca=False,
    n_pca=256,
    cnn_epochs=30,
    cnn_lr=1e-3,
    cnn_batch_size=64,
    verbose=True,
):
    """
    Run Table 2 original benchmark on saved CNN/deep embeddings.

    This function is a backward-compatible helper for old notebooks that call
    run_table2_original_with_cnn(...). It evaluates:
        CNN, ANN, RF, SVM, KNN, LR, XGB

    Expected files under root
    -------------------------
    - features_b3_1536.npy
    - labels_b3.npy
    - classes.json or class_to_idx.json

    Output CSV columns
    ------------------
    Method, Accuracy, Precision, Recall, F1, Acc_std, F1_std
    """

    import json
    import random
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch

    from sklearn.base import clone
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from xgboost import XGBClassifier

    root = Path(root)
    feat_path = root / feat_name
    label_path = root / label_name
    class_path = root / class_name
    fallback_path = root / fallback_class_name
    out_csv = root / out_csv_name if out_csv_name is not None else None

    if not feat_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feat_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    X = np.load(feat_path).astype(np.float32)
    y = np.load(label_path).astype(np.int64).ravel()

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X/y size mismatch: X={X.shape}, y={y.shape}")

    if class_path.exists():
        with open(class_path, "r", encoding="utf-8") as f:
            class_names = [str(x) for x in json.load(f)]
    elif fallback_path.exists():
        with open(fallback_path, "r", encoding="utf-8") as f:
            c2i = json.load(f)
        class_names = [None] * len(c2i)
        for k, v in c2i.items():
            class_names[int(v)] = str(k)
    else:
        class_names = [str(i) for i in sorted(np.unique(y))]

    n_classes = len(class_names)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_bank = {
        "CNN": TorchCNNWrapper(
            epochs=cnn_epochs,
            lr=cnn_lr,
            batch_size=cnn_batch_size,
            n_classes=n_classes,
            device=device,
            verbose=False,
        ),
        "ANN": MLPClassifier(
            hidden_layer_sizes=(512, 256),
            batch_size=64,
            max_iter=600,
            random_state=seed,
        ),
        "RF": RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        ),
        "SVM": SVC(
            kernel="rbf",
            C=2.0,
            gamma="scale",
            probability=True,
            random_state=seed,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1,
        ),
        "LR": LogisticRegression(
            max_iter=5000,
            C=2.0,
            solver="lbfgs",
            multi_class="auto",
        ),
        "XGB": XGBClassifier(
            max_depth=7,
            n_estimators=400,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        ),
    }

    def _norm_method(m):
        m = str(m).upper().strip()
        if m == "XGBOOST":
            m = "XGB"
        return m

    methods = [_norm_method(m) for m in methods]
    unknown = [m for m in methods if m not in model_bank]
    if unknown:
        raise ValueError(f"Unsupported methods: {unknown}. Supported: {list(model_bank)}")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []

    if verbose:
        print("Device:", device)
        print("X:", X.shape, "| y:", y.shape)
        print("Classes:", class_names)
        print("Methods:", methods)

    for method in methods:
        accs, pres, recs, f1s = [], [], [], []

        if verbose:
            print(f"\n========== {method} ==========")

        for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
            steps = []

            if use_scaler:
                steps.append(("scaler", StandardScaler()))

            if use_pca:
                steps.append(("pca", PCA(n_components=n_pca, random_state=seed)))

            clf = clone(model_bank[method])
            steps.append(("clf", clf))
            pipe = Pipeline(steps)

            pipe.fit(X[tr], y[tr])
            pred = pipe.predict(X[va])

            acc = accuracy_score(y[va], pred)
            pre, rec, f1, _ = precision_recall_fscore_support(
                y[va],
                pred,
                average="macro",
                zero_division=0,
            )

            accs.append(acc)
            pres.append(pre)
            recs.append(rec)
            f1s.append(f1)

            if verbose:
                print(f"[{method} fold {fold:02d}] Acc={acc*100:.2f}% | F1={f1*100:.2f}%")

        rows.append({
            "Method": method,
            "Accuracy": float(np.mean(accs)),
            "Precision": float(np.mean(pres)),
            "Recall": float(np.mean(recs)),
            "F1": float(np.mean(f1s)),
            "Acc_std": float(np.std(accs, ddof=1)),
            "F1_std": float(np.std(f1s, ddof=1)),
        })

    df = pd.DataFrame(rows)

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        if verbose:
            print("\nSaved Table 2 CSV ->", out_csv)

    if verbose:
        print("\n====== TABLE 2 ORIGINAL WITH CNN ======")
        print(df.to_string(index=False))

    return df


# Short alias for convenience.
run_table2_with_cnn = run_table2_original_with_cnn

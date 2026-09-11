from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as Fv
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader
from torchvision import datasets

from .utils_data import make_loader_from_indices
from .utils_model import FocalLoss, SAM, build_cnn_model


# ---------------------------------------------------------------------
# ENV HELPERS — merged from utils_env.py
# ---------------------------------------------------------------------
def set_seed(seed: int = 1337, deterministic: bool = False) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def get_amp_context(device=None, dtype=None):
    """
    Return a callable context manager for mixed precision.

    Usage
    -----
    amp_ctx = get_amp_context(device)
    with amp_ctx():
        ...
    """
    from contextlib import nullcontext
    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda":
        dtype = dtype or torch.float16
        return lambda: torch.amp.autocast("cuda", dtype=dtype)

    return nullcontext


def check_ml_environment() -> None:
    """Print basic ML environment information."""
    import torch
    import sklearn

    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)

    print("PyTorch:", torch.__version__)
    print("sklearn:", sklearn.__version__)

    try:
        import xgboost
        print("xgboost:", xgboost.__version__)
    except Exception as exc:
        print("xgboost: not available", exc)


def _save_history_csv(history: Dict, out_dir, filename: str = "history.csv"):
    """Save a training history dict to CSV, padding shorter columns with NaN."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean = {}
    max_len = 0
    for key, value in history.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy().tolist()
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, (int, float, str)):
            value = [value]
        elif value is None:
            continue
        else:
            try:
                value = list(value)
            except TypeError:
                value = [value]

        clean[key] = value
        max_len = max(max_len, len(value))

    if max_len == 0:
        return None

    for key, value in clean.items():
        if len(value) < max_len:
            clean[key] = value + [np.nan] * (max_len - len(value))

    if "epoch" not in clean:
        clean = {"epoch": list(range(1, max_len + 1)), **clean}

    path = out_dir / filename
    pd.DataFrame(clean).to_csv(path, index=False)
    return path


def mixup_alpha(epoch: int) -> float:
    if epoch < 4:
        return 0.2
    if epoch < 8:
        return 0.1
    return 0.0


def mixup_alpha_schedule(epoch: int, total_epochs: Optional[int] = None) -> float:
    return mixup_alpha(epoch)


def apply_mixup(x, y, alpha: float, p: float = 1.0):
    if alpha <= 0 or np.random.rand() > p:
        return x, y, y, 1.0, False
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam, True


def build_scheduler(optimizer, epochs: int, lr: float, warmup_iters: int = 3):
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_iters)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs - warmup_iters),
        eta_min=lr * 0.01,
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_iters])


@torch.no_grad()
def evaluate_classifier(model, loader, criterion, device, tta_mode: str = "none") -> Dict[str, float]:
    model.eval()
    y_true, y_pred = [], []
    loss_sum, total = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = tta_batch_logits(model, imgs, mode=tta_mode) if tta_mode != "none" else model(imgs)
        loss = criterion(logits, labels)
        loss_sum += loss.item() * imgs.size(0)
        total += labels.size(0)
        y_true.append(labels.cpu().numpy())
        y_pred.append(logits.argmax(1).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "loss": loss_sum / total,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train_sam_mixup_model(
    model,
    train_loader,
    val_loader,
    out_dir: str,
    epochs: int = 25,
    lr: float = 2e-4,
    weight_decay: float = 5e-4,
    patience: int = 7,
    device=None,
    criterion=None,
    save_prefix: str = "best",
    track_grad_norm: bool = True,
    save_history: bool = True,
    history_name: str = "history.csv",
):
    """
    Train image classifier using FocalLoss + SAM + Mixup.

    The returned history contains:
        train_loss, val_loss, train_acc, val_acc, lr, grad_norm, grad_norms.

    If save_history=True, the function automatically writes:
        out_dir/history.csv

    This allows utils_plot.plot_training_curves() to draw all 4 panels,
    including Gradient Norm Evolution.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    amp_ctx = get_amp_context(device)
    criterion = criterion or FocalLoss(alpha=1.0, gamma=2.0)

    optimizer = SAM(model.parameters(), torch.optim.AdamW, lr=lr, weight_decay=weight_decay)

    # SAM.second_step() internally calls base_optimizer.step(). Therefore,
    # the scheduler should track optimizer.base_optimizer to avoid scheduler warnings.
    scheduler = build_scheduler(optimizer.base_optimizer, epochs=epochs, lr=lr)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
        "grad_norm": [],
        "grad_norms": [],
    }

    best_acc, best_loss = 0.0, float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        alpha = mixup_alpha(epoch)
        loss_sum, correct, total = 0.0, 0, 0
        epoch_grad_norms = []

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            imgs, ya, yb, lam, used = apply_mixup(imgs, labels, alpha)

            optimizer.zero_grad(set_to_none=True)

            with amp_ctx():
                logits = model(imgs)
                loss = (
                    lam * criterion(logits, ya)
                    + (1 - lam) * criterion(logits, yb)
                ) if used else criterion(logits, labels)

            loss.backward()

            if track_grad_norm:
                grads = [
                    p.grad.detach().norm(p=2)
                    for p in model.parameters()
                    if p.grad is not None
                ]
                if len(grads) > 0:
                    grad_norm = torch.norm(torch.stack(grads), p=2).item()
                    epoch_grad_norms.append(grad_norm)

            optimizer.first_step(zero_grad=True)

            with amp_ctx():
                logits = model(imgs)
                loss = (
                    lam * criterion(logits, ya)
                    + (1 - lam) * criterion(logits, yb)
                ) if used else criterion(logits, labels)

            loss.backward()
            optimizer.second_step(zero_grad=True)

            loss_sum += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = loss_sum / total
        train_acc = correct / total
        val = evaluate_classifier(model, val_loader, criterion, device)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val["acc"])
        history["lr"].append(optimizer.base_optimizer.param_groups[0]["lr"])

        if track_grad_norm:
            mean_grad = float(np.mean(epoch_grad_norms)) if len(epoch_grad_norms) > 0 else np.nan
            history["grad_norm"].append(mean_grad)
            history["grad_norms"].append(mean_grad)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train {train_acc*100:.2f}%/{train_loss:.4f} | "
            f"Val {val['acc']*100:.2f}%/{val['loss']:.4f}"
        )

        improved = False
        if val["acc"] > best_acc:
            best_acc = val["acc"]
            torch.save(model.state_dict(), out_dir / f"{save_prefix}_by_acc.pt")
            improved = True

        if val["loss"] < best_loss:
            best_loss = val["loss"]
            torch.save(model.state_dict(), out_dir / f"{save_prefix}_by_loss.pt")
            improved = True

        if save_history:
            _save_history_csv(history, out_dir, history_name)

        patience_counter = 0 if improved else patience_counter + 1
        if patience_counter >= patience:
            print("[EarlyStopping]")
            break

    if save_history:
        path = _save_history_csv(history, out_dir, history_name)
        print(f"[SAVED] {path}")

    return history


def train_standard_classifier(
    model,
    train_loader,
    val_loader,
    out_dir: str,
    epochs: int = 25,
    lr: float = 3e-5,
    weight_decay: float = 1e-4,
    patience: int = 7,
    device=None,
    save_name: str = "best.pt",
    label_smoothing: float = 0.05,
    save_history: bool = True,
    history_name: str = "history.csv",
):
    """Standard AdamW + CrossEntropy training loop, useful for ViT baseline."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(True)

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, epochs=epochs, lr=lr)

    best_acc, patience_counter = 0.0, 0
    history = {
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = total_correct / total
        train_loss = total_loss / total

        val = evaluate_classifier(model, val_loader, criterion, device)
        scheduler.step()

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val["acc"])
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train {train_acc*100:.2f}% | "
            f"Val {val['acc']*100:.2f}%"
        )

        if val["acc"] > best_acc:
            best_acc = val["acc"]
            torch.save(model.state_dict(), out_dir / save_name)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                if save_history:
                    _save_history_csv(history, out_dir, history_name)
                break

        if save_history:
            _save_history_csv(history, out_dir, history_name)

    if save_history:
        path = _save_history_csv(history, out_dir, history_name)
        print(f"[SAVED] {path}")

    return history


def tta_batch_logits(model, imgs, mode: str = "none"):
    """Test-time augmentation for validation/inference."""
    if mode == "none":
        return model(imgs)
    if mode == "flip":
        logits1 = model(imgs)
        logits2 = model(torch.flip(imgs, dims=[-1]))
        return (logits1 + logits2) / 2
    if mode == "flip_fivecrop":
        b, c, h, w = imgs.shape
        th, tw = int(h * 0.9), int(w * 0.9)
        patches = []
        for i in range(b):
            crops = Fv.five_crop(imgs[i], size=(th, tw))
            crops = [Fv.resize(crop, (h, w)) for crop in crops]
            crops = torch.stack(crops, dim=0)
            flips = torch.flip(crops, dims=[-1])
            all_patches = torch.cat([imgs[i].unsqueeze(0), crops, flips], dim=0)
            patches.append(all_patches)
        big = torch.stack(patches, dim=0).view(-1, c, h, w)
        logits = model(big)
        return logits.view(b, -1, logits.shape[-1]).mean(1)
    raise ValueError(f"Unknown TTA mode: {mode}")


def run_kfold_training(
    data_dir: str,
    out_dir: str,
    model_name: str = "effnet_b3",
    img_size: int = 320,
    batch_size: int = 16,
    epochs: int = 50,
    lr: float = 2e-4,
    weight_decay: float = 5e-4,
    patience: int = 7,
    seed: int = 1337,
    k_folds: int = 5,
    use_swa: bool = True,
    swa_start: float = 0.7,
    tta_mode: str = "flip",
):
    """K-fold CNN training with SAM, optional SWA and TTA."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_ds = datasets.ImageFolder(data_dir)
    labels = np.array([y for _, y in full_ds.samples], dtype=np.int64)
    classes = full_ds.classes
    num_classes = len(classes)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    rows = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        fold_dir = out_dir / f"fold{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        dl_train = make_loader_from_indices(
            data_dir, tr_idx, img_size, batch_size, train=True, labels=labels[tr_idx]
        )
        dl_val = make_loader_from_indices(
            data_dir, va_idx, img_size, batch_size, train=False
        )

        model = build_cnn_model(
            model_name, num_classes=num_classes, dropout=0.6, pretrained=True
        ).to(device)
        criterion = FocalLoss(1.0, 2.0, 0.05)
        optimizer = SAM(model.parameters(), torch.optim.AdamW, lr=lr, weight_decay=weight_decay)
        scheduler = build_scheduler(optimizer.base_optimizer, epochs, lr)
        amp_ctx = get_amp_context(device)
        swa_model = AveragedModel(model) if use_swa else None
        swa_begin_epoch = int(np.ceil(epochs * swa_start))

        best_acc, best_loss = 0.0, float("inf")
        best_acc_ep, best_loss_ep = -1, -1
        patience_counter = 0

        fold_history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "lr": [],
        }

        print(f"\n[fold{fold:02d}] Train {len(tr_idx)} | Val {len(va_idx)}")

        for ep in range(1, epochs + 1):
            model.train()
            total, correct, loss_sum = 0, 0, 0.0

            for x, y in dl_train:
                x, y = x.to(device), y.to(device)
                alpha = mixup_alpha_schedule(ep, epochs)
                x, ya, yb, lam, used = apply_mixup(x, y, alpha, p=0.5)
                optimizer.zero_grad(set_to_none=True)

                with amp_ctx():
                    logits = model(x)
                    loss = (
                        lam * criterion(logits, ya)
                        + (1 - lam) * criterion(logits, yb)
                    ) if used else criterion(logits, y)

                loss.backward()
                optimizer.first_step(zero_grad=True)

                with amp_ctx():
                    logits = model(x)
                    loss = (
                        lam * criterion(logits, ya)
                        + (1 - lam) * criterion(logits, yb)
                    ) if used else criterion(logits, y)

                loss.backward()
                optimizer.second_step(zero_grad=True)

                loss_sum += loss.item() * x.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)

            tr_acc = correct / total
            tr_loss = loss_sum / total
            val = evaluate_classifier(model, dl_val, criterion, device, tta_mode=tta_mode)
            scheduler.step()

            fold_history["train_loss"].append(tr_loss)
            fold_history["val_loss"].append(val["loss"])
            fold_history["train_acc"].append(tr_acc)
            fold_history["val_acc"].append(val["acc"])
            fold_history["val_precision"].append(val["precision"])
            fold_history["val_recall"].append(val["recall"])
            fold_history["val_f1"].append(val["f1"])
            fold_history["lr"].append(optimizer.base_optimizer.param_groups[0]["lr"])

            _save_history_csv(fold_history, fold_dir, f"history_fold{fold:02d}.csv")

            print(
                f"[fold{fold:02d}] Ep {ep:02d}/{epochs} | "
                f"Train {tr_acc*100:.2f}%/{tr_loss:.4f} | "
                f"Val {val['acc']*100:.2f}%/{val['loss']:.4f} | "
                f"F1 {val['f1']*100:.1f}"
            )

            if use_swa and ep >= swa_begin_epoch:
                swa_model.update_parameters(model)

            improved = False
            if val["acc"] > best_acc:
                best_acc, best_acc_ep = val["acc"], ep
                torch.save(model.state_dict(), fold_dir / f"best_by_acc_fold{fold:02d}.pt")
                improved = True
            if val["loss"] < best_loss:
                best_loss, best_loss_ep = val["loss"], ep
                torch.save(model.state_dict(), fold_dir / f"best_by_loss_fold{fold:02d}.pt")
                improved = True

            if improved:
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[fold{fold:02d}] EarlyStopping at epoch {ep}.")
                    break

        if use_swa and swa_model is not None and swa_model.n_averaged > 0:
            update_bn(dl_train, swa_model, device=device)
            swa_val = evaluate_classifier(swa_model, dl_val, criterion, device, tta_mode=tta_mode)
            print(
                f"[fold{fold:02d}] SWA Eval -> "
                f"Acc {swa_val['acc']*100:.2f} | F1 {swa_val['f1']*100:.1f}"
            )
            if swa_val["acc"] >= best_acc:
                torch.save(swa_model.state_dict(), fold_dir / f"best_by_acc_fold{fold:02d}_SWA.pt")

        rows.append({
            "Fold": fold,
            "Method": f"CNN ({model_name})",
            "BestAcc": best_acc,
            "BestAccEpoch": best_acc_ep,
            "BestLoss": best_loss,
            "BestLossEpoch": best_loss_ep,
        })

    df = pd.DataFrame(rows)
    csv_path = out_dir / f"cv_summary_{model_name}_{k_folds}fold.csv"
    df.to_csv(csv_path, index=False)
    print(
        "\nMean Acc ± Std over folds:",
        f"{df['BestAcc'].mean()*100:.2f}% ± {df['BestAcc'].std(ddof=1)*100:.2f}%"
    )
    print("[SAVED]", csv_path)
    return df


# ---------------------------------------------------------------------
# Compatibility wrapper for old notebooks
# ---------------------------------------------------------------------
def run_cnn_kfold_training(
    data_dir,
    out_dir,
    model_name="effnet_b3",
    img_size=320,
    batch_size=16,
    epochs=50,
    lr=2e-4,
    weight_decay=5e-4,
    patience=7,
    seed=1337,
    k_folds=5,
    sanity_check=False,
    use_swa=True,
    swa_start=0.7,
    tta_mode="flip",
    **kwargs,
):
    """
    Backward-compatible wrapper for notebooks that call
    run_cnn_kfold_training(...).

    This wrapper calls run_kfold_training(...), while accepting
    older notebook arguments such as sanity_check.

    Parameters
    ----------
    sanity_check : bool
        If True, performs a lightweight path and dataset check before training.
    **kwargs
        Extra legacy keyword arguments are accepted and ignored with a warning.
    """
    from pathlib import Path

    data_dir = Path(data_dir)
    out_dir = Path(out_dir)

    if kwargs:
        print(f"[WARN] Ignoring unsupported legacy arguments: {sorted(kwargs.keys())}")

    if sanity_check:
        print("[sanity_check] data_dir:", data_dir)
        print("[sanity_check] out_dir :", out_dir)
        print("[sanity_check] data_dir exists:", data_dir.exists())

        if not data_dir.exists():
            raise FileNotFoundError(f"data_dir not found: {data_dir}")

        # Quick ImageFolder structure check: at least one class folder with files.
        class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
        if len(class_dirs) == 0:
            raise FileNotFoundError(
                f"No class subfolders found under data_dir: {data_dir}. "
                "Expected ImageFolder format: data_dir/class_name/images..."
            )

        print("[sanity_check] classes:", [p.name for p in class_dirs])
        n_files = sum(1 for cls in class_dirs for p in cls.iterdir() if p.is_file())
        print("[sanity_check] total files:", n_files)

        if n_files == 0:
            raise FileNotFoundError(f"No image files found under class folders in: {data_dir}")

    return run_kfold_training(
        data_dir=str(data_dir),
        out_dir=str(out_dir),
        model_name=model_name,
        img_size=img_size,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        seed=seed,
        k_folds=k_folds,
        use_swa=use_swa,
        swa_start=swa_start,
        tta_mode=tta_mode,
    )

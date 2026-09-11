import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def reset_dataset_dir(data_root: str, safe_keyword: Optional[str] = "dataset") -> None:
    """Remove and recreate a dataset directory safely.

    If safe_keyword is given, the path must contain that keyword to reduce accidental deletion risk.
    """
    path = Path(data_root)
    if safe_keyword and safe_keyword.lower() not in str(path).lower():
        raise ValueError(f"Refusing to delete path without safe keyword '{safe_keyword}': {path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def create_stratified_image_split(
    raw_root: str,
    data_root: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    img_exts: Optional[Iterable[str]] = None,
    clean_output: bool = True,
) -> pd.DataFrame:
    """Create train/val ImageFolder split by class with image-level stratification."""
    img_exts = {e.lower() for e in (img_exts or IMG_EXTS)}
    rng = random.Random(seed)
    raw_root = Path(raw_root)
    data_root = Path(data_root)

    if clean_output and data_root.exists():
        shutil.rmtree(data_root)
    for split in ["train", "val"]:
        (data_root / split).mkdir(parents=True, exist_ok=True)

    rows = []
    for cls in sorted(os.listdir(raw_root)):
        cls_dir = raw_root / cls
        if not cls_dir.is_dir():
            continue
        images = [p for p in cls_dir.iterdir() if p.suffix.lower() in img_exts]
        rng.shuffle(images)
        n_train = int(len(images) * train_ratio)
        split_map = {"train": images[:n_train], "val": images[n_train:]}

        for split, split_imgs in split_map.items():
            out_dir = data_root / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for img in split_imgs:
                shutil.copy2(img, out_dir / img.name)

        rows.append({
            "Class": cls,
            "Train": len(split_map["train"]),
            "Val": len(split_map["val"]),
            "Total": len(images),
            "Train Ratio (%)": 100 * len(split_map["train"]) / len(images) if images else 0.0,
        })

    return pd.DataFrame(rows)


def summarize_train_val_split(data_root: str) -> pd.DataFrame:
    """Summarize train/val counts for an ImageFolder dataset root."""
    train_dir = Path(data_root) / "train"
    val_dir = Path(data_root) / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("Expected data_root/train and data_root/val directories.")

    rows = []
    for cls in sorted(os.listdir(train_dir)):
        train_cls = train_dir / cls
        val_cls = val_dir / cls
        if not train_cls.is_dir():
            continue
        train_count = len([p for p in train_cls.iterdir() if p.is_file()])
        val_count = len([p for p in val_cls.iterdir() if p.is_file()]) if val_cls.exists() else 0
        total = train_count + val_count
        rows.append({
            "Class": cls,
            "Train": train_count,
            "Val": val_count,
            "Total": total,
            "Train Ratio (%)": 100 * train_count / total if total else 0.0,
        })
    return pd.DataFrame(rows)


class SafeAutoCrop:
    """Crop non-white content then pad to a square canvas."""
    def __init__(self, threshold: int = 245, fill=(255, 255, 255)):
        self.threshold = threshold
        self.fill = fill

    def __call__(self, img):
        im = np.array(img.convert("RGB"))
        mask = (im < self.threshold).any(axis=2)
        coords = np.argwhere(mask)
        if coords.size == 0:
            return img
        y0, x0 = coords.min(0)
        y1, x1 = coords.max(0) + 1
        cropped = Image.fromarray(im[y0:y1, x0:x1])
        w, h = cropped.size
        s = max(w, h)
        canvas = Image.new("RGB", (s, s), self.fill)
        canvas.paste(cropped, ((s - w) // 2, (s - h) // 2))
        return canvas


def build_transforms(
    img_size: int = 320,
    train: bool = True,
    autocrop: bool = True,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> T.Compose:
    """Build Raman image transforms for CNN training/evaluation."""
    ops = []
    if autocrop:
        ops.append(SafeAutoCrop())
    ops.append(T.Grayscale(3))

    if train:
        ops += [
            T.Resize(int(img_size * 1.08)),
            T.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
            T.RandomAffine(1, translate=(0.03, 0.04)),
            T.ColorJitter(0.05, 0.05),
        ]
    else:
        ops += [
            T.Resize(int(img_size * 1.08)),
            T.CenterCrop(img_size),
        ]

    ops += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(ops)


def build_imagefolder_datasets(
    train_dir: str,
    val_dir: str,
    img_size: int = 320,
    autocrop: bool = True,
):
    train_ds = datasets.ImageFolder(train_dir, transform=build_transforms(img_size, True, autocrop))
    val_ds = datasets.ImageFolder(val_dir, transform=build_transforms(img_size, False, autocrop))
    if train_ds.classes != val_ds.classes:
        raise ValueError("Train/val class names do not match.")
    return train_ds, val_ds


def build_balanced_sampler(targets: Sequence[int]) -> WeightedRandomSampler:
    """Create inverse-frequency WeightedRandomSampler."""
    targets = np.asarray(targets, dtype=np.int64)
    class_counts = np.bincount(targets)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[targets]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def build_dataloaders(
    train_dir: str,
    val_dir: str,
    img_size: int = 320,
    batch_size: int = 16,
    num_workers: int = 0,
    balanced: bool = True,
    autocrop: bool = True,
):
    """Build train/val ImageFolder dataloaders."""
    train_ds, val_ds = build_imagefolder_datasets(train_dir, val_dir, img_size, autocrop)
    sampler = build_balanced_sampler(train_ds.targets) if balanced else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_ds.classes


def make_loader_from_indices(
    data_dir: str,
    indices: Sequence[int],
    img_size: int = 320,
    batch_size: int = 16,
    train: bool = True,
    labels: Optional[Sequence[int]] = None,
    balanced: bool = True,
    num_workers: int = 0,
):
    """Create a loader from ImageFolder subset indices, optionally class-balanced."""
    base = datasets.ImageFolder(data_dir, transform=build_transforms(img_size, train=train))
    subset = Subset(base, indices)
    sampler = None
    if train and balanced:
        if labels is None:
            labels = np.array([base.samples[i][1] for i in indices], dtype=np.int64)
        sampler = build_balanced_sampler(labels)
    return DataLoader(
        subset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(train and sampler is None),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_random_dataset_split(
    raw_root,
    data_root,
    train_ratio=0.8,
    seed=None,
    clean=True,
    img_exts=None,
):
    """
    Create a random train/val ImageFolder split from raw images.

    Parameters
    ----------
    raw_root : str or Path
        Folder containing class subfolders.
    data_root : str or Path
        Output folder. Creates data_root/train and data_root/val.
    train_ratio : float
        Fraction of images used for training.
    seed : int or None
        If None, a new random seed is generated each run.
        If int, the split is reproducible.
    clean : bool
        If True, delete old data_root before creating the new split.
    img_exts : iterable or None
        Image file extensions. Defaults to IMG_EXTS.

    Returns
    -------
    summary : pandas.DataFrame
        Per-class train/val counts.
    seed : int
        Seed actually used.
    """
    import time

    raw_root = Path(raw_root)
    data_root = Path(data_root)
    img_exts = {e.lower() for e in (img_exts or IMG_EXTS)}

    seed = int(time.time_ns() % 1_000_000_000) if seed is None else int(seed)
    rng = random.Random(seed)

    if clean and data_root.exists():
        shutil.rmtree(data_root)

    for split in ["train", "val"]:
        (data_root / split).mkdir(parents=True, exist_ok=True)

    rows = []

    for cls_dir in sorted([p for p in raw_root.iterdir() if p.is_dir()]):
        images = sorted([p for p in cls_dir.iterdir() if p.suffix.lower() in img_exts])
        rng.shuffle(images)

        n_train = int(len(images) * train_ratio)
        split_map = {
            "train": images[:n_train],
            "val": images[n_train:],
        }

        for split, split_imgs in split_map.items():
            out_dir = data_root / split / cls_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)

            for img in split_imgs:
                shutil.copy2(img, out_dir / img.name)

        rows.append({
            "Class": cls_dir.name,
            "Train": len(split_map["train"]),
            "Val": len(split_map["val"]),
            "Total": len(images),
            "Train Ratio (%)": 100 * len(split_map["train"]) / len(images) if images else 0.0,
        })

    return pd.DataFrame(rows), seed

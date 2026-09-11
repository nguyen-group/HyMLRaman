import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


def build_feature_transform(img_size: int = 320):
    return transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_effnet_b3_feature_extractor(pretrained: bool = True):
    weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b3(weights=weights)
    model.classifier = torch.nn.Identity()
    return model


def load_finetuned_effnet_b3_feature_extractor(checkpoint: str, num_classes: int, dropout: float = 0.6, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b3(weights=None)
    in_f = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(torch.nn.Dropout(dropout), torch.nn.Linear(in_f, num_classes))
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.classifier = torch.nn.Identity()
    return model


@torch.no_grad()
def extract_features(model, dataloader, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    feats, labels = [], []
    for imgs, lbls in tqdm(dataloader, desc="Extracting"):
        imgs = imgs.to(device)
        out = model(imgs)
        feats.append(out.cpu().numpy())
        labels.append(lbls.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def extract_cnn_embeddings(
    data_root: str,
    out_dir: str,
    checkpoint: Optional[str] = None,
    img_size: int = 320,
    batch_size: int = 32,
    finetuned: bool = False,
):
    """Extract train/val EfficientNet-B3 embeddings and save .npy outputs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = build_feature_transform(img_size)
    train_ds = datasets.ImageFolder(Path(data_root) / "train", transform=tf)
    val_ds = datasets.ImageFolder(Path(data_root) / "val", transform=tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    if finetuned:
        if checkpoint is None:
            raise ValueError("checkpoint is required when finetuned=True")
        model = load_finetuned_effnet_b3_feature_extractor(checkpoint, len(train_ds.classes), device=device)
        suffix = "_finetuned"
    else:
        model = build_effnet_b3_feature_extractor(pretrained=True)
        suffix = ""

    train_feats, train_labels = extract_features(model, train_loader, device)
    val_feats, val_labels = extract_features(model, val_loader, device)

    np.save(out_dir / f"embeddings_train{suffix}.npy", train_feats)
    np.save(out_dir / f"labels_train{suffix}.npy", train_labels)
    np.save(out_dir / f"embeddings_val{suffix}.npy", val_feats)
    np.save(out_dir / f"labels_val{suffix}.npy", val_labels)

    return {
        "train_feats": train_feats,
        "train_labels": train_labels,
        "val_feats": val_feats,
        "val_labels": val_labels,
        "classes": train_ds.classes,
    }


def normalize_path(p: str) -> str:
    return os.path.normpath(str(p)).replace("\\", "/").lower()


def collect_path_endings(root_dir: str):
    endings = set()
    for dirpath, _, fnames in os.walk(root_dir):
        for fn in fnames:
            full = normalize_path(os.path.join(dirpath, fn))
            endings.add(os.path.basename(full))
            parts = full.split("/")
            if len(parts) >= 2:
                endings.add("/".join(parts[-2:]))
            if len(parts) >= 3:
                endings.add("/".join(parts[-3:]))
    return endings


def match_embedding_indices_to_split(filenames_json: str, train_dir: str, val_dir: str):
    files = [normalize_path(p) for p in json.load(open(filenames_json, "r", encoding="utf-8"))]
    by_basename = defaultdict(list)
    for i, p in enumerate(files):
        by_basename[os.path.basename(p)].append(i)

    def lookup_indices(endings_set):
        idx = set()
        for ending in endings_set:
            candidates = by_basename.get(os.path.basename(ending), [])
            if not candidates:
                continue
            if len(candidates) == 1:
                idx.add(candidates[0])
            else:
                for i in candidates:
                    if files[i].endswith(ending):
                        idx.add(i)
        return sorted(idx)

    idx_train = lookup_indices(collect_path_endings(train_dir))
    idx_val = lookup_indices(collect_path_endings(val_dir))
    overlap = sorted(set(idx_train) & set(idx_val))
    return idx_train, idx_val, overlap

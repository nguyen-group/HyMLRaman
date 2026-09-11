import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np


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
    return idx_train, idx_val, overlap, files


def save_split_indices(idx_train: Sequence[int], idx_val: Sequence[int], out_dir: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump([int(i) for i in idx_train], open(out_dir / "train_idx.json", "w"))
    json.dump([int(i) for i in idx_val], open(out_dir / "val_idx.json", "w"))


def check_embedding_leakage_cosine(
    features_path: str,
    filenames_path: str,
    idx_train: Sequence[int],
    idx_val: Sequence[int],
    cos_thresh: float = 0.9990,
    topk_per_val: int = 1,
    print_limit: int = 50,
    block_size: int = 2048,
):
    X = np.load(features_path).astype(np.float32)
    files = json.load(open(filenames_path, "r", encoding="utf-8"))
    Xtr = X[list(idx_train)].copy()
    Xva = X[list(idx_val)].copy()
    Xtr /= np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-8
    Xva /= np.linalg.norm(Xva, axis=1, keepdims=True) + 1e-8

    warn_pairs = []
    for s in range(0, Xva.shape[0], block_size):
        e = min(s + block_size, Xva.shape[0])
        S = Xva[s:e] @ Xtr.T
        kth = min(topk_per_val, S.shape[1] - 1)
        top_idx = np.argpartition(-S, kth=kth, axis=1)[:, :topk_per_val]
        for ii, row in enumerate(S):
            for jj in top_idx[ii]:
                sim = float(row[jj])
                if sim >= cos_thresh:
                    vi = idx_val[s + ii]
                    ti = idx_train[jj]
                    warn_pairs.append((sim, files[vi], files[ti]))
                    if len(warn_pairs) >= print_limit:
                        return warn_pairs
    return warn_pairs

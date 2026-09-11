"""HyRaman plotting and 10-fold ML evaluation utilities.

This module is intentionally centralized so notebooks do not need separate
`utils_ml_xgb_patch.py` / `utils_ml_svm_patch.py` files.  All figures use the
same paper-style font, visible major tick marks, and no minor ticks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MultipleLocator, NullLocator, PercentFormatter
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC


# =============================================================================
# Global paper style helpers
# =============================================================================

PAPER_FONT = "Arial"
FALLBACK_SERIF = ["Arial", "Liberation Sans", "DejaVu Sans", "Bitstream Vera Sans", "sans-serif"]

# Canonical drug-class labels used throughout the manuscript and figures.
# The aliases preserve compatibility with older JSON/CSV outputs.
CANONICAL_CLASS_ORDER = ("AMX", "CHL", "CIP", "IBU", "PAR", "TET")

CLASS_NAME_ALIASES = {
    "amx": "AMX",
    "amoxicillin": "AMX",
    "chlor": "CHL",
    "chl": "CHL",
    "chloramphenicol": "CHL",
    "cpf": "CIP",
    "cip": "CIP",
    "ciprofloxacin": "CIP",
    "ibup": "IBU",
    "ibu": "IBU",
    "ibuprofen": "IBU",
    "para": "PAR",
    "par": "PAR",
    "paracetamol": "PAR",
    "tetra": "TET",
    "tet": "TET",
    "tetracycline": "TET",
    "micro": "micro",
    "micro-average": "micro",
    "micro_average": "micro",
}


def standardize_class_name(name) -> str:
    """Return the canonical class label while accepting legacy aliases."""
    value = str(name).strip()
    return CLASS_NAME_ALIASES.get(value.lower(), value)


def standardize_class_names(names: Sequence[str]):
    """Standardize a sequence without changing its class order."""
    return [standardize_class_name(name) for name in names]


def display_class_name(name) -> str:
    """Return the manuscript display label for a class."""
    canonical = standardize_class_name(name)
    return "micro-average" if canonical == "micro" else canonical


def set_paper_style(font_size: int = 18, font_family: str = "serif") -> None:
    """Set global paper-style Matplotlib rcParams."""
    plt.rcParams.update({
        "font.family": font_family,
        "font.serif": FALLBACK_SERIF,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def paper_rc(
    base: int = 18,
    axes: int = 22,
    ticks: int = 20,
    legend: int = 16,
    title: int = 22,
):
    """Return a reusable rc_context dict for paper figures."""
    return {
        "font.family": "serif",
        "font.serif": FALLBACK_SERIF,
        "font.size": base,
        "axes.titlesize": title,
        "axes.labelsize": axes,
        "xtick.labelsize": ticks,
        "ytick.labelsize": ticks,
        "legend.fontsize": legend,
        "axes.linewidth": 1.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }


def apply_visible_axis_ticks(
    ax,
    x_major: Optional[float] = None,
    y_major: Optional[float] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    x_rotation: float = 0,
    y_rotation: float = 0,
    tick_length: float = 5.0,
    tick_width: float = 1.05,
    labelsize: Optional[int] = None,
    direction: str = "out",
    top: bool = False,
    right: bool = False,
):
    """Apply consistent visible major ticks to normal x/y axes.

    Policy used throughout the project:
    - major ticks only;
    - no minor ticks;
    - outward ticks;
    - both x and y tick marks are visible.
    """
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if x_major is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_major))
    if y_major is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_major))

    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(
        axis="x", which="major", bottom=True, top=top,
        direction=direction, length=tick_length, width=tick_width,
        labelsize=labelsize, rotation=x_rotation, pad=6,
    )
    ax.tick_params(
        axis="y", which="major", left=True, right=right,
        direction=direction, length=tick_length, width=tick_width,
        labelsize=labelsize, rotation=y_rotation, pad=6,
    )
    ax.tick_params(axis="both", which="minor", bottom=False, top=False, left=False, right=False, length=0)
    return ax


def apply_cm_axis_ticks(
    ax,
    class_names: Sequence[str],
    x_rotation: float = 45,
    tick_length: float = 5.0,
    tick_width: float = 1.05,
    labelsize: Optional[int] = None,
    use_half_positions: bool = True,
):
    """Apply visible tick marks for heatmap/confusion-matrix axes."""
    n = len(class_names)
    pos = np.arange(n) + 0.5 if use_half_positions else np.arange(n)
    ax.set_xticks(pos)
    ax.set_yticks(pos)
    ax.set_xticklabels(class_names, rotation=x_rotation, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(class_names, rotation=0, va="center")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(
        axis="x", which="major", bottom=True, top=False,
        direction="out", length=tick_length, width=tick_width,
        labelsize=labelsize, pad=2,
    )
    ax.tick_params(
        axis="y", which="major", left=True, right=False,
        direction="out", length=tick_length, width=tick_width,
        labelsize=labelsize, pad=2,
    )
    ax.tick_params(axis="both", which="minor", bottom=False, top=False, left=False, right=False, length=0)
    return ax


def style_spines(ax, visible_sides=("left", "bottom", "top", "right"), linewidth: float = 1.05, color: str = "#333333"):
    for side in ["left", "bottom", "top", "right"]:
        ax.spines[side].set_visible(side in visible_sides)
        if side in visible_sides:
            ax.spines[side].set_linewidth(linewidth)
            ax.spines[side].set_color(color)
    return ax


def _save_figure(fig, out_png=None, out_pdf=None, dpi: int = 700, pad_inches: float = 0.03):
    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    if out_pdf is not None:
        out_pdf = Path(out_pdf)
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=pad_inches)


# =============================================================================
# Data loading helpers
# =============================================================================


def _norm_method_name(x) -> str:
    return (
        str(x)
        .replace("ORIG-", "")
        .replace("XGBoost", "XGB")
        .replace("XGBOOST", "XGB")
        .strip()
        .upper()
    )


def _load_class_names(root: Path, class_name="classes.json", fallback_class_name="class_to_idx.json", y=None):
    class_path = root / class_name
    fallback_path = root / fallback_class_name
    if class_path.exists():
        with open(class_path, "r", encoding="utf-8") as f:
            names = json.load(f)
    elif fallback_path.exists():
        with open(fallback_path, "r", encoding="utf-8") as f:
            c2i = json.load(f)
        names = [None] * len(c2i)
        for k, v in c2i.items():
            names[int(v)] = k
    elif y is not None:
        names = [str(i) for i in sorted(np.unique(y))]
    else:
        raise FileNotFoundError(f"No class file found under {root}")
    return standardize_class_names(names)


def _load_features_labels(root, feat_name="features_b3_1536.npy", label_name="labels_b3.npy"):
    root = Path(root)
    feat_path = root / feat_name
    label_path = root / label_name
    if not feat_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feat_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")
    X = np.load(feat_path).astype(np.float32)
    y = np.load(label_path).astype(np.int64).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X/y size mismatch: X={X.shape}, y={y.shape}")
    return X, y


def _load_cm_csv(csv_path, class_names: Sequence[str]):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Confusion matrix CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, index_col=0)
    class_names = standardize_class_names(class_names)
    df.index = [standardize_class_name(x) for x in df.index]
    df.columns = [standardize_class_name(x) for x in df.columns]

    if set(class_names).issubset(set(df.index)) and set(class_names).issubset(set(df.columns)):
        cm = df.loc[list(class_names), list(class_names)].to_numpy(dtype=float)
    elif df.shape == (len(class_names), len(class_names)):
        cm = df.to_numpy(dtype=float)
    else:
        df2 = pd.read_csv(csv_path)
        df2 = df2.loc[:, ~df2.columns.astype(str).str.contains("^Unnamed")]
        if df2.shape == (len(class_names), len(class_names)):
            cm = df2.to_numpy(dtype=float)
        else:
            raise KeyError(
                f"Cannot align CM CSV with classes.\nCSV: {csv_path}\n"
                f"Expected classes: {class_names}\n"
                f"Found index: {list(df.index)[:10]}\n"
                f"Found columns: {list(df.columns)[:10]}\n"
                f"Shape after index_col=0: {df.shape}; shape without unnamed columns: {df2.shape}"
            )

    row_sums = cm.sum(axis=1, keepdims=True)
    if np.allclose(row_sums, 100, atol=2):
        return cm
    if np.allclose(row_sums, 1.0, atol=0.05):
        return cm * 100.0
    return np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0


# Backward-compatible public alias
load_cm_csv = lambda csv_path, class_names: _load_cm_csv(csv_path, class_names) / 100.0


# =============================================================================
# Generic plotting primitives
# =============================================================================


def warm_deep_cmap():
    return LinearSegmentedColormap.from_list(
        "warm_deep_cmap",
        ["#fffdf0", "#fff3b0", "#fdd067", "#fca03b", "#f03b20", "#d7191c", "#a50026"],
    )


def cm_orange_cmap():
    return LinearSegmentedColormap.from_list(
        "paper_dark_orange",
        ["#fffdf8", "#fbe8c8", "#f3c98b", "#e59644", "#c8641a"],
    )


def sort_classes(keys, preferred_order=CANONICAL_CLASS_ORDER):
    """Sort canonical or legacy class labels in the manuscript class order."""
    canonical_to_original = {}
    for key in keys:
        canonical = standardize_class_name(key)
        canonical_to_original.setdefault(canonical, key)

    preferred = [standardize_class_name(x) for x in preferred_order]
    ordered_names = [name for name in preferred if name in canonical_to_original]
    others = sorted(
        name for name in canonical_to_original
        if name not in ordered_names and name != "micro"
    )
    if "micro" in canonical_to_original:
        ordered_names.append("micro")

    return [canonical_to_original[name] for name in ordered_names + others]


def _plot_cm_percent(
    cm_pct,
    class_names,
    title,
    out_png=None,
    out_pdf=None,
    show=True,
    figsize=(6, 5),
    dpi=700,
    title_size=18,
    label_size=18,
    tick_size=16,
    annot_size=16,
    cbar=True,
    cbar_label="Recall (%)",
):
    class_names = standardize_class_names(class_names)
    with plt.rc_context(paper_rc(base=20, axes=label_size, ticks=tick_size, legend=16, title=title_size)):
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        im = ax.imshow(cm_pct, cmap=cm_orange_cmap(), vmin=0, vmax=100, interpolation="nearest")
        ax.set_title(title, fontsize=title_size, pad=8)
        ax.set_xlabel("Predicted label", fontsize=label_size, labelpad=8)
        ax.set_ylabel("True label", fontsize=label_size, labelpad=8)

        n = len(class_names)
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(class_names, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticklabels(class_names, rotation=0, va="center")
        apply_visible_axis_ticks(ax, tick_length=5.0, tick_width=1.05, labelsize=tick_size)

        for r in range(n):
            for c in range(n):
                val = cm_pct[r, c]
                ax.text(c, r, f"{val:.1f}", ha="center", va="center", fontsize=annot_size,
                        color="white" if val >= 75 else "#222222")
        style_spines(ax)

        if cbar:
            cb = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.045)
            cb.set_label(cbar_label, fontsize=label_size)
            cb.ax.tick_params(which="major", direction="out", length=5.0, width=1.05, labelsize=tick_size)
            cb.ax.yaxis.set_minor_locator(NullLocator())

        fig.tight_layout(pad=0.45)
        _save_figure(fig, out_png, out_pdf, dpi=dpi)
        if show:
            plt.show()
        return fig, ax


def _plot_roc_curves(
    fpr,
    tpr,
    roc_auc,
    class_names,
    title,
    out_png=None,
    out_pdf=None,
    xlim=(0.0, 0.30),
    ylim=(0.80, 1.00),
    show=True,
    figsize=(6, 5),
    dpi=700,
    title_size=18,
    label_size=18,
    tick_size=16,
    legend_size=13,
):
    class_names = standardize_class_names(class_names)
    with plt.rc_context(paper_rc(base=20, axes=label_size, ticks=tick_size, legend=legend_size, title=title_size)):
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#8F63C2", "#9D755D", "#72B7B2", "#B279A2"]
        for i, name in enumerate(class_names):
            ax.plot(fpr[i], tpr[i], lw=1.65, color=colors[i % len(colors)], label=f"{display_class_name(name)} (AUC={roc_auc[i]:.3f})")
        if "micro" in fpr:
            ax.plot(fpr["micro"], tpr["micro"], lw=2.0, linestyle="--", color="black",
                    label=f"micro-average (AUC={roc_auc['micro']:.3f})")
        ax.plot([0, 1], [0, 1], linestyle=":", lw=1.0, color="#888888")

        ax.set_xlabel("False Positive Rate", fontsize=label_size, labelpad=8)
        ax.set_ylabel("True Positive Rate", fontsize=label_size, labelpad=8)
        ax.set_title(title, fontsize=title_size, pad=8)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        apply_visible_axis_ticks(
            ax,
            x_major=0.05 if xlim[1] <= 0.35 else 0.2,
            y_major=0.05 if ylim[0] >= 0.75 else 0.1,
            xlim=xlim,
            ylim=ylim,
            tick_length=5.0,
            tick_width=1.05,
            labelsize=tick_size,
        )
        style_spines(ax)
        ax.grid(False)
        ax.legend(loc="lower right", frameon=False, handlelength=1.5, handletextpad=0.45,
                  labelspacing=0.18, borderaxespad=0.25, fontsize=legend_size)
        fig.tight_layout(pad=0.45)
        _save_figure(fig, out_png, out_pdf, dpi=dpi)
        if show:
            plt.show()
        return fig, ax


# =============================================================================
# Bar charts / summary figures
# =============================================================================


def plot_class_distribution(class_counts: Dict[str, int], save_path: Optional[str] = None):
    raw_labels, values = zip(*sorted(class_counts.items(), key=lambda x: x[1], reverse=True))
    labels = [display_class_name(label) for label in raw_labels]
    values = np.asarray(values)
    total_samples = int(values.sum())
    with plt.rc_context(paper_rc(base=18, axes=22, ticks=20, legend=16, title=20)):
        fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
        bars = ax.bar(labels, values, color="#1F3A5F", edgecolor="black", linewidth=1.2)
        y_max = values.max()
        ax.set_ylim(0, y_max * 1.15)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + y_max * 0.03, f"{val}",
                    ha="center", va="bottom", fontsize=18, fontweight="bold")
        ax.set_xlabel("Class", fontweight="bold")
        ax.set_ylabel("Number of Images", fontweight="bold")
        ax.set_title(f"Class Distribution of Raman Spectrum Images\n(Total samples = {total_samples})")
        apply_visible_axis_ticks(ax, tick_length=5.0, tick_width=1.05)
        ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.25)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=600, bbox_inches="tight")
        return fig, ax



def plot_class_distribution_si(
    class_counts: Dict[str, int],
    save_path: Optional[str] = None,
    order: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (6.4, 3.8),
    color: str = "#24476F",
    show: bool = True,
):
    """Paper/SI-style bar chart for Raman spectral image class distribution.

    This figure uses Arial locally, with normal-weight axis labels/ticks.
    Value annotations above bars remain bold for readability.
    """
    order = list(order) if order is not None else list(class_counts.keys())
    labels = [display_class_name(k) for k in order]
    values = np.array([class_counts[k] for k in order], dtype=float)
    x = np.arange(len(labels))

    rc = paper_rc(base=18, axes=22, ticks=20, legend=16, title=20)
    rc.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.weight": "normal",
        "axes.labelweight": "normal",
        "axes.titleweight": "normal",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=figsize, dpi=300)

        bars = ax.bar(
            x,
            values,
            width=0.68,
            color=color,
            edgecolor="black",
            linewidth=0.9,
            zorder=3,
        )

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 3,
                f"{int(val)}",
                ha="center",
                va="bottom",
                fontsize=16,
                fontfamily="Arial",
                fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontfamily="Arial", fontweight="normal")
        ax.set_ylabel("Number of samples", fontfamily="Arial", fontweight="normal")
        ax.set_xlabel("Class", fontfamily="Arial", fontweight="normal")

        ymax = float(values.max() * 1.18)
        ax.set_ylim(0, ymax)
        ax.set_xlim(-0.62, len(labels) - 0.35)

        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.22, zorder=0)
        ax.set_axisbelow(True)

        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_visible(False)

        arrow_kw = dict(
            arrowstyle="-|>",
            lw=1.15,
            color="black",
            mutation_scale=12,
            shrinkA=0,
            shrinkB=0,
        )
        ax.annotate(
            "",
            xy=(len(labels) - 0.25, 0),
            xytext=(-0.62, 0),
            arrowprops=arrow_kw,
            annotation_clip=False,
        )
        ax.annotate(
            "",
            xy=(-0.62, ymax * 1.01),
            xytext=(-0.62, 0),
            arrowprops=arrow_kw,
            annotation_clip=False,
        )

        ax.tick_params(axis="both", direction="out", length=4.5, width=1.0)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontfamily("Arial")
            tick.set_fontweight("normal")

        fig.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            _save_figure(fig, save_path, save_path.with_suffix(".pdf"), dpi=700)

        if show:
            plt.show()
        return fig, ax

def plot_accuracy_f1_bar_from_csv(
    csv_file,
    methods=("RF", "LR", "XGB", "SVM", "ANN", "KNN"),
    save_path=None,
    show=True,
    dpi=600,
    figsize=(6.4, 3.9),
    ytick_step=0.5,
):
    """Compact paper-style grouped bar chart for Accuracy and F1-score."""
    csv_file = Path(csv_file)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    df = pd.read_csv(csv_file)
    required_cols = {"Method", "Accuracy", "F1"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV thiếu cột bắt buộc: {sorted(missing_cols)}")

    df["Method"] = df["Method"].map(_norm_method_name)
    methods = [_norm_method_name(m) for m in methods]
    plot_df = df.set_index("Method").reindex(methods).dropna(subset=["Accuracy", "F1"], how="any").reset_index()
    if plot_df.empty:
        raise ValueError(
            "Không tìm thấy method hợp lệ nào trong CSV.\n"
            f"Requested: {methods}\nAvailable: {sorted(df['Method'].unique().tolist())}"
        )

    accuracy = plot_df["Accuracy"].to_numpy(float)
    f1 = plot_df["F1"].to_numpy(float)
    if np.nanmax(accuracy) <= 1.5:
        accuracy *= 100.0
    if np.nanmax(f1) <= 1.5:
        f1 *= 100.0

    labels = ["XGBoost" if m == "XGB" else m for m in plot_df["Method"].tolist()]
    x = np.arange(len(labels)) * 0.86
    width = 0.31

    with plt.rc_context(paper_rc(base=19, axes=21, ticks=20, legend=19, title=20)):
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        ax.bar(x - width / 2, accuracy, width, color="#274472", edgecolor="#1A1A1A", linewidth=0.85, label="Accuracy", zorder=3)
        ax.bar(x + width / 2, f1, width, color="#B7BEC8", edgecolor="#1A1A1A", linewidth=0.85, label="F1-score", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Score (%)", labelpad=4)
        ax.set_xlabel("Machine Learning Model", fontweight="bold", labelpad=3)

        vals = np.concatenate([accuracy, f1])
        ymin = np.floor((np.nanmin(vals) - 0.15) * 2) / 2
        ymax = np.ceil((np.nanmax(vals) + 0.15) * 2) / 2
        if ymax <= ymin:
            ymax = ymin + 1.0
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_locator(MultipleLocator(ytick_step))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        apply_visible_axis_ticks(ax, tick_length=5.0, tick_width=1.05)
        style_spines(ax)
        ax.grid(axis="y", which="major", linestyle="--", linewidth=0.55, alpha=0.14)
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, loc="upper right", handlelength=1.6)
        fig.tight_layout(pad=0.5)
        if save_path is not None:
            save_path = Path(save_path)
            _save_figure(fig, save_path, save_path.with_suffix(".pdf") if save_path.suffix.lower() in [".png", ".jpg", ".jpeg"] else None, dpi=dpi)
        if show:
            plt.show()

    plot_df["Accuracy_plot"] = accuracy
    plot_df["F1_plot"] = f1
    return fig, ax, plot_df


def plot_grouped_performance_from_csv(
    csv_file,
    out_png=None,
    out_pdf=None,
    methods=("RF", "LR", "XGB", "SVM", "ANN", "KNN"),
    show=True,
    dpi=600,
    figsize=(6.4, 3.9),
    ytick_step=0.5,
):
    """Backward-compatible wrapper for Accuracy/F1 grouped bar chart."""
    fig, ax, plot_df = plot_accuracy_f1_bar_from_csv(
        csv_file=csv_file,
        methods=methods,
        save_path=out_png,
        show=show,
        dpi=dpi,
        figsize=figsize,
        ytick_step=ytick_step,
    )
    if out_pdf is not None:
        Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight")
    return fig, ax, plot_df


def plot_performance_grouped_bar_from_csv(
    csv_path: str,
    out_png=None,
    out_pdf=None,
    models=("CNN", "ANN", "RF", "SVM", "KNN", "LR", "XGB"),
    figsize=(11.2, 5.2),
    dpi=300,
):
    """Plot Accuracy / Precision / Recall grouped bar chart from CSV."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = {"Method", "Accuracy", "Precision", "Recall"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV thiếu cột bắt buộc: {sorted(missing_cols)}")

    df["Method"] = df["Method"].map(_norm_method_name)
    vals = {m: [] for m in ["Accuracy", "Precision", "Recall"]}
    for m in models:
        row = df[df["Method"] == _norm_method_name(m)]
        for metric in vals:
            vals[metric].append(row[metric].values[0] if len(row) else np.nan)

    acc = np.asarray(vals["Accuracy"], dtype=float)
    pre = np.asarray(vals["Precision"], dtype=float)
    rec = np.asarray(vals["Recall"], dtype=float)
    if np.nanmax(acc) > 1.5: acc = acc / 100.0
    if np.nanmax(pre) > 1.5: pre = pre / 100.0
    if np.nanmax(rec) > 1.5: rec = rec / 100.0

    x = np.arange(len(models)) * 0.78
    width = 0.16
    with plt.rc_context(paper_rc(base=22, axes=25, ticks=23, legend=23, title=24)):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        colors = {"Accuracy": "#4C9A2A", "Precision": "#1F6B1B", "Recall": "#0B5D4B"}
        ax.bar(x - width, acc, width, label="Accuracy", color=colors["Accuracy"], edgecolor="black", linewidth=0.75, zorder=3)
        ax.bar(x, pre, width, label="Precision", color=colors["Precision"], edgecolor="black", linewidth=0.75, zorder=3)
        ax.bar(x + width, rec, width, label="Recall", color=colors["Recall"], edgecolor="black", linewidth=0.75, zorder=3)
        ax.set_ylim(0.80, 1.00)
        ax.set_ylabel("Score", labelpad=8)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=10)
        apply_visible_axis_ticks(ax, tick_length=5.5, tick_width=1.1)
        ax.grid(axis="y", which="major", linestyle="--", linewidth=0.65, alpha=0.18)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        style_spines(ax, visible_sides=("left", "bottom"), linewidth=1.05)
        ax.set_xlim(x[0] - 0.42, x[-1] + 0.42)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=3, frameon=False, handlelength=1.9, columnspacing=1.8)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        _save_figure(fig, out_png, out_pdf, dpi=900)
    return fig, ax


def plot_model_ranking_bar(labels, scores, deep_models=None, best_label=None, save_png: Optional[str] = None, save_pdf: Optional[str] = None):
    deep_models = set(deep_models or [])
    best_label = best_label or labels[int(np.argmax(scores))]
    with plt.rc_context(paper_rc(base=18, axes=22, ticks=18, legend=16, title=22)):
        fig, ax = plt.subplots(figsize=(10.6, 6.0), dpi=300)
        best_color, ml_color, dl_color, edge = "#C65A1E", "#2F5D95", "#B7C3D0", "#222222"
        colors = [best_color if lab == best_label else dl_color if lab in deep_models else ml_color for lab in labels]
        bars = ax.barh(labels, scores, color=colors, edgecolor=edge, linewidth=0.9, height=0.72)
        ax.invert_yaxis()
        for bar, val, lab in zip(bars, scores, labels):
            ax.text(val + 0.18, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", ha="left", fontsize=16,
                    fontweight="bold" if lab == best_label else "semibold")
        ax.set_xlabel("Top-1 Accuracy (%)", fontweight="bold", labelpad=12)
        ax.set_ylabel("Model / Classifier", fontweight="bold", labelpad=12)
        apply_visible_axis_ticks(ax, tick_length=5.0, tick_width=1.05)
        ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.20)
        handles = [Patch(facecolor=best_color, edgecolor=edge, label=f"Best model ({best_label})"),
                   Patch(facecolor=ml_color, edgecolor=edge, label="Classical ML"),
                   Patch(facecolor=dl_color, edgecolor=edge, label="Deep learning")]
        ax.legend(handles=handles, frameon=False, loc="lower right")
        fig.tight_layout()
        _save_figure(fig, save_png, save_pdf, dpi=600)
        return fig, ax


# =============================================================================
# Confusion matrix / ROC from CSV
# =============================================================================


def load_roc_csv(csv_path: str):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.loc[:, ~df.columns.astype(str).str.contains("^unnamed")]
    required = {"class", "fpr", "tpr"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain class/fpr/tpr: {csv_path}")
    df["class"] = df["class"].map(standardize_class_name)
    df["fpr"] = pd.to_numeric(df["fpr"], errors="coerce")
    df["tpr"] = pd.to_numeric(df["tpr"], errors="coerce")
    if "auc" in df.columns:
        df["auc"] = pd.to_numeric(df["auc"], errors="coerce")
    roc_dict, auc_dict = {}, {}
    for cls, sub in df.groupby("class"):
        sub = sub.dropna(subset=["fpr", "tpr"]).sort_values("fpr")
        if len(sub) == 0:
            continue
        roc_dict[cls] = (sub["fpr"].to_numpy(float), sub["tpr"].to_numpy(float))
        if "auc" in sub.columns:
            vals = sub["auc"].dropna().unique()
            if len(vals) > 0:
                auc_dict[cls] = float(vals[0])
    return roc_dict, auc_dict


def draw_roc_panel(ax, roc_dict, auc_dict, title, xlim=(0.0, 0.30), ylim=(0.80, 1.00), fontsize=13):
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#8F63C2", "#9D755D"]
    color_id = 0
    for cls in sort_classes(roc_dict.keys()):
        fpr, tpr = roc_dict[cls]
        auc_text = f"{auc_dict[cls]:.3f}" if cls in auc_dict else "NA"
        canonical = standardize_class_name(cls)
        if canonical == "micro":
            ax.plot(fpr, tpr, lw=1.65, linestyle=(0, (4, 2)), color="black", label=f"micro-average (AUC={auc_text})", zorder=6)
        else:
            ax.plot(fpr, tpr, lw=1.45, color=colors[color_id % len(colors)], label=f"{display_class_name(canonical)} (AUC={auc_text})")
            color_id += 1
    ax.plot([0, 1], [0, 1], linestyle=":", lw=1.0, color="#9a9a9a", zorder=1)
    ax.set_box_aspect(0.80)
    ax.set_xlabel("False Positive Rate", fontsize=17, labelpad=5)
    ax.set_ylabel("True Positive Rate", fontsize=17, labelpad=5)
    ax.set_title(title, fontsize=18, pad=7)
    apply_visible_axis_ticks(
        ax,
        x_major=0.05 if xlim[1] <= 0.35 else 0.2,
        y_major=0.05 if ylim[0] >= 0.75 else 0.1,
        xlim=xlim,
        ylim=ylim,
        tick_length=4.8,
        tick_width=1.0,
        labelsize=14,
    )
    ax.grid(False)
    style_spines(ax, linewidth=0.95)
    ax.legend(loc="lower right", frameon=False, ncol=1, handlelength=1.4, handletextpad=0.45,
              labelspacing=0.16, borderaxespad=0.20, fontsize=fontsize)


def plot_roc_triptych_2top_1bottom(svm_csv, xgb_csv, knn_csv, out_png=None, out_pdf=None):
    roc_svm, auc_svm = load_roc_csv(svm_csv)
    roc_xgb, auc_xgb = load_roc_csv(xgb_csv)
    roc_knn, auc_knn = load_roc_csv(knn_csv)
    set_paper_style(14)
    fig = plt.figure(figsize=(7.4, 5.6), dpi=300)
    gs = GridSpec(2, 4, figure=fig)
    axes = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[1, 1:3])]
    draw_roc_panel(axes[0], roc_svm, auc_svm, "(a) SVM")
    draw_roc_panel(axes[1], roc_xgb, auc_xgb, "(b) XGBoost")
    draw_roc_panel(axes[2], roc_knn, auc_knn, "(c) KNN")
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.06, top=0.96, wspace=0.28, hspace=0.38)
    _save_figure(fig, out_png, out_pdf, dpi=700)
    return fig, axes


def plot_cm_triptych_from_csv(
    root,
    class_file="classes.json",
    cm_files=("cm_svm_fold10.csv", "cm_xgb_fold10.csv", "cm_knn_fold10_sum.csv"),
    titles=("(a) SVM", "(b) XGBoost", "(c) KNN"),
    out_png="Figure_CM_SVM_XGB_KNN_dark_orange_clean_auto.png",
    out_pdf="Figure_CM_SVM_XGB_KNN_dark_orange_clean_auto.pdf",
    figsize=(15.2, 5.2),
    dpi=600,
    annot_size=20,
    show=True,
):
    root = Path(root)
    class_names = _load_class_names(root, class_file)
    cms = [_load_cm_csv(root / f, class_names) for f in cm_files]
    n = len(class_names)
    with plt.rc_context(paper_rc(base=21, axes=24, ticks=22, legend=16, title=25)):
        fig, axes = plt.subplots(1, len(cms), figsize=figsize, dpi=dpi)
        axes = np.ravel(axes)
        for ax, cm, title in zip(axes, cms, titles):
            sns.heatmap(
                cm,
                ax=ax,
                annot=True,
                fmt=".1f",
                cmap=cm_orange_cmap(),
                vmin=0,
                vmax=100,
                square=True,
                xticklabels=class_names,
                yticklabels=class_names,
                linewidths=0,
                cbar=False,
                annot_kws={"size": annot_size},
            )
            ax.set_title(title, pad=8)
            ax.set_xlabel("Predicted", labelpad=6)
            ax.set_ylabel("True", labelpad=6)
            apply_cm_axis_ticks(ax, class_names, x_rotation=45, tick_length=5.0, tick_width=1.0, use_half_positions=True)
            style_spines(ax, linewidth=0.95, color="#7a7a7a")
            for text, val in zip(ax.texts, cm.flatten()):
                text.set_color("white" if val >= 75 else "#222222")
        plt.subplots_adjust(left=0.045, right=0.995, bottom=0.20, top=0.86, wspace=0.12)
        _save_figure(fig, root / out_png if out_png else None, root / out_pdf if out_pdf else None, dpi=dpi)
        if show:
            plt.show()
    return fig, axes, cms


def plot_confusion_panel_from_csv(
    csv_map: Dict[str, str],
    class_file: str,
    out_fig: Optional[str] = None,
    order=("CNN", "ANN", "RF", "XGB", "SVM", "KNN", "LR"),
    out_pdf: Optional[str] = None,
    out_svg: Optional[str] = None,
    figsize=(16.5, 13.2),
    title_size: float = 28,
    x_tick_size: float = 17,
    y_tick_size: float = 20,
    annot_size: float = 21,
    x_rotation: float = 45,
    wspace: float = 0.25,
    hspace: float = 0.46,
    dpi: int = 700,
    show: bool = True,
):
    """Plot a 3 x 3 panel of confusion matrices loaded from CSV files.

    The seven requested models occupy the first seven panel positions. The
    final two positions remain blank. Tick positions are fixed at the centers
    of the matrix cells, and x-axis rotation is applied without being reset by
    a later tick-style helper.
    """
    class_file = Path(class_file)

    with open(class_file, "r", encoding="utf-8") as file:
        class_names = standardize_class_names(json.load(file))

    avg_cms = {
        model: _load_cm_csv(path, class_names) / 100.0
        for model, path in csv_map.items()
    }

    titles = {
        name: f"({chr(97 + index)}) {name}"
        for index, name in enumerate(order)
    }

    positions = {
        "CNN": (0, 0),
        "ANN": (0, 1),
        "RF": (0, 2),
        "XGB": (1, 0),
        "SVM": (1, 1),
        "KNN": (1, 2),
        "LR": (2, 0),
    }

    n_classes = len(class_names)
    tick_positions = np.arange(n_classes)

    fig = plt.figure(figsize=figsize, dpi=300)

    gs = GridSpec(
        3,
        3,
        figure=fig,
        left=0.055,
        right=0.985,
        bottom=0.075,
        top=0.955,
        wspace=wspace,
        hspace=hspace,
    )

    for name in order:
        if name not in avg_cms:
            continue

        if name not in positions:
            raise ValueError(
                f"Unsupported panel name: {name}. "
                f"Supported names: {sorted(positions)}"
            )

        row, col = positions[name]
        ax = fig.add_subplot(gs[row, col])
        cm = avg_cms[name]

        ax.imshow(
            cm,
            vmin=0.00,
            vmax=0.96,
            cmap=warm_deep_cmap(),
            interpolation="nearest",
            aspect="equal",
        )

        ax.set_title(
            titles[name],
            fontsize=title_size,
            fontweight="bold",
            pad=10,
        )

        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)

        ax.set_xticklabels(
            class_names,
            fontsize=x_tick_size,
            rotation=x_rotation,
            ha="right",
            va="top",
            rotation_mode="anchor",
        )

        ax.set_yticklabels(
            class_names,
            fontsize=y_tick_size,
            rotation=0,
            ha="right",
            va="center",
        )

        ax.tick_params(
            axis="x",
            which="major",
            bottom=True,
            top=False,
            labelbottom=True,
            direction="out",
            length=6,
            width=1.2,
            pad=5,
        )

        ax.tick_params(
            axis="y",
            which="major",
            left=True,
            right=False,
            labelleft=True,
            direction="out",
            length=6,
            width=1.2,
            pad=5,
        )

        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())

        ax.set_xlim(-0.5, n_classes - 0.5)
        ax.set_ylim(n_classes - 0.5, -0.5)
        ax.set_aspect("equal")

        style_spines(
            ax,
            visible_sides=("left", "bottom", "top", "right"),
            linewidth=1.1,
            color="black",
        )

        for true_index in range(n_classes):
            for pred_index in range(n_classes):
                value = cm[true_index, pred_index]

                ax.text(
                    pred_index,
                    true_index,
                    f"{value * 100:.0f}",
                    ha="center",
                    va="center",
                    fontsize=annot_size,
                    fontweight="bold" if true_index == pred_index else "normal",
                    color="white" if value >= 0.55 else "black",
                )

    fig.add_subplot(gs[2, 1]).axis("off")
    fig.add_subplot(gs[2, 2]).axis("off")

    save_targets = [
        (out_fig, dpi),
        (out_pdf, None),
        (out_svg, None),
    ]

    for target, save_dpi in save_targets:
        if target is None:
            continue

        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)

        save_kwargs = {
            "bbox_inches": "tight",
            "pad_inches": 0.05,
            "facecolor": "white",
        }

        if save_dpi is not None:
            save_kwargs["dpi"] = save_dpi

        fig.savefig(target, **save_kwargs)

    if show:
        plt.show()

    return fig, avg_cms


# =============================================================================
# Dimensionality-reduction plots
# =============================================================================


def run_pca_tsne(X, seed=1337, pca_components=50, perplexity=40):
    X50 = PCA(n_components=min(pca_components, X.shape[1]), random_state=seed).fit_transform(X)
    perplexity = min(perplexity, max(5, (len(X50) - 1) // 3))
    return TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed).fit_transform(X50)


def plot_tsne_original_vs_ddpm(
    feature_path,
    label_path,
    class_json,
    aug_feature_path=None,
    aug_label_path=None,
    out_png=None,
    out_pdf=None,
    seed=1337,
    require_aug=True,
):
    """Plot t-SNE of original embeddings and DDPM-augmented embeddings.

    Safety behavior:
    - If require_aug=True, both aug_X.npy and aug_y.npy must exist.
    - This prevents panel (b) from being mislabeled as "Real + DDPM" when
      augmented DDPM features have not actually been generated/saved.
    - If require_aug=False and DDPM files are missing, panel (b) is labeled
      "Original only" instead of "Real + DDPM".
    """
    feature_path = Path(feature_path)
    label_path = Path(label_path)
    class_json = Path(class_json)

    X = np.load(feature_path).astype(np.float32)
    y = np.load(label_path).astype(np.int64).ravel()
    class_names = standardize_class_names(json.load(open(class_json, "r", encoding="utf-8")))

    aug_feature_path = Path(aug_feature_path) if aug_feature_path is not None else None
    aug_label_path = Path(aug_label_path) if aug_label_path is not None else None

    has_aug = (
        aug_feature_path is not None
        and aug_label_path is not None
        and aug_feature_path.exists()
        and aug_label_path.exists()
    )

    if require_aug and not has_aug:
        raise FileNotFoundError(
            "DDPM augmented files are missing. The figure will NOT be drawn as 'Real + DDPM'. "
            f"Expected aug_feature_path={aug_feature_path}, aug_label_path={aug_label_path}. "
            "Please generate and save aug_X.npy / aug_y.npy before plotting."
        )

    print("[INFO] Original features loaded:")
    print("       X:", X.shape)
    print("       y:", y.shape)

    Z_real = run_pca_tsne(X, seed=seed)
    Zr, Za, y_aug = Z_real, None, None

    if has_aug:
        X_aug = np.load(aug_feature_path).astype(np.float32)
        y_aug = np.load(aug_label_path).astype(np.int64).ravel()

        if X_aug.ndim != 2:
            raise ValueError(f"X_aug must be 2D, got shape: {X_aug.shape}")
        if len(X_aug) == 0:
            raise ValueError(
                "DDPM augmented file exists, but X_aug has zero samples. "
                "Panel (b) cannot be labeled as Real + DDPM."
            )
        if X_aug.shape[1] != X.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: original X has {X.shape[1]} dimensions, "
                f"but X_aug has {X_aug.shape[1]} dimensions."
            )
        if len(y_aug) != len(X_aug):
            raise ValueError(
                f"aug_X / aug_y size mismatch: X_aug={X_aug.shape}, y_aug={y_aug.shape}"
            )
        if np.any(y_aug < 0) or np.any(y_aug >= len(class_names)):
            raise ValueError(
                "aug_y contains class labels outside the valid range "
                f"0..{len(class_names)-1}. Found: {np.unique(y_aug)}"
            )

        print("[INFO] DDPM augmented features loaded:")
        print("       X_aug:", X_aug.shape)
        print("       y_aug:", y_aug.shape)

        Z_mix = run_pca_tsne(np.vstack([X, X_aug]), seed=seed)
        Zr, Za = Z_mix[:len(X)], Z_mix[len(X):]
    else:
        print("[WARN] DDPM augmented files were not found.")
        print("       Panel (b) will be labeled as 'Original only', not 'Real + DDPM'.")

    with plt.rc_context(paper_rc(base=16, axes=16, ticks=14, legend=12, title=18)):
        cmap = mpl.colormaps.get_cmap("tab10")
        colors = [cmap(i) for i in range(len(class_names))]

        all_x = [Z_real[:, 0], Zr[:, 0]]
        all_y = [Z_real[:, 1], Zr[:, 1]]
        if Za is not None:
            all_x.append(Za[:, 0])
            all_y.append(Za[:, 1])

        xmin, xmax = min(a.min() for a in all_x), max(a.max() for a in all_x)
        ymin, ymax = min(a.min() for a in all_y), max(a.max() for a in all_y)
        padx, pady = 0.06 * (xmax - xmin), 0.06 * (ymax - ymin)

        fig, axes = plt.subplots(
            1, 2, figsize=(10.2, 5.2), sharex=True, sharey=True, dpi=300
        )

        for i in range(len(class_names)):
            axes[0].scatter(
                Z_real[y == i, 0], Z_real[y == i, 1],
                s=28, alpha=0.90, color=colors[i], edgecolors="none"
            )
            axes[1].scatter(
                Zr[y == i, 0], Zr[y == i, 1],
                s=28, alpha=0.90, color=colors[i], edgecolors="none"
            )
            if Za is not None:
                pastel = np.array(colors[i]).copy()
                pastel[:3] = np.clip(pastel[:3] + 0.25, 0, 1)
                axes[1].scatter(
                    Za[y_aug == i, 0], Za[y_aug == i, 1],
                    s=16, alpha=0.30, color=pastel, edgecolors="none"
                )

        axes[0].set_title("(a) Original", fontweight="bold")
        axes[1].set_title(
            "(b) Real + DDPM" if Za is not None else "(b) Original only",
            fontweight="bold"
        )

        for ax in axes:
            ax.set_xlim(xmin - padx, xmax + padx)
            ax.set_ylim(ymin - pady, ymax + pady)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            apply_visible_axis_ticks(ax, tick_length=4.0, tick_width=0.9, labelsize=12)
            ax.grid(False)

        handles = [
            Line2D(
                [0], [0], marker="o", linestyle="None",
                markerfacecolor=colors[i], markeredgecolor="none",
                markersize=6.5, label=class_names[i]
            )
            for i in range(len(class_names))
        ]
        fig.legend(
            handles=handles, labels=class_names,
            loc="center left", bbox_to_anchor=(0.83, 0.5), frameon=False
        )
        plt.subplots_adjust(left=0.08, right=0.82, bottom=0.15, top=0.88, wspace=0.23)
        _save_figure(fig, out_png, out_pdf, dpi=1000)
        return fig, axes


def plot_pca_tsne_umap_panel(feature_path, label_path, class_json, out_png=None, out_pdf=None, random_state=1337, umap_fallback="pca", verbose=True):
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    X = np.load(feature_path).astype(np.float32)
    y = np.load(label_path).astype(np.int64).ravel()
    class_names = standardize_class_names(json.load(open(class_json, "r", encoding="utf-8")))
    X_scaled = StandardScaler().fit_transform(X)
    X_pca = PCA(n_components=2, random_state=random_state).fit_transform(X_scaled)
    perplexity = min(35, max(5, (len(y) - 1) // 3))
    X_tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto", init="pca", random_state=random_state).fit_transform(X_scaled)
    umap_ok = True
    try:
        from umap.umap_ import UMAP
        X_umap = UMAP(n_components=2, n_neighbors=30, min_dist=0.2, random_state=random_state).fit_transform(X_scaled)
    except Exception as exc:
        umap_ok = False
        if verbose:
            print(f"[WARN] UMAP failed ({type(exc).__name__}: {exc}). Using {umap_fallback} fallback.")
        X_umap = X_pca.copy() if str(umap_fallback).lower() == "pca" else X_tsne.copy()

    def rescale_to_target(Z, ref):
        Z, ref = np.asarray(Z, float), np.asarray(ref, float)
        Zs = Z.copy()
        for d in range(2):
            zmin, zmax = Z[:, d].min(), Z[:, d].max()
            rmin, rmax = ref[:, d].min(), ref[:, d].max()
            denom = zmax - zmin
            Zs[:, d] = 0.5 * (rmin + rmax) if abs(denom) < 1e-12 else ((Z[:, d] - zmin) / denom) * (rmax - rmin) + rmin
        return Zs

    embeddings = [rescale_to_target(X_pca, X_tsne), X_tsne, rescale_to_target(X_umap, X_tsne)]
    titles = ["(a) PCA", "(b) t-SNE", "(c) UMAP" if umap_ok else "(c) PCA fallback"]
    xmin, xmax = X_tsne[:, 0].min(), X_tsne[:, 0].max()
    ymin, ymax = X_tsne[:, 1].min(), X_tsne[:, 1].max()
    padx, pady = 0.04 * (xmax - xmin), 0.04 * (ymax - ymin)
    with plt.rc_context(paper_rc(base=14, axes=14, ticks=11, legend=11, title=15)):
        colors = plt.cm.tab10.colors
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 4.50), dpi=300)
        axes = np.ravel(axes)
        for j, (ax, Z, title) in enumerate(zip(axes, embeddings, titles)):
            for i in range(len(class_names)):
                ax.scatter(Z[y == i, 0], Z[y == i, 1], s=10, alpha=0.90, color=colors[i % len(colors)], edgecolors="none")
            ax.set_xlim(xmin - padx, xmax + padx); ax.set_ylim(ymin - pady, ymax + pady)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(title, fontweight="bold", pad=3)
            ax.set_xlabel("Dimension 1", labelpad=1)
            ax.set_ylabel("Dimension 2" if j == 0 else "", labelpad=1)
            apply_visible_axis_ticks(ax, tick_length=3.0, tick_width=0.8, labelsize=11)
            ax.grid(False)
            style_spines(ax, linewidth=0.8)
        handles = [Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=colors[i % len(colors)], markeredgecolor="none", markersize=5.2, label=class_names[i]) for i in range(len(class_names))]
        fig.legend(handles=handles, labels=class_names, loc="center left", bbox_to_anchor=(0.885, 0.5), frameon=False)
        plt.subplots_adjust(left=0.07, right=0.87, bottom=0.16, top=0.90, wspace=0.28)
        _save_figure(fig, out_png, out_pdf, dpi=900)
        return fig, axes


def plot_tsne_from_features(
    root,
    feat_name="features_b3_1536.npy",
    label_name="labels_b3.npy",
    class_name="classes.json",
    fallback_class_name="class_to_idx.json",
    save_name="tsne_features_b3.png",
    perplexity=35,
    learning_rate="auto",
    max_iter=1200,
    random_state=42,
    point_size=7,
    alpha=0.75,
    figsize=(4.0, 3.2),
    dpi=150,
    save_dpi=300,
    title="t-SNE",
    show=True,
    verbose=True,
):
    root = Path(root)
    X, y = _load_features_labels(root, feat_name, label_name)
    names = _load_class_names(root, class_name, fallback_class_name, y)
    if verbose:
        print(f"Loaded: X={X.shape}, y={y.shape}")
        print("Classes:", names)
    X_scaled = StandardScaler().fit_transform(X)
    perplexity = min(perplexity, max(5, (len(y) - 1) // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, max_iter=max_iter, init="pca", random_state=random_state, verbose=1 if verbose else 0)
    X_2d = tsne.fit_transform(X_scaled)
    with plt.rc_context(paper_rc(base=12, axes=12, ticks=11, legend=10, title=12)):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        palette = sns.color_palette("tab10", len(names))
        for i, name in enumerate(names):
            ax.scatter(X_2d[y == i, 0], X_2d[y == i, 1], s=point_size, alpha=alpha, label=name, color=palette[i % len(palette)])
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_title(title, fontweight="bold", pad=10)
        apply_visible_axis_ticks(ax, tick_length=3.0, tick_width=0.8, labelsize=10)
        ax.legend(markerscale=1.2, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, frameon=False)
        fig.tight_layout()
        if save_name is not None:
            save_path = root / save_name
            _save_figure(fig, save_path, save_path.with_suffix(".pdf") if save_path.suffix.lower() in [".png", ".jpg", ".jpeg"] else None, dpi=save_dpi)
            if verbose:
                print("Saved:", save_path)
        if show:
            plt.show()
        return fig, ax, X_2d


# =============================================================================
# Training curves
# =============================================================================


def plot_training_curves(
    history=None,
    save_path=None,
    grad_norms=None,
    show=True,
    dpi=300,
    figsize=None,
    train_accs=None,
    val_accs=None,
    train_losses=None,
    val_losses=None,
):
    """Plot training curves.

    Supports both:
    1) history dict with keys train_loss, val_loss, train_acc, val_acc, lr;
    2) explicit arrays train_accs / val_accs / train_losses / val_losses.
    """
    if history is not None and isinstance(history, dict):
        train_losses = np.asarray(history.get("train_loss", []), dtype=float)
        val_losses = np.asarray(history.get("val_loss", []), dtype=float)
        train_accs = np.asarray(history.get("train_acc", []), dtype=float)
        val_accs = np.asarray(history.get("val_acc", []), dtype=float)
        lr = np.asarray(history.get("lr", history.get("lrs", [])), dtype=float) if len(history.get("lr", history.get("lrs", []))) else None
        if grad_norms is None:
            grad_norms = history.get("grad_norm", history.get("grad_norms", None))
    else:
        lr = None
        train_losses = np.asarray(train_losses or [], dtype=float)
        val_losses = np.asarray(val_losses or [], dtype=float)
        train_accs = np.asarray(train_accs or [], dtype=float)
        val_accs = np.asarray(val_accs or [], dtype=float)

    n_epochs = max(len(train_losses), len(train_accs))
    epochs = np.arange(1, n_epochs + 1)
    grad_norms = np.asarray(grad_norms, dtype=float) if grad_norms is not None else None
    has_grad = grad_norms is not None and len(grad_norms) == n_epochs and n_epochs > 0
    has_lr = lr is not None and len(lr) > 0
    n_panels = 2 + int(has_lr) + int(has_grad)
    if figsize is None:
        figsize = (12, 5) if n_panels == 2 else (5.2 * n_panels, 5)

    with plt.rc_context(paper_rc(base=16, axes=18, ticks=15, legend=13, title=18)):
        fig, axes = plt.subplots(1, n_panels, figsize=figsize, dpi=150)
        axes = np.ravel(axes)
        idx = 0
        axes[idx].plot(epochs[:len(train_accs)], train_accs, marker="o", linewidth=2, markersize=4, color="C3", label="Training")
        axes[idx].plot(epochs[:len(val_accs)], val_accs, marker="s", linewidth=2, markersize=4, color="C0", label="Validation")
        axes[idx].set_title("Training vs Validation Accuracy")
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel("Accuracy")
        apply_visible_axis_ticks(axes[idx], tick_length=4.0, tick_width=0.9)
        axes[idx].grid(True, alpha=0.25)
        axes[idx].legend(frameon=False)
        idx += 1

        axes[idx].plot(epochs[:len(train_losses)], train_losses, marker="o", linewidth=2, markersize=4, color="C3", label="Training")
        axes[idx].plot(epochs[:len(val_losses)], val_losses, marker="s", linewidth=2, markersize=4, color="C0", label="Validation")
        axes[idx].set_title("Training vs Validation Loss")
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel("Loss")
        apply_visible_axis_ticks(axes[idx], tick_length=4.0, tick_width=0.9)
        axes[idx].grid(True, alpha=0.25)
        axes[idx].legend(frameon=False)
        idx += 1

        if has_lr:
            axes[idx].plot(np.arange(1, len(lr) + 1), lr, linewidth=2, color="green")
            axes[idx].set_title("Learning Rate Schedule")
            axes[idx].set_xlabel("Epoch")
            axes[idx].set_ylabel("LR")
            apply_visible_axis_ticks(axes[idx], tick_length=4.0, tick_width=0.9)
            axes[idx].grid(True, alpha=0.25)
            idx += 1
        if has_grad:
            axes[idx].plot(epochs, grad_norms, linewidth=2, color="purple", label="Grad Norm")
            axes[idx].set_title("Gradient Norm Evolution")
            axes[idx].set_xlabel("Epoch")
            axes[idx].set_ylabel(r"$||\nabla W_t||_2$")
            apply_visible_axis_ticks(axes[idx], tick_length=4.0, tick_width=0.9)
            axes[idx].grid(True, alpha=0.25)
            axes[idx].legend(frameon=False)

        fig.tight_layout()
        if save_path is not None:
            save_path = Path(save_path)
            _save_figure(fig, save_path, save_path.with_suffix(".pdf") if save_path.suffix.lower() in [".png", ".jpg", ".jpeg"] else None, dpi=dpi)
        if show:
            plt.show()
        return fig, axes


# =============================================================================
# Unified 10-fold ML: SVM / XGBoost / KNN
# =============================================================================


def _make_classifier(model_type: str, random_state: int = 42, **kwargs):
    model_type = str(model_type).lower()
    if model_type == "svm":
        return SVC(
            kernel=kwargs.get("kernel", "rbf"),
            C=kwargs.get("C", 2.0),
            gamma=kwargs.get("gamma", "scale"),
            probability=True,
            random_state=random_state,
        )
    if model_type == "knn":
        return KNeighborsClassifier(
            n_neighbors=kwargs.get("n_neighbors", 5),
            weights=kwargs.get("weights", "distance"),
            n_jobs=kwargs.get("n_jobs", -1),
        )
    if model_type == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            max_depth=kwargs.get("max_depth", 7),
            n_estimators=kwargs.get("n_estimators", 400),
            learning_rate=kwargs.get("learning_rate", 0.08),
            subsample=kwargs.get("subsample", 0.9),
            colsample_bytree=kwargs.get("colsample_bytree", 0.9),
            eval_metric=kwargs.get("eval_metric", "mlogloss"),
            tree_method=kwargs.get("tree_method", "hist"),
            n_jobs=kwargs.get("n_jobs", -1),
            random_state=random_state,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def _run_cm_roc_10fold(
    root,
    model_type: str,
    display_name: str,
    feat_name="features_b3_1536.npy",
    label_name="labels_b3.npy",
    class_name="classes.json",
    fallback_class_name="class_to_idx.json",
    out_cm_csv_name=None,
    out_cm_png_name=None,
    out_cm_pdf_name=None,
    out_roc_csv_name=None,
    out_roc_png_name=None,
    out_roc_pdf_name=None,
    n_splits=10,
    cv_seed=1337,
    model_seed=42,
    plot=True,
    show=True,
    verbose=True,
    **model_kwargs,
):
    root = Path(root)
    X, y = _load_features_labels(root, feat_name, label_name)
    names = _load_class_names(root, class_name, fallback_class_name, y)
    n_classes = len(names)
    class_ids = np.arange(n_classes)

    if out_cm_csv_name is None:
        out_cm_csv_name = f"cm_{model_type}_fold10.csv"
    if out_cm_png_name is None:
        out_cm_png_name = f"cm_{model_type}.png"
    if out_cm_pdf_name is None:
        out_cm_pdf_name = f"cm_{model_type}.pdf"
    if out_roc_csv_name is None:
        out_roc_csv_name = f"roc_{model_type}_10fold.csv"
    if out_roc_png_name is None:
        out_roc_png_name = f"roc_{model_type}_10fold.png"
    if out_roc_pdf_name is None:
        out_roc_pdf_name = f"roc_{model_type}_10fold.pdf"

    if verbose:
        print(f"Loaded: X={X.shape}, y={y.shape}")
        print("Classes:", names)
        print(f"Running {display_name} CM/ROC with {n_splits}-fold CV...")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)
    y_true_all, y_pred_all = [], []
    y_score_all, y_true_bin_all = [], []
    accs, pres, recs, f1s = [], [], [], []

    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xva = scaler.transform(X[va])
        clf = _make_classifier(model_type, random_state=model_seed, **model_kwargs)
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xva)
        proba = clf.predict_proba(Xva)

        aligned = np.zeros((len(va), n_classes), dtype=float)
        for j, cls in enumerate(clf.classes_):
            cls = int(cls)
            if 0 <= cls < n_classes:
                aligned[:, cls] = proba[:, j]

        y_true_all.extend(y[va])
        y_pred_all.extend(pred)
        y_score_all.append(aligned)
        y_true_bin_all.append(label_binarize(y[va], classes=class_ids))

        acc = accuracy_score(y[va], pred)
        pre, rec, f1, _ = precision_recall_fscore_support(y[va], pred, average="macro", zero_division=0)
        accs.append(acc); pres.append(pre); recs.append(rec); f1s.append(f1)
        if verbose:
            print(f"[{display_name} fold {fold:02d}] Acc={acc*100:.2f}% | F1={f1*100:.2f}%")

    y_true_all = np.asarray(y_true_all)
    y_pred_all = np.asarray(y_pred_all)
    y_score = np.vstack(y_score_all)
    y_true_bin = np.vstack(y_true_bin_all)

    report = classification_report(y_true_all, y_pred_all, target_names=names, digits=4, zero_division=0)
    if verbose:
        print(f"\n====== CLASSIFICATION REPORT: {display_name} 10-fold CV ======\n")
        print(report)

    cm_count = confusion_matrix(y_true_all, y_pred_all, labels=class_ids)
    cm_pct = np.divide(
        cm_count,
        cm_count.sum(axis=1, keepdims=True),
        out=np.zeros_like(cm_count, dtype=float),
        where=cm_count.sum(axis=1, keepdims=True) != 0,
    ) * 100.0

    out_cm_csv = root / out_cm_csv_name
    pd.DataFrame(cm_pct, index=names, columns=names).to_csv(out_cm_csv, encoding="utf-8-sig", float_format="%.1f")
    if model_type == "knn":
        out_cm_alias = root / "cm_knn_fold10.csv"
        pd.DataFrame(cm_pct, index=names, columns=names).to_csv(out_cm_alias, encoding="utf-8-sig", float_format="%.1f")
    else:
        out_cm_alias = None

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    roc_auc["macro"] = float(np.mean([roc_auc[i] for i in range(n_classes)]))

    rows = []
    for i in range(n_classes):
        for fp, tp in zip(fpr[i], tpr[i]):
            rows.append({"class": names[i], "fpr": float(fp), "tpr": float(tp), "auc": float(roc_auc[i])})
    for fp, tp in zip(fpr["micro"], tpr["micro"]):
        rows.append({"class": "micro", "fpr": float(fp), "tpr": float(tp), "auc": float(roc_auc["micro"])})
    out_roc_csv = root / out_roc_csv_name
    pd.DataFrame(rows).to_csv(out_roc_csv, index=False, float_format="%.6f", encoding="utf-8-sig")

    if plot:
        _plot_cm_percent(
            cm_pct,
            names,
            title=f"Confusion Matrix – {display_name} (10-fold CV)",
            out_png=root / out_cm_png_name,
            out_pdf=root / out_cm_pdf_name,
            show=show,
            figsize=(5.0, 4.0),
            title_size=18,
            label_size=18,
            tick_size=14,
            annot_size=14,
        )
        _plot_roc_curves(
            fpr,
            tpr,
            roc_auc,
            names,
            title=f"ROC Curves – {display_name} (10-fold CV)",
            out_png=root / out_roc_png_name,
            out_pdf=root / out_roc_pdf_name,
            figsize=(5.0, 4.0),
            show=show,
            title_size=18,
            label_size=18,
            tick_size=14,
            legend_size=13,
        )

    if verbose:
        print("\nSaved CM CSV  ->", out_cm_csv)
        if out_cm_alias is not None:
            print("Saved CM CSV alias ->", out_cm_alias)
        print("Saved ROC CSV ->", out_roc_csv)
        if plot:
            print("Saved CM PNG  ->", root / out_cm_png_name)
            print("Saved CM PDF  ->", root / out_cm_pdf_name)
            print("Saved ROC PNG ->", root / out_roc_png_name)
            print("Saved ROC PDF ->", root / out_roc_pdf_name)
        print(f"micro-AUC={roc_auc['micro']:.4f} | macro-AUC={roc_auc['macro']:.4f}")

    result = {
        "report": report,
        "cm": cm_count,
        "cm_pct": cm_pct,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "y_score": y_score,
        "summary": {
            "Accuracy": float(np.mean(accs)),
            "Precision": float(np.mean(pres)),
            "Recall": float(np.mean(recs)),
            "F1": float(np.mean(f1s)),
            "Acc_std": float(np.std(accs, ddof=1)),
            "F1_std": float(np.std(f1s, ddof=1)),
        },
        "out_cm_csv": out_cm_csv,
        "out_roc_csv": out_roc_csv,
    }
    if out_cm_alias is not None:
        result["out_cm_alias"] = out_cm_alias
    return result


def run_knn_cm_roc_10fold(
    root,
    feat_name="features_b3_1536.npy",
    label_name="labels_b3.npy",
    class_name="classes.json",
    fallback_class_name="class_to_idx.json",
    out_cm_csv_name="cm_knn_fold10_sum.csv",
    out_cm_png_name="cm_knn.png",
    out_cm_pdf_name="cm_knn.pdf",
    out_roc_csv_name="roc_knn_10fold.csv",
    out_roc_png_name="roc_knn_10fold.png",
    out_roc_pdf_name="roc_knn_10fold.pdf",
    n_splits=10,
    seed=1337,
    n_neighbors=5,
    weights="distance",
    plot=True,
    show=True,
    verbose=True,
):
    return _run_cm_roc_10fold(
        root=root,
        model_type="knn",
        display_name="KNN",
        feat_name=feat_name,
        label_name=label_name,
        class_name=class_name,
        fallback_class_name=fallback_class_name,
        out_cm_csv_name=out_cm_csv_name,
        out_cm_png_name=out_cm_png_name,
        out_cm_pdf_name=out_cm_pdf_name,
        out_roc_csv_name=out_roc_csv_name,
        out_roc_png_name=out_roc_png_name,
        out_roc_pdf_name=out_roc_pdf_name,
        n_splits=n_splits,
        cv_seed=seed,
        model_seed=42,
        plot=plot,
        show=show,
        verbose=verbose,
        n_neighbors=n_neighbors,
        weights=weights,
    )


def run_xgb_cm_roc_10fold(
    root,
    feat_name="features_b3_1536.npy",
    label_name="labels_b3.npy",
    class_name="classes.json",
    fallback_class_name="class_to_idx.json",
    out_cm_png_name="cm_xgb.png",
    out_roc_png_name="roc_xgb.png",
    out_cm_csv_name="cm_xgb_fold10.csv",
    out_roc_csv_name="roc_xgb_auc.csv",
    n_splits=10,
    cv_seed=1337,
    model_seed=42,
    max_depth=7,
    n_estimators=400,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist",
    n_jobs=-1,
    plot=True,
    show=True,
    verbose=True,
):
    return _run_cm_roc_10fold(
        root=root,
        model_type="xgb",
        display_name="XGBoost",
        feat_name=feat_name,
        label_name=label_name,
        class_name=class_name,
        fallback_class_name=fallback_class_name,
        out_cm_csv_name=out_cm_csv_name,
        out_cm_png_name=out_cm_png_name,
        out_cm_pdf_name=Path(out_cm_png_name).with_suffix(".pdf").name,
        out_roc_csv_name=out_roc_csv_name,
        out_roc_png_name=out_roc_png_name,
        out_roc_pdf_name=Path(out_roc_png_name).with_suffix(".pdf").name,
        n_splits=n_splits,
        cv_seed=cv_seed,
        model_seed=model_seed,
        plot=plot,
        show=show,
        verbose=verbose,
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        tree_method=tree_method,
        n_jobs=n_jobs,
    )


def run_svm_cm_roc_10fold(
    root,
    feat_name="features_b3_1536.npy",
    label_name="labels_b3.npy",
    class_name="classes.json",
    fallback_class_name="class_to_idx.json",
    out_cm_csv_name="cm_svm_fold10.csv",
    out_cm_png_name="cm_svm.png",
    out_cm_pdf_name="cm_svm.pdf",
    out_roc_csv_name="roc_svm_10fold.csv",
    out_roc_png_name="roc_svm_10fold.png",
    out_roc_pdf_name="roc_svm_10fold.pdf",
    n_splits=10,
    cv_seed=1337,
    model_seed=42,
    C=2.0,
    gamma="scale",
    plot=True,
    show=True,
    verbose=True,
):
    return _run_cm_roc_10fold(
        root=root,
        model_type="svm",
        display_name="SVM",
        feat_name=feat_name,
        label_name=label_name,
        class_name=class_name,
        fallback_class_name=fallback_class_name,
        out_cm_csv_name=out_cm_csv_name,
        out_cm_png_name=out_cm_png_name,
        out_cm_pdf_name=out_cm_pdf_name,
        out_roc_csv_name=out_roc_csv_name,
        out_roc_png_name=out_roc_png_name,
        out_roc_pdf_name=out_roc_pdf_name,
        n_splits=n_splits,
        cv_seed=cv_seed,
        model_seed=model_seed,
        plot=plot,
        show=show,
        verbose=verbose,
        C=C,
        gamma=gamma,
    )


# Backward-compatible aliases
run_svm_cm_10fold = run_svm_cm_roc_10fold


def plot_svm_roc_10fold_from_features(*args, **kwargs):
    """Backward-compatible wrapper. It now runs both CM and ROC for SVM."""
    return run_svm_cm_roc_10fold(*args, **kwargs)

# =============================================================================
# DDPM feature-space t-SNE visualization
# =============================================================================

def plot_ddpm_tsne_real_vs_synthetic(
    root,
    feat_name="features_b3_1536.npy",
    label_name="labels_b3.npy",
    class_name="classes.json",
    fallback_class_name="class_to_idx.json",
    train_frac: float = 0.25,
    fold_id: int = 1,
    n_splits: int = 10,
    seed: int = 1337,
    pca_dim: int = 64,
    diffusion_steps: int = 100,
    ddpm_train_steps: int = 4000,
    ddpm_batch_size: int = 128,
    ddpm_lr: float = 2e-4,
    ddpm_width: int = 512,
    confidence_threshold: float = 0.85,
    nn_percentile: float = 95,
    nn_multiplier: float = 1.25,
    oversample_factor: float = 4.0,
    max_real_per_class: int = 30,
    max_syn_per_class: int = 30,
    tsne_perplexity: int = 30,
    out_png=None,
    out_pdf=None,
    show: bool = True,
    verbose: bool = True,
):
    """Train DDPM on a low-data fold and plot real vs synthetic features with t-SNE.

    This is intended for paper-style visualization only. A balanced subset is
    used for plotting so the clusters are readable; the DDPM itself is trained
    on the selected train_frac subset.
    """
    import importlib
    import torch
    from sklearn.model_selection import train_test_split

    import utils.utils_ddpm as D
    D = importlib.reload(D)

    root = Path(root)
    X, y = _load_features_labels(root, feat_name, label_name)
    class_names = _load_class_names(root, class_name, fallback_class_name, y)
    n_classes = len(class_names)

    rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print("Device:", device)
        print("Loaded:", X.shape, y.shape)

    # Select one stratified fold, then take train_frac of the training split.
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(cv.split(X, y))
    if fold_id < 1 or fold_id > len(splits):
        raise ValueError(f"fold_id must be 1..{len(splits)}, got {fold_id}")
    tr_idx, _ = splits[fold_id - 1]

    Xtr_full, ytr_full = X[tr_idx], y[tr_idx]
    if train_frac < 1.0:
        Xtr, _, ytr, _ = train_test_split(
            Xtr_full,
            ytr_full,
            train_size=train_frac,
            stratify=ytr_full,
            random_state=seed,
        )
    else:
        Xtr, ytr = Xtr_full, ytr_full

    # Scale + PCA latent space.
    Xtr_s = StandardScaler().fit_transform(Xtr)
    actual_pca_dim = int(min(pca_dim, Xtr_s.shape[0] - 1, Xtr_s.shape[1]))
    pca = PCA(n_components=actual_pca_dim, random_state=seed)
    Xreal = pca.fit_transform(Xtr_s).astype(np.float32)
    pca_var = float(pca.explained_variance_ratio_.sum())

    if verbose:
        print(f"Real: {Xreal.shape} | PCA var: {pca_var:.4f}")

    # Train DDPM + generate filtered synthetic features.
    schedule = D.DiffusionSchedule(T=diffusion_steps, device=device)
    ddpm = D.train_ddpm(
        Xreal,
        ytr,
        n_classes=n_classes,
        steps=ddpm_train_steps,
        batch_size=ddpm_batch_size,
        lr=ddpm_lr,
        schedule=schedule,
        width=ddpm_width,
        verbose=verbose,
        log_every=500,
    )

    gen_fn = getattr(D, "generate_controlled_ddpm_features", None)
    if gen_fn is None:
        gen_fn = getattr(D, "_generate_controlled_ddpm_features")

    Xsyn, ysyn = gen_fn(
        ddpm=ddpm,
        X_real_latent=Xreal,
        y_real=ytr,
        n_classes=n_classes,
        n_total=len(ytr),
        schedule=schedule,
        seed=seed,
        filter_synthetic=True,
        confidence_threshold=confidence_threshold,
        nn_percentile=nn_percentile,
        nn_multiplier=nn_multiplier,
        oversample_factor=oversample_factor,
        allow_filter_fallback=False,
        verbose=False,
    )

    if verbose:
        print("Synthetic:", Xsyn.shape)

    # Balanced subset for clean visualization.
    r_keep, s_keep = [], []
    for c in range(n_classes):
        ir = np.where(ytr == c)[0]
        isy = np.where(ysyn == c)[0]
        if len(ir) > max_real_per_class:
            ir = rng.choice(ir, max_real_per_class, replace=False)
        if len(isy) > max_syn_per_class:
            isy = rng.choice(isy, max_syn_per_class, replace=False)
        r_keep.extend(ir.tolist())
        s_keep.extend(isy.tolist())

    r_keep = np.asarray(r_keep, dtype=int)
    s_keep = np.asarray(s_keep, dtype=int)
    Xr, yr = Xreal[r_keep], ytr[r_keep]
    Xs, ys = Xsyn[s_keep], ysyn[s_keep]

    Xall = np.vstack([Xr, Xs]).astype(np.float32)
    yall = np.r_[yr, ys]
    is_syn = np.r_[np.zeros(len(yr), dtype=bool), np.ones(len(ys), dtype=bool)]

    perplexity = min(tsne_perplexity, max(5, len(Xall) // 4))
    Z = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(Xall)

    # Paper-style square plot.
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    with plt.rc_context(paper_rc(base=12, axes=15, ticks=12, legend=12, title=20)):
        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        ax.set_box_aspect(1)

        for c in range(n_classes):
            m_real = (yall == c) & (~is_syn)
            m_syn = (yall == c) & (is_syn)
            ax.scatter(
                Z[m_real, 0], Z[m_real, 1],
                s=60, marker="o", color=colors[c], alpha=0.75,
                edgecolors="white", linewidths=0.5, zorder=2,
            )
            ax.scatter(
                Z[m_syn, 0], Z[m_syn, 1],
                s=110, marker="X", color=colors[c], alpha=0.98,
                edgecolors="black", linewidths=0.6, zorder=3,
            )

        ax.set_xlabel("t-SNE 1", fontweight="bold")
        ax.set_ylabel("t-SNE 2", fontweight="bold")
        ax.set_title(
            f"t-SNE of Real and DDPM-Generated Features ({int(train_frac*100)}% Train)",
            fontsize=20, fontweight="bold", pad=14,
        )
        style_spines(ax, visible_sides=("left", "bottom", "top", "right"), linewidth=1.2, color="black")
        ax.tick_params(axis="both", which="major", length=5, width=1.1)
        ax.grid(False)

        class_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[i],
                   markeredgecolor=colors[i], markersize=9, label=class_names[i])
            for i in range(n_classes)
        ]
        source_handles = [
            Line2D([0], [0], marker="o", color="black", linestyle="None", markersize=7, label="Real"),
            Line2D([0], [0], marker="X", color="black", linestyle="None", markersize=8, label="DDPM"),
        ]
        leg1 = ax.legend(handles=class_handles, title="Class", title_fontsize=13,
                         bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
        ax.legend(handles=source_handles, title="Source", title_fontsize=13,
                  bbox_to_anchor=(1.02, 0.45), loc="upper left", frameon=False)
        ax.add_artist(leg1)

        fig.tight_layout()
        _save_figure(fig, out_png, out_pdf, dpi=700)
        if show:
            plt.show()

    info = {
        "Xreal": Xreal,
        "yreal": ytr,
        "Xsyn": Xsyn,
        "ysyn": ysyn,
        "Z": Z,
        "pca_var": pca_var,
        "train_frac": train_frac,
        "n_real_plot": len(yr),
        "n_syn_plot": len(ys),
        "n_syn_total": len(ysyn),
    }
    return fig, ax, info

RANKED_ACCURACY_PLOT_VERSION = "reference_style_fixed_v5"

# =============================================================================
# Ranked accuracy figure from result CSV
# =============================================================================


def plot_ranked_accuracy_from_results(
    ml_csv,
    project_root=None,
    out_png="final_paper_ranked_bar_highlight.png",
    out_pdf="final_paper_ranked_bar_highlight.pdf",
    xlim=(72, 98),
    show=True,
    dpi=700,
    figsize=(7.4, 4.2),
    highlight_model="SVM",
    classical_order=("RF", "XGB", "LR", "SVM", "ANN", "KNN"),
    deep_results=None,
    deep_csv=None,
    deep_order=("EffNet-B3", "EffNet-B0", "RegNetY", "ViT-B16"),
    classical_results=None,
    use_manuscript_values=True,
    value_decimals=2,
    value_offset=0.18,
    right_label_padding=2.25,
):
    """Plot a paper-style Top-1 accuracy comparison figure.

    The upper block contains classical ML classifiers loaded from ``ml_csv``.
    The lower block contains deep-learning backbones loaded from ``deep_csv``
    or ``deep_results``. If neither is supplied, the manuscript backbone
    accuracies are used.

    The right x-axis limit is expanded automatically when necessary so value
    labels remain fully inside the axes. Therefore, a requested limit such as
    ``xlim=(72, 98)`` is treated as the minimum visible range rather than a
    rigid boundary that can clip the largest value label.

    Returns
    -------
    fig, ax, plot_df
        Matplotlib figure, axis, and the exact data used for plotting.
    """

    def _resolve_input_path(value):
        value = Path(value)
        if value.is_absolute():
            return value
        if project_root is not None:
            return Path(project_root) / value
        return value.resolve()

    def _resolve_output_path(path_value, source_parent):
        if path_value is None:
            return None
        path_value = Path(path_value)
        return path_value if path_value.is_absolute() else source_parent / path_value

    def _read_accuracy_table(csv_path):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed", case=False)]
        if df.empty:
            raise ValueError(f"CSV contains no data: {csv_path}")

        normalized_columns = {
            str(col).strip().lower().replace(" ", "_").replace("-", "_"): col
            for col in df.columns
        }

        model_candidates = (
            "method",
            "model",
            "classifier",
            "algorithm",
            "backbone",
            "model_name",
            "classifier_name",
            "method_name",
        )
        accuracy_candidates = (
            "accuracy",
            "acc",
            "accuracy_mean",
            "mean_accuracy",
            "avg_accuracy",
            "average_accuracy",
            "cv_accuracy",
            "test_accuracy",
            "top1_accuracy",
            "top_1_accuracy",
        )

        model_col = next(
            (
                normalized_columns[name]
                for name in model_candidates
                if name in normalized_columns
            ),
            None,
        )
        accuracy_col = next(
            (
                normalized_columns[name]
                for name in accuracy_candidates
                if name in normalized_columns
            ),
            None,
        )

        if model_col is None:
            text_cols = [
                col
                for col in df.columns
                if pd.api.types.is_object_dtype(df[col])
                or pd.api.types.is_string_dtype(df[col])
            ]
            if text_cols:
                model_col = text_cols[0]

        if accuracy_col is None:
            accuracy_like = [
                col
                for col in df.columns
                if "acc" in str(col).strip().lower()
            ]
            if accuracy_like:
                accuracy_col = accuracy_like[0]

        if model_col is None or accuracy_col is None:
            raise ValueError(
                "Cannot identify the model and accuracy columns. "
                f"Available columns: {list(df.columns)}"
            )

        raw_accuracy = (
            df[accuracy_col]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )

        result = pd.DataFrame(
            {
                "Model": df[model_col].astype(str).str.strip(),
                "Accuracy": pd.to_numeric(raw_accuracy, errors="coerce"),
            }
        )
        result = result.replace({"Model": {"": np.nan, "nan": np.nan}})
        result = result.dropna(subset=["Model", "Accuracy"]).copy()

        if result.empty:
            raise ValueError(f"No valid model/accuracy rows were found in {csv_path}")

        if float(result["Accuracy"].max()) <= 1.5:
            result["Accuracy"] *= 100.0

        return result

    def _canonical_model_name(value):
        raw = str(value).strip()
        normalized = _norm_method_name(raw)
        aliases = {
            "XGBOOST": "XGB",
            "XGB": "XGB",
            "EFFICIENTNET-B3": "EffNet-B3",
            "EFFICIENTNET B3": "EffNet-B3",
            "EFFNET-B3": "EffNet-B3",
            "EFFNET B3": "EffNet-B3",
            "B3": "EffNet-B3",
            "EFFICIENTNET-B0": "EffNet-B0",
            "EFFICIENTNET B0": "EffNet-B0",
            "EFFNET-B0": "EffNet-B0",
            "EFFNET B0": "EffNet-B0",
            "B0": "EffNet-B0",
            "REGNETY-400MF": "RegNetY",
            "REGNETY_400MF": "RegNetY",
            "REGNETY": "RegNetY",
            "VIT-B16": "ViT-B16",
            "VIT_B16": "ViT-B16",
            "VIT-B/16": "ViT-B16",
            "VITB16": "ViT-B16",
        }
        return aliases.get(normalized, normalized)

    ml_path = _resolve_input_path(ml_csv)

    manuscript_classical_results = {
        "RF": 94.91,
        "XGB": 95.31,
        "LR": 95.61,
        "SVM": 96.31,
        "ANN": 95.91,
        "KNN": 94.62,
    }

    if classical_results is not None:
        classical_map = {
            _canonical_model_name(key): float(value)
            for key, value in dict(classical_results).items()
        }
        classical_data_source = "classical_results"
    elif use_manuscript_values:
        classical_map = manuscript_classical_results.copy()
        classical_data_source = "manuscript_values"
    else:
        ml_df = _read_accuracy_table(ml_path)
        ml_df["RawModel"] = ml_df["Model"].astype(str)
        ml_df["Model"] = ml_df["Model"].map(_canonical_model_name)
        duplicate_models = ml_df.loc[ml_df.duplicated("Model", keep=False), "Model"].unique().tolist()
        if duplicate_models:
            raise ValueError(
                "Duplicate classifier rows were found after name normalization: "
                f"{duplicate_models}. Pass classical_results explicitly or keep "
                "use_manuscript_values=True to avoid accidental averaging."
            )
        classical_map = dict(zip(ml_df["Model"], ml_df["Accuracy"]))
        classical_data_source = "csv"

    classical_order = [_canonical_model_name(name) for name in classical_order]
    classical_rows = [
        {
            "Model": model,
            "Accuracy": float(classical_map[model]),
            "Category": "Classical ML",
        }
        for model in classical_order
        if model in classical_map
    ]

    if not classical_rows:
        raise ValueError(
            "No requested classical ML models were found. "
            f"Available models: {ml_df['Model'].tolist()}"
        )

    default_deep_results = {
        "EffNet-B3": 89.60,
        "EffNet-B0": 86.63,
        "RegNetY": 88.12,
        "ViT-B16": 85.64,
    }

    if deep_csv is not None:
        deep_path = _resolve_input_path(deep_csv)
        deep_df = _read_accuracy_table(deep_path)
        deep_df["Model"] = deep_df["Model"].map(_canonical_model_name)
        deep_df = (
            deep_df.sort_values(["Model", "Accuracy"], ascending=[True, False])
            .drop_duplicates(subset="Model", keep="first")
            .reset_index(drop=True)
        )
        deep_map = dict(zip(deep_df["Model"], deep_df["Accuracy"]))
    else:
        source_deep_results = default_deep_results if deep_results is None else deep_results
        deep_map = {
            _canonical_model_name(key): float(value)
            for key, value in dict(source_deep_results).items()
        }

    deep_order = [_canonical_model_name(name) for name in deep_order]
    deep_rows = [
        {
            "Model": model,
            "Accuracy": float(deep_map[model]),
            "Category": "Deep learning",
        }
        for model in deep_order
        if model in deep_map
    ]

    if not deep_rows:
        raise ValueError("No deep-learning results were available for plotting.")

    plot_df = pd.DataFrame(classical_rows + deep_rows)
    plot_df.insert(0, "Position", np.arange(1, len(plot_df) + 1))

    if highlight_model is None:
        highlight_label = str(plot_df.loc[plot_df["Accuracy"].idxmax(), "Model"])
    else:
        highlight_label = _canonical_model_name(highlight_model)
        if highlight_label not in set(plot_df["Model"]):
            raise ValueError(
                f"highlight_model={highlight_model!r} was not found. "
                f"Available models: {plot_df['Model'].tolist()}"
            )

    labels = plot_df["Model"].tolist()
    scores = plot_df["Accuracy"].to_numpy(dtype=float)
    categories = plot_df["Category"].tolist()
    y = np.arange(len(labels))

    best_color = "#C65A1E"
    ml_color = "#35679E"
    deep_color = "#B8C4D0"
    edge_color = "#333333"

    colors = [
        best_color
        if label == highlight_label
        else ml_color
        if category == "Classical ML"
        else deep_color
        for label, category in zip(labels, categories)
    ]

    if value_decimals < 0:
        raise ValueError("value_decimals must be non-negative.")
    if value_offset < 0:
        raise ValueError("value_offset must be non-negative.")
    if right_label_padding <= 0:
        raise ValueError("right_label_padding must be positive.")

    with plt.rc_context(
        paper_rc(base=10, axes=12, ticks=10, legend=9, title=12)
    ):
        fig, ax = plt.subplots(figsize=figsize, dpi=300)

        bars = ax.barh(
            y,
            scores,
            height=0.72,
            color=colors,
            edgecolor=edge_color,
            linewidth=0.75,
            zorder=3,
        )

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(
            "Top-1 Accuracy (%)",
            fontsize=12,
            fontweight="bold",
            labelpad=7,
        )
        ax.set_ylabel(
            "Model / Classifier",
            fontsize=12,
            fontweight="bold",
            labelpad=8,
        )

        if xlim is None:
            requested_xmin = max(0.0, float(np.floor(scores.min() - 3.0)))
            requested_xmax = float(np.ceil(scores.max() + 2.0))
        else:
            if len(xlim) != 2 or float(xlim[1]) <= float(xlim[0]):
                raise ValueError(
                    f"xlim must be an increasing two-value tuple, got {xlim}"
                )
            requested_xmin = float(xlim[0])
            requested_xmax = float(xlim[1])

        # Reserve enough data-space to display the longest numeric label.
        # For the manuscript figure this expands (72, 98) to about (72, 98.56),
        # keeping 96.31 fully inside the frame instead of overflowing.
        xmax_needed = float(scores.max() + value_offset + right_label_padding)
        final_xmax = max(requested_xmax, xmax_needed)
        if final_xmax <= requested_xmin:
            final_xmax = requested_xmin + 5.0
        ax.set_xlim(requested_xmin, final_xmax)

        for bar, value in zip(bars, scores):
            yc = bar.get_y() + bar.get_height() / 2
            ax.text(
                value + value_offset,
                yc,
                f"{value:.{value_decimals}f}",
                va="center",
                ha="left",
                fontsize=9.2,
                fontweight="bold",
                color="#222222",
                clip_on=True,
                zorder=4,
            )

        separator_y = len(classical_rows) - 0.5
        ax.axhline(
            separator_y,
            color="#A7A7A7",
            linewidth=0.8,
            zorder=2,
        )

        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(
            axis="both",
            which="major",
            direction="out",
            length=4.0,
            width=0.9,
            labelsize=10,
        )

        style_spines(
            ax,
            visible_sides=("left", "bottom", "top", "right"),
            linewidth=0.85,
            color="#666666",
        )

        ax.grid(
            axis="x",
            which="major",
            linestyle="--",
            linewidth=0.55,
            alpha=0.18,
            zorder=0,
        )
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)

        legend_handles = [
            Patch(
                facecolor=best_color,
                edgecolor=edge_color,
                label=f"Best model ({highlight_label})",
            ),
            Patch(
                facecolor=ml_color,
                edgecolor=edge_color,
                label="Classical ML",
            ),
            Patch(
                facecolor=deep_color,
                edgecolor=edge_color,
                label="Deep learning",
            ),
        ]
        ax.legend(
            handles=legend_handles,
            frameon=False,
            loc="lower right",
            bbox_to_anchor=(0.985, 0.02),
            borderaxespad=0.0,
            handlelength=1.7,
            handletextpad=0.6,
            labelspacing=0.35,
            fontsize=9.2,
        )

        fig.subplots_adjust(left=0.17, right=0.985, bottom=0.17, top=0.96)

        png_path = _resolve_output_path(out_png, ml_path.parent)
        pdf_path = _resolve_output_path(out_pdf, ml_path.parent)
        _save_figure(fig, png_path, pdf_path, dpi=dpi, pad_inches=0.04)

        if show:
            plt.show()

    plot_df.attrs["source_csv"] = str(ml_path)
    plot_df.attrs["png_path"] = str(png_path) if png_path is not None else None
    plot_df.attrs["pdf_path"] = str(pdf_path) if pdf_path is not None else None
    plot_df.attrs["highlight_model"] = highlight_label
    plot_df.attrs["plot_version"] = RANKED_ACCURACY_PLOT_VERSION
    plot_df.attrs["classical_data_source"] = classical_data_source
    plot_df.attrs["requested_xlim"] = (
        requested_xmin,
        requested_xmax,
    )
    plot_df.attrs["final_xlim"] = tuple(ax.get_xlim())
    return fig, ax, plot_df

# =============================================================================
# Final two-panel accuracy comparison
# =============================================================================

def plot_accuracy_two_panel_final(
    project_root,
    out_dir=None,
    stem="final_paper_accuracy_two_panel_FINAL_NO_GRID",
    show=True,
    dpi=700,
):
    """Plot the final two-panel accuracy comparison used in the manuscript.

    Panel (a) compares standalone deep-learning backbones.
    Panel (b) compares ML classifiers trained on EfficientNet-B3 features.
    """
    project_root = Path(project_root)

    out_dir = Path(out_dir) if out_dir is not None else project_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Panel (a): standalone deep-learning backbones
    dl_models = [
        "EfficientNet-B3",
        "RegNetY",
        "EfficientNet-B0",
        "ViT-B16",
    ]
    dl_acc = np.array([89.60, 88.12, 86.63, 85.64], dtype=float)

    # Panel (b): ML classifiers on EfficientNet-B3 features
    ml_models = [
        "SVM",
        "ANN",
        "LR",
        "XGB",
        "RF",
        "KNN",
    ]
    ml_acc = np.array([96.31, 95.91, 95.61, 95.31, 94.91, 94.62], dtype=float)

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.size": 21,
        "axes.labelsize": 23,
        "xtick.labelsize": 19,
        "ytick.labelsize": 21,
        "axes.linewidth": 1.15,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    c_dl = "#B7C3CF"
    c_ml = "#4A7FAE"
    c_best = "#D66018"
    c_edge = "#404040"

    with plt.rc_context(rc):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.2))

        xmin, xmax = 84, 98.3
        xticks = np.arange(84, 99, 2)

        # -----------------------------------------------------------------
        # Panel (a): deep-learning backbones
        # -----------------------------------------------------------------
        y1 = np.arange(len(dl_models))
        bars1 = ax1.barh(
            y1,
            dl_acc,
            height=0.66,
            color=[c_best if m == "EfficientNet-B3" else c_dl for m in dl_models],
            edgecolor=c_edge,
            linewidth=1.0,
            zorder=3,
        )

        ax1.set_yticks(y1)
        ax1.set_yticklabels(dl_models, fontsize=21)
        ax1.invert_yaxis()
        ax1.set_xlim(xmin, xmax)
        ax1.set_xticks(xticks)
        ax1.set_xlabel("Top-1 Accuracy (%)", fontsize=23, fontweight="normal", labelpad=13)
        ax1.set_ylabel("Deep-learning backbone", fontsize=23, fontweight="normal", labelpad=16)

        for model, bar, value in zip(dl_models, bars1, dl_acc):
            best = model == "EfficientNet-B3"
            ax1.text(
                value + 0.15,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                ha="left",
                fontsize=21,
                fontweight="bold" if best else "normal",
                clip_on=False,
            )

        for tick in ax1.get_yticklabels():
            tick.set_fontweight("bold" if tick.get_text() == "EfficientNet-B3" else "normal")

        # -----------------------------------------------------------------
        # Panel (b): ML classifiers
        # -----------------------------------------------------------------
        y2 = np.arange(len(ml_models))
        bars2 = ax2.barh(
            y2,
            ml_acc,
            height=0.66,
            color=[c_best if m == "SVM" else c_ml for m in ml_models],
            edgecolor=c_edge,
            linewidth=1.0,
            zorder=3,
        )

        ax2.set_yticks(y2)
        ax2.set_yticklabels(ml_models, fontsize=21)
        ax2.invert_yaxis()
        ax2.set_xlim(xmin, xmax)
        ax2.set_xticks(xticks)
        ax2.set_xlabel("Top-1 Accuracy (%)", fontsize=23, fontweight="normal", labelpad=13)
        ax2.set_ylabel("Classifier", fontsize=23, fontweight="normal", labelpad=16)

        for model, bar, value in zip(ml_models, bars2, ml_acc):
            best = model == "SVM"
            ax2.text(
                value + 0.15,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                ha="left",
                fontsize=21,
                fontweight="bold" if best else "normal",
                clip_on=False,
            )

        for tick in ax2.get_yticklabels():
            tick.set_fontweight("bold" if tick.get_text() == "SVM" else "normal")

        # -----------------------------------------------------------------
        # Common formatting
        # -----------------------------------------------------------------
        for ax in (ax1, ax2):
            ax.grid(False)
            ax.tick_params(
                axis="x",
                direction="out",
                length=6,
                width=1.1,
                labelsize=19,
            )
            ax.tick_params(
                axis="y",
                direction="out",
                length=5,
                width=1.1,
                labelsize=21,
            )
            for tick in ax.get_xticklabels():
                tick.set_fontweight("normal")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.15)
                spine.set_color("#555555")

        fig.subplots_adjust(
            left=0.17,
            right=0.985,
            bottom=0.20,
            top=0.80,
            wspace=0.31,
        )

        # -----------------------------------------------------------------
        # Panel labels and titles
        # -----------------------------------------------------------------
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        pos1 = ax1.get_position()
        pos2 = ax2.get_position()

        bbox_y1 = ax1.yaxis.label.get_window_extent(renderer=renderer)
        bbox_y2 = ax2.yaxis.label.get_window_extent(renderer=renderer)

        x_a_display = (bbox_y1.x0 + bbox_y1.x1) / 2
        x_b_display = (bbox_y2.x0 + bbox_y2.x1) / 2

        x_a_fig = fig.transFigure.inverted().transform((x_a_display, 0))[0]
        x_b_fig = fig.transFigure.inverted().transform((x_b_display, 0))[0]

        header_y = pos1.y1 + 0.055

        fig.text(
            x_a_fig,
            header_y,
            "a",
            fontsize=31,
            fontweight="bold",
            ha="center",
            va="center",
        )
        fig.text(
            (pos1.x0 + pos1.x1) / 2,
            header_y,
            "Standalone deep-learning backbones",
            fontsize=23,
            fontweight="normal",
            ha="center",
            va="center",
        )
        fig.text(
            x_b_fig,
            header_y,
            "b",
            fontsize=31,
            fontweight="bold",
            ha="center",
            va="center",
        )
        fig.text(
            (pos2.x0 + pos2.x1) / 2,
            header_y,
            "ML classifiers on EfficientNet-B3 features",
            fontsize=23,
            fontweight="normal",
            ha="center",
            va="center",
        )

        # -----------------------------------------------------------------
        # Save
        # -----------------------------------------------------------------
        png_path = out_dir / f"{stem}.png"
        pdf_path = out_dir / f"{stem}.pdf"

        fig.savefig(
            png_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        fig.savefig(
            pdf_path,
            bbox_inches="tight",
            facecolor="white",
        )

        if show:
            plt.show()

    print("PNG:", png_path)
    print("PDF:", pdf_path)

    return fig, (ax1, ax2), png_path, pdf_path


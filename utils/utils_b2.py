# ============================================================
# utils_b2.py
# Utilities for Figure 2(a,b,c) in HyMLRaman paper
# - Arial font
# - Standardized class abbreviations: AMX, CHL, CIP, IBU, PAR, TET
# - Consistent class colors across panels
# - Exports PNG + PDF + SVG
# ============================================================

from pathlib import Path
import sys
import subprocess
import importlib.util
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ============================================================
# GLOBAL STANDARD
# ============================================================

CANON_ORDER = ["AMX", "CHL", "CIP", "IBU", "PAR", "TET"]

NAME_MAP = {
    "amx": "AMX",
    "amoxicillin": "AMX",

    "chl": "CHL",
    "chlor": "CHL",
    "chloramphenicol": "CHL",

    "cip": "CIP",
    "cpf": "CIP",
    "ciprofloxacin": "CIP",

    "ibu": "IBU",
    "ibup": "IBU",
    "ibuprofen": "IBU",

    "par": "PAR",
    "para": "PAR",
    "paracetamol": "PAR",

    "tet": "TET",
    "tetra": "TET",
    "tetracycline": "TET",
}

CLASS_COLORS = {
    "AMX":   "#6A1B9A",  # purple
    "CHL": "#2E7D32",  # green
    "CIP":   "#1565C0",  # blue
    "IBU":  "#EF6C00",  # orange
    "PAR":  "#D7191C",  # red
    "TET": "#008C99",  # teal
}


def canonical_name(name):
    """Convert class names to the paper-style uppercase abbreviations."""
    text = str(name).strip()
    return NAME_MAP.get(text.lower(), text.upper())


def setup_figure_style(font="Arial"):
    """Set journal-style matplotlib defaults."""
    plt.rcParams["font.family"] = font
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"


def save_figure(fig, out_dir, stem, dpi=600):
    """Save one figure as PNG, PDF, and SVG."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_png = out_dir / f"{stem}.png"
    out_pdf = out_dir / f"{stem}.pdf"
    out_svg = out_dir / f"{stem}.svg"

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white")

    print("Saved:")
    print(out_png)
    print(out_pdf)
    print(out_svg)

    return out_png, out_pdf, out_svg


# ============================================================
# FIGURE 2(a) — CLASS DISTRIBUTION
# ============================================================

def plot_fig2a_class_distribution(
    class_counts,
    out_dir,
    stem="Figure2a_class_distribution_Arial_uppercase",
    fs_scale=1.5,
    show=True,
):
    setup_figure_style("Arial")

    counts_std = {canonical_name(k): v for k, v in class_counts.items()}
    values = [counts_std[k] for k in CANON_ORDER]
    colors = [CLASS_COLORS[k] for k in CANON_ORDER]

    fig, ax = plt.subplots(figsize=(8.6, 6.0), dpi=300)

    x = np.arange(len(CANON_ORDER))
    bars = ax.bar(
        x,
        values,
        color=colors,
        edgecolor="black",
        linewidth=1.2,
    )

    for rect, val in zip(bars, values):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 4,
            f"{val}",
            ha="center",
            va="bottom",
            fontsize=14 * fs_scale,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(CANON_ORDER, fontsize=16 * fs_scale)

    ax.set_ylabel(
        "Number of images",
        fontsize=20 * fs_scale,
        fontweight="normal",
        labelpad=10,
    )

    ax.set_xlabel(
        "Class",
        fontsize=20 * fs_scale,
        fontweight="normal",
        labelpad=8,
    )

    ax.set_ylim(0, 230)
    ax.set_yticks(np.arange(0, 231, 50))

    ax.tick_params(axis="y", labelsize=16 * fs_scale, width=1.2, length=6, pad=6)
    ax.tick_params(axis="x", labelsize=16 * fs_scale, width=1.2, length=6, pad=6)

    ax.set_title(
        "Class distribution",
        fontsize=22 * fs_scale,
        fontweight="bold",
        pad=16,
    )

    ax.text(
        -0.16,
        1.075,
        "(a)",
        transform=ax.transAxes,
        fontsize=22 * fs_scale,
        fontweight="bold",
        va="top",
        ha="left",
    )

    ax.grid(axis="y", linestyle="--", alpha=0.25)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.subplots_adjust(left=0.16, right=0.97, bottom=0.17, top=0.85)

    save_figure(fig, out_dir, stem)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


# ============================================================
# FIGURE 2(b) — REPRESENTATIVE RAMAN FINGERPRINTS
# ============================================================

def ensure_openpyxl():
    if importlib.util.find_spec("openpyxl") is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])


def find_data_file(path):
    path = Path(path)

    if path.is_file():
        return path

    if path.is_dir():
        files = []
        for ext in ["*.csv", "*.xlsx", "*.xls", "*.txt", "*.dat"]:
            files.extend(list(path.rglob(ext)))

        if len(files) == 0:
            raise FileNotFoundError(f"No data file found in: {path}")

        return sorted(files, key=lambda p: p.stat().st_size, reverse=True)[0]

    raise FileNotFoundError(path)


def load_table(path):
    path = Path(path)

    if path.suffix.lower() in [".xlsx", ".xls"]:
        ensure_openpyxl()

        sheets = pd.read_excel(
            path,
            sheet_name=None,
            header=None,
            engine="openpyxl",
        )

        best_df = None
        best_score = -1

        for _, df in sheets.items():
            score = 0
            for c in df.columns:
                score += pd.to_numeric(df[c], errors="coerce").notna().sum()

            if score > best_score:
                best_score = score
                best_df = df

        return best_df

    if path.suffix.lower() == ".csv":
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, header=None)

    return pd.read_csv(path, sep=None, engine="python", header=None)


def extract_xy_from_table(df, x_min=1100, x_max=1700, label=""):
    num = df.apply(lambda col: pd.to_numeric(col, errors="coerce"))
    cols = list(num.columns)

    best = None
    best_score = -1

    for i, c in enumerate(cols):
        x = num[c].to_numpy(float)
        finite_x = np.isfinite(x)

        if finite_x.sum() < 20:
            continue

        xv = x[finite_x]
        in_region = ((xv >= x_min) & (xv <= x_max)).sum()

        if len(xv) > 5:
            dx = np.diff(xv)
            monotonic_score = max((dx > 0).mean(), (dx < 0).mean())
        else:
            monotonic_score = 0

        for j in range(i + 1, min(i + 6, len(cols))):
            y = num[cols[j]].to_numpy(float)
            finite_y = np.isfinite(y)
            both = finite_x & finite_y

            if both.sum() < 20:
                continue

            yy = y[both]
            y_range = np.nanmax(yy) - np.nanmin(yy)

            if y_range <= 0:
                continue

            score = in_region + 100 * monotonic_score + min(both.sum(), 1000) * 0.02

            if score > best_score:
                best_score = score
                best = (c, cols[j])

    if best is None:
        raise ValueError(f"Cannot detect x/y columns for {label}")

    x_col, y_col = best

    x = num[x_col].to_numpy(float)
    y = num[y_col].to_numpy(float)

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    idx = np.argsort(x)
    x, y = x[idx], y[idx]

    mask = (x >= x_min) & (x <= x_max)
    x, y = x[mask], y[mask]

    if len(x) < 20:
        raise ValueError(f"{label}: not enough points in {x_min}-{x_max} cm-1")

    return x, y, x_col, y_col


def smooth(y, window=9):
    if window % 2 == 0:
        window += 1

    return (
        pd.Series(y)
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def rolling_baseline(y, window=301, q=0.08):
    return (
        pd.Series(y)
        .rolling(window=window, center=True, min_periods=1)
        .quantile(q)
        .rolling(window=max(31, window // 4), center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def preprocess_spectrum(x, y, cls, x_grid):
    cls = canonical_name(cls)

    y_grid = np.interp(x_grid, x, y)
    y_grid = np.nan_to_num(y_grid, nan=np.nanmedian(y_grid))

    if cls in ["IBU", "PAR"]:
        base = rolling_baseline(y_grid, window=451, q=0.14)
        y_corr = y_grid - base
        y_corr[y_corr < 0] = 0
        y_corr = smooth(y_corr, window=41)

    elif cls == "TET":
        base = rolling_baseline(y_grid, window=351, q=0.12)
        y_corr = y_grid - base
        y_corr[y_corr < 0] = 0
        y_corr = smooth(y_corr, window=31)

    else:
        base = rolling_baseline(y_grid, window=221, q=0.08)
        y_corr = y_grid - base
        y_corr[y_corr < 0] = 0
        y_corr = smooth(y_corr, window=11)

    p1, p99 = np.percentile(y_corr, [1, 99.5])
    y_norm = (y_corr - p1) / (p99 - p1 + 1e-12)
    y_norm = np.clip(y_norm, 0, 1)
    y_norm = smooth(y_norm, window=7)

    return np.clip(y_norm, 0, 1)


def default_raman_sources(db_dir):
    db_dir = Path(db_dir)
    return {
        "AMX": db_dir / "amoxiclin",
        "CHL": db_dir / "chloramphenicol_digitized_raw.csv",
        "CIP": db_dir / "cipro.xlsx",
        "IBU": db_dir / "ibuprofen do lai",
        "PAR": db_dir / "para",
        "TET": db_dir / "tetracycline",
    }


def plot_fig2b_raman_fingerprints(
    db_dir,
    out_dir,
    sources=None,
    stem="Figure2b_raman_fingerprints_Arial_uppercase",
    x_min=1100,
    x_max=1700,
    fs_scale=1.5,
    show=True,
):
    setup_figure_style("Arial")

    db_dir = Path(db_dir)
    out_dir = Path(out_dir)

    if sources is None:
        sources = default_raman_sources(db_dir)

    sources = {canonical_name(k): Path(v) for k, v in sources.items()}

    x_grid = np.linspace(x_min, x_max, 1201)
    spectra = {}

    print("Checking input paths:")
    for cls, p in sources.items():
        print(f"{cls:6s} -> {p} | exists = {p.exists()}")

    print("\nLoading spectra:")
    for cls in CANON_ORDER:
        data_file = find_data_file(sources[cls])
        df = load_table(data_file)
        x, y, x_col, y_col = extract_xy_from_table(df, x_min=x_min, x_max=x_max, label=cls)

        spectra[cls] = preprocess_spectrum(x, y, cls, x_grid)

        print(f"{cls:6s} | {data_file.name}")
        print(f"       x_col={x_col}, y_col={y_col}, points={len(x)}, range={x.min():.1f}-{x.max():.1f}")

    fig, ax = plt.subplots(figsize=(7.8, 4.9), dpi=300)

    amp = 0.42
    base_start = 0.15
    base_step = 0.62

    plot_order = ["TET", "PAR", "IBU", "CIP", "CHL", "AMX"]

    for i, cls in enumerate(plot_order):
        y_plot = base_start + i * base_step + amp * spectra[cls]
        ax.plot(x_grid, y_plot, lw=1.15 * fs_scale, color=CLASS_COLORS[cls], label=cls)

    ax.text(-0.14, 1.075, "(b)", transform=ax.transAxes,
            fontsize=12 * fs_scale, fontweight="bold", ha="left", va="top")

    ax.set_title("Representative Raman fingerprints",
                 fontsize=11.5 * fs_scale, fontweight="bold", pad=10)

    ax.set_xlabel(r"Raman shift (cm$^{-1}$)",
                  fontsize=10.5 * fs_scale, fontweight="normal", labelpad=8)

    ax.set_ylabel("Normalized intensity (a.u.)",
                  fontsize=10.5 * fs_scale, fontweight="normal", labelpad=8)

    ax.set_xlim(x_min, x_max)
    ax.set_xticks(np.arange(x_min, x_max + 1, 100))

    ax.set_ylim(0, 4.0)
    ax.set_yticks(np.arange(0, 4.1, 0.5))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="both", labelsize=8.6 * fs_scale, width=1.1, length=4.5, pad=6)

    handles, labels = ax.get_legend_handles_labels()
    hmap = dict(zip(labels, handles))

    ax.legend(
        [hmap[c] for c in CANON_ORDER],
        CANON_ORDER,
        frameon=False,
        fontsize=8.2 * fs_scale,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        handlelength=1.8,
        borderaxespad=0.2,
    )

    plt.subplots_adjust(left=0.15, right=0.80, bottom=0.18, top=0.86)

    save_figure(fig, out_dir, stem)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax, spectra


# ============================================================
# FIGURE 2(c) — FEATURE-SPACE VISUALIZATION
# ============================================================

def load_class_names(class_json):
    with open(class_json, "r", encoding="utf-8") as f:
        raw_class_names = json.load(f)

    class_names = [canonical_name(cls) for cls in raw_class_names]
    return raw_class_names, class_names


def plot_fig2c_feature_space(
    feature_path,
    label_path,
    class_json,
    out_dir,
    stem="Figure2c_real_tsne_feature_space_Arial_uppercase",
    cache_name="figure2c_tsne_embedding_seed1337.npy",
    force_recompute=False,
    fs_scale=1.5,
    show=True,
):
    setup_figure_style("Arial")

    feature_path = Path(feature_path)
    label_path = Path(label_path)
    class_json = Path(class_json)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(feature_path).astype(np.float32)
    y = np.load(label_path).astype(int).ravel()

    raw_class_names, class_names = load_class_names(class_json)

    print("Feature path:", feature_path)
    print("Label path:", label_path)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Raw classes:", raw_class_names)
    print("Standardized classes:", class_names)

    count_df = pd.DataFrame({
        "Class": class_names,
        "Count": [int(np.sum(y == i)) for i in range(len(class_names))]
    })
    print(count_df)

    X_scaled = StandardScaler().fit_transform(X)

    n_pca = min(50, X_scaled.shape[0] - 1, X_scaled.shape[1])
    X_pca = PCA(n_components=n_pca, random_state=1337).fit_transform(X_scaled)

    print("PCA input to t-SNE:", X_pca.shape)

    cache_tsne = out_dir / cache_name

    if cache_tsne.exists() and not force_recompute:
        X_2d = np.load(cache_tsne)
        print("Loaded cached t-SNE:", cache_tsne)
    else:
        try:
            tsne = TSNE(
                n_components=2,
                perplexity=35,
                learning_rate="auto",
                init="pca",
                max_iter=1500,
                random_state=1337,
                metric="euclidean",
            )
        except TypeError:
            tsne = TSNE(
                n_components=2,
                perplexity=35,
                learning_rate=300,
                init="pca",
                n_iter=1500,
                random_state=1337,
                metric="euclidean",
            )

        X_2d = tsne.fit_transform(X_pca)
        np.save(cache_tsne, X_2d)
        print("Saved cached t-SNE:", cache_tsne)

    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    fig, ax = plt.subplots(figsize=(7.8, 5.4), dpi=300)

    for cls in CANON_ORDER:
        if cls not in class_to_idx:
            print(f"Warning: {cls} not found in classes.json")
            continue

        idx = y == class_to_idx[cls]

        ax.scatter(
            X_2d[idx, 0],
            X_2d[idx, 1],
            s=22,
            alpha=0.78,
            c=CLASS_COLORS[cls],
            edgecolors="white",
            linewidths=0.35,
            label=cls,
        )

    ax.axhline(0, color="0.65", lw=1.0, ls="--", zorder=0)
    ax.axvline(0, color="0.65", lw=1.0, ls="--", zorder=0)

    ax.set_xlabel("Component 1", fontsize=15 * fs_scale, fontweight="normal", labelpad=8)
    ax.set_ylabel("Component 2", fontsize=15 * fs_scale, fontweight="normal", labelpad=8)

    ax.set_title("Feature-space visualization",
                 fontsize=16 * fs_scale, fontweight="bold", pad=12)

    ax.text(-0.16, 1.08, "(c)", transform=ax.transAxes,
            fontsize=18 * fs_scale, fontweight="bold", va="top", ha="left")

    leg = ax.legend(
        frameon=True,
        fontsize=10.5 * fs_scale,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderpad=0.7,
        labelspacing=0.65,
        handletextpad=0.5,
        markerscale=1.2,
    )

    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_alpha(1.0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="both", labelsize=11 * fs_scale, width=1.2, length=4.5, pad=6)

    plt.subplots_adjust(left=0.15, right=0.78, bottom=0.16, top=0.86)

    save_figure(fig, out_dir, stem)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax, X_2d, count_df


# ============================================================
# QUICK RUN EXAMPLE
# ============================================================

if __name__ == "__main__":
    PROJECT_ROOT = Path(r"D:\HyRaman")
    DB_DIR = PROJECT_ROOT / "database"
    OUT_DIR = PROJECT_ROOT / "cnn" / "abx_final_tuned" / "figure2_database"

    class_counts = {
        "AMX": 154,
        "CHL": 178,
        "CIP": 178,
        "IBU": 150,
        "PAR": 200,
        "TET": 143,
    }

    plot_fig2a_class_distribution(
        class_counts=class_counts,
        out_dir=OUT_DIR,
        show=True,
    )

    plot_fig2b_raman_fingerprints(
        db_dir=DB_DIR,
        out_dir=OUT_DIR,
        show=True,
    )

    plot_fig2c_feature_space(
        feature_path=PROJECT_ROOT / "cnn" / "abx_final_tuned" / "features_b3_1536.npy",
        label_path=PROJECT_ROOT / "cnn" / "abx_final_tuned" / "labels_b3.npy",
        class_json=PROJECT_ROOT / "cnn" / "abx_final_tuned" / "classes.json",
        out_dir=OUT_DIR,
        show=True,
    )
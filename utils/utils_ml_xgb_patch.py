# utils_ml_xgb_patch.py
# ------------------------------------------------------------
# XGBoost 10-fold CV: Confusion Matrix + ROC
# Copy this file into: D:\HyRaman\utils\utils_ml_xgb_patch.py
# Import using:
# from utils.utils_ml_xgb_patch import run_xgb_cm_roc_10fold
# ------------------------------------------------------------


def _apply_major_ticks(ax, x_major=None, y_major=None, xlim=None, ylim=None, labelsize=22):
    """Major ticks only, visible on both axes."""
    from matplotlib.ticker import MultipleLocator, NullLocator

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
    ax.tick_params(axis="x", which="major", bottom=True, top=False, direction="out", length=5.2, width=1.05, labelsize=labelsize, pad=6)
    ax.tick_params(axis="y", which="major", left=True, right=False, direction="out", length=5.2, width=1.05, labelsize=labelsize, pad=6)
    ax.tick_params(axis="both", which="minor", bottom=False, top=False, left=False, right=False, length=0)
    return ax


def run_xgb_cm_roc_10fold(
    root,
    feat_name="features_b3_1536.npy",
    label_name="labels_b3.npy",
    class_name="classes.json",
    fallback_class_name="class_to_idx.json",
    out_cm_png_name="cm_xgb.png",
    out_roc_png_name="roc_xgb.png",
    out_cm_pdf_name="cm_xgb.pdf",
    out_roc_pdf_name="roc_xgb.pdf",
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
    """
    Run XGBoost 10-fold CV analysis:
    - classification report
    - confusion matrix CSV + PNG/PDF
    - ROC CSV + PNG/PDF

    Paper plotting style:
    - larger labels / tick numbers
    - visible major tick marks on x and y axes
    - no minor ticks
    """
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.preprocessing import StandardScaler, label_binarize
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
    from xgboost import XGBClassifier

    root = Path(root)

    feat_path = root / feat_name
    label_path = root / label_name
    class_path = root / class_name
    fallback_path = root / fallback_class_name

    out_cm_png = root / out_cm_png_name
    out_roc_png = root / out_roc_png_name
    out_cm_pdf = root / out_cm_pdf_name
    out_roc_pdf = root / out_roc_pdf_name
    out_cm_csv = root / out_cm_csv_name
    out_roc_csv = root / out_roc_csv_name

    if not feat_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feat_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    X = np.load(feat_path).astype(np.float32)
    y = np.load(label_path).astype(np.int64).ravel()

    if class_path.exists():
        with open(class_path, "r", encoding="utf-8") as f:
            names = json.load(f)
    elif fallback_path.exists():
        with open(fallback_path, "r", encoding="utf-8") as f:
            c2i = json.load(f)
        names = [None] * len(c2i)
        for k, v in c2i.items():
            names[int(v)] = k
    else:
        names = [str(i) for i in sorted(np.unique(y))]

    names = [str(x) for x in names]
    n_classes = len(names)

    if verbose:
        print(f"Loaded: X={X.shape}, y={y.shape}")
        print("Classes:", names)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)

    model = XGBClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        eval_metric="mlogloss",
        tree_method=tree_method,
        n_jobs=n_jobs,
        random_state=model_seed,
    )

    y_true_all, y_pred_all = [], []
    proba_all, y_true_bin_all = [], []

    for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xva = scaler.transform(X[va])

        model.fit(Xtr, y[tr])
        pred = model.predict(Xva)
        proba = model.predict_proba(Xva)

        # Align probability columns to class order 0..n_classes-1.
        aligned = np.zeros((len(va), n_classes), dtype=float)
        for j, cls in enumerate(model.classes_):
            cls = int(cls)
            if 0 <= cls < n_classes:
                aligned[:, cls] = proba[:, j]

        y_pred_all.extend(pred)
        y_true_all.extend(y[va])
        proba_all.append(aligned)
        y_true_bin_all.append(label_binarize(y[va], classes=np.arange(n_classes)))

        if verbose:
            print(f"[Fold {fold:02d}] done")

    y_true_all = np.asarray(y_true_all)
    y_pred_all = np.asarray(y_pred_all)
    y_score = np.vstack(proba_all)
    y_true_bin = np.vstack(y_true_bin_all)

    report = classification_report(y_true_all, y_pred_all, target_names=names, digits=4, zero_division=0)
    if verbose:
        print("\n====== CLASSIFICATION REPORT (XGBoost, 10-fold CV) ======\n")
        print(report)

    cm = confusion_matrix(y_true_all, y_pred_all, labels=np.arange(n_classes))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0

    pd.DataFrame(cm_pct, index=names, columns=names).to_csv(out_cm_csv, encoding="utf-8-sig", float_format="%.1f")

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
    pd.DataFrame(rows).to_csv(out_roc_csv, index=False, float_format="%.6f", encoding="utf-8-sig")

    fig_cm = ax_cm = fig_roc = ax_roc = None

    if plot:
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.ticker import MultipleLocator, NullLocator

        with plt.rc_context({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 22,
            "axes.titlesize": 28,
            "axes.labelsize": 26,
            "xtick.labelsize": 23,
            "ytick.labelsize": 23,
            "legend.fontsize": 15,
            "axes.linewidth": 1.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }):
            # Confusion matrix figure.
            cmap = LinearSegmentedColormap.from_list(
                "paper_xgb_orange",
                ["#fffdf8", "#fbe8c8", "#f3c98b", "#e59644", "#c8641a"],
            )
            fig_cm, ax_cm = plt.subplots(figsize=(6.3, 5.7), dpi=300)
            hm = sns.heatmap(
                cm_pct,
                ax=ax_cm,
                annot=True,
                fmt=".1f",
                cmap=cmap,
                vmin=0,
                vmax=100,
                square=True,
                xticklabels=names,
                yticklabels=names,
                linewidths=0,
                cbar=True,
                annot_kws={"size": 20},
                cbar_kws={"label": "Recall (%)", "shrink": 0.76},
            )

            ax_cm.set_title("Confusion Matrix – XGBoost (10-fold CV)", pad=10)
            ax_cm.set_xlabel("Predicted label", labelpad=8)
            ax_cm.set_ylabel("True label", labelpad=8)
            ax_cm.set_xticks(np.arange(n_classes) + 0.5)
            ax_cm.set_yticks(np.arange(n_classes) + 0.5)
            ax_cm.set_xticklabels(names, rotation=45, ha="right", rotation_mode="anchor")
            ax_cm.set_yticklabels(names, rotation=0, va="center")
            ax_cm.xaxis.set_minor_locator(NullLocator())
            ax_cm.yaxis.set_minor_locator(NullLocator())
            ax_cm.xaxis.set_ticks_position("bottom")
            ax_cm.yaxis.set_ticks_position("left")
            ax_cm.tick_params(axis="x", which="major", bottom=True, top=False, direction="out", length=5.2, width=1.05, pad=4)
            ax_cm.tick_params(axis="y", which="major", left=True, right=False, direction="out", length=5.2, width=1.05, pad=4)
            ax_cm.tick_params(axis="both", which="minor", bottom=False, top=False, left=False, right=False, length=0)

            for text, val in zip(ax_cm.texts, cm_pct.flatten()):
                text.set_color("white" if val >= 75 else "#222222")

            for spine in ax_cm.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.05)
                spine.set_color("#7a7a7a")

            cbar = hm.collections[0].colorbar
            cbar.ax.tick_params(labelsize=22, length=5.2, width=1.05, direction="out")
            cbar.set_label("Recall (%)", fontsize=25, labelpad=10)

            fig_cm.tight_layout(pad=0.55)
            fig_cm.savefig(out_cm_png, dpi=700, bbox_inches="tight")
            fig_cm.savefig(out_cm_pdf, bbox_inches="tight")
            if show:
                plt.show()

            # ROC figure.
            fig_roc, ax_roc = plt.subplots(figsize=(6.2, 5.2), dpi=300)
            colors = plt.cm.tab10.colors
            for i in range(n_classes):
                ax_roc.plot(fpr[i], tpr[i], lw=1.55, color=colors[i % len(colors)], label=f"{names[i]} (AUC={roc_auc[i]:.3f})")
            ax_roc.plot(fpr["micro"], tpr["micro"], linestyle="--", color="black", lw=1.8, label=f"micro (AUC={roc_auc['micro']:.3f})")
            ax_roc.plot([0, 1], [0, 1], linestyle=":", color="gray", lw=1.0)

            ax_roc.set_xlim(0.0, 0.30)
            ax_roc.set_ylim(0.80, 1.00)
            ax_roc.xaxis.set_major_locator(MultipleLocator(0.05))
            ax_roc.yaxis.set_major_locator(MultipleLocator(0.05))
            ax_roc.xaxis.set_minor_locator(NullLocator())
            ax_roc.yaxis.set_minor_locator(NullLocator())
            ax_roc.xaxis.set_ticks_position("bottom")
            ax_roc.yaxis.set_ticks_position("left")
            ax_roc.tick_params(axis="both", which="major", direction="out", length=5.2, width=1.05, labelsize=23, pad=6)
            ax_roc.tick_params(axis="both", which="minor", bottom=False, top=False, left=False, right=False, length=0)

            ax_roc.set_xlabel("False Positive Rate", labelpad=8)
            ax_roc.set_ylabel("True Positive Rate", labelpad=8)
            ax_roc.set_title("ROC Curves – XGBoost (10-fold CV)", pad=10)
            ax_roc.grid(False)
            for spine in ax_roc.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.05)
                spine.set_color("#333333")
            ax_roc.legend(loc="lower right", ncol=1, frameon=False, fontsize=14, handlelength=1.8, labelspacing=0.22)
            fig_roc.tight_layout(pad=0.55)
            fig_roc.savefig(out_roc_png, dpi=700, bbox_inches="tight")
            fig_roc.savefig(out_roc_pdf, bbox_inches="tight")
            if show:
                plt.show()

    if verbose:
        print("\nSaved CM PNG  ->", out_cm_png)
        print("Saved CM PDF  ->", out_cm_pdf)
        print("Saved ROC PNG ->", out_roc_png)
        print("Saved ROC PDF ->", out_roc_pdf)
        print("Saved CM CSV  ->", out_cm_csv)
        print("Saved ROC CSV ->", out_roc_csv)
        print(f"micro-AUC={roc_auc['micro']:.4f} | macro-AUC={roc_auc['macro']:.4f}")

    return {
        "report": report,
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "y_score": y_score,
        "y_true_bin": y_true_bin,
        "cm": cm,
        "cm_pct": cm_pct,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "fig_cm": fig_cm,
        "ax_cm": ax_cm,
        "fig_roc": fig_roc,
        "ax_roc": ax_roc,
        "out_cm_png": out_cm_png,
        "out_cm_pdf": out_cm_pdf,
        "out_roc_png": out_roc_png,
        "out_roc_pdf": out_roc_pdf,
        "out_cm_csv": out_cm_csv,
        "out_roc_csv": out_roc_csv,
    }

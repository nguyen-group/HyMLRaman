import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from torchvision import transforms


MODEL_PATH = r"D:\VIET BAO CNN\cnn\abx_b3_reg\best_by_acc.pt"

OUTPUT_PROFILE = os.path.join(
    os.path.dirname(MODEL_PATH),
    "raman_known_only_domain_profile.npz",
)

CLASSES = ["AMX", "CHL", "CIP", "IBU", "PAR", "TET"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 320
SEED = 1337
REFERENCE_RATIO = 0.80
SOURCE_BATCH_SIZE = 10
MAX_PCA_COMPONENTS = 48
MIN_IMAGES_PER_CLASS = 8
TOPK_REFERENCE_K = 5
CONFIDENCE_FLOOR = 0.50

VIEW_WEIGHTS = np.asarray(
    [0.40, 0.25, 0.20, 0.15],
    dtype=np.float32,
)

SCORE_WEIGHTS = np.asarray(
    [1.00, 1.20, 1.00, 0.60, 0.50, 0.35],
    dtype=np.float32,
)

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp",
    ".tif", ".tiff", ".webp",
}

ALIASES = {
    "amx": "AMX",
    "amoxicillin": "AMX",
    "amoxiclin": "AMX",
    "chl": "CHL",
    "chlor": "CHL",
    "chloramphenicol": "CHL",
    "cip": "CIP",
    "cpf": "CIP",
    "cipro": "CIP",
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

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def normalize_token(value):
    return (
        str(value).strip().lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def infer_class_from_path(path):
    for part in reversed(path.parts[:-1]):
        token = normalize_token(part)

        if token in ALIASES:
            return ALIASES[token]

        for alias, canonical in ALIASES.items():
            if alias in token:
                return canonical

    return None


def collect_labeled_images(root):
    items = []

    for path in Path(root).rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        class_name = infer_class_from_path(path)

        if class_name in CLASSES:
            items.append((path, CLASSES.index(class_name)))

    return items


def stratified_split(items):
    rng = random.Random(SEED)
    reference = []
    calibration = []

    for class_index in range(len(CLASSES)):
        class_items = [
            item for item in items
            if item[1] == class_index
        ]
        rng.shuffle(class_items)

        split_index = int(
            round(len(class_items) * REFERENCE_RATIO)
        )
        split_index = min(
            max(split_index, 1),
            len(class_items) - 1,
        )

        reference.extend(class_items[:split_index])
        calibration.extend(class_items[split_index:])

    return reference, calibration


def build_views(path):
    image = Image.open(path).convert("RGB")
    gray = ImageOps.grayscale(image)

    auto_gray = ImageOps.autocontrast(gray, cutoff=1)
    auto_rgb = auto_gray.convert("RGB")

    contrast_gray = ImageEnhance.Contrast(
        auto_gray
    ).enhance(1.35)

    contrast_gray = contrast_gray.filter(
        ImageFilter.UnsharpMask(
            radius=1.0,
            percent=110,
            threshold=2,
        )
    )
    contrast_rgb = contrast_gray.convert("RGB")

    gray_array = np.asarray(gray, dtype=np.uint8)
    clahe_array = cv2.createCLAHE(
        clipLimit=1.8,
        tileGridSize=(8, 8),
    ).apply(gray_array)

    clahe_rgb = Image.fromarray(
        clahe_array,
        mode="L",
    ).convert("RGB")

    return [image, auto_rgb, contrast_rgb, clahe_rgb]


def load_model():
    model = EfficientNet.from_name(
        "efficientnet-b3",
        num_classes=len(CLASSES),
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    state_dict = {
        key.replace("module.", "", 1): value
        for key, value in state_dict.items()
    }

    model.load_state_dict(state_dict, strict=False)
    return model.to(DEVICE).eval()


@torch.no_grad()
def extract_view_outputs(model, batch):
    features = model.extract_features(batch)
    pooled = model._avg_pooling(features)
    flat = pooled.flatten(start_dim=1)

    embeddings = F.normalize(flat, p=2, dim=1)
    logits = model._fc(model._dropout(flat))
    probabilities = torch.softmax(logits, dim=1)

    return (
        embeddings.cpu().numpy().astype(np.float32),
        probabilities.cpu().numpy().astype(np.float32),
    )


def aggregate_outputs(view_embeddings, view_probabilities):
    n_views = len(VIEW_WEIGHTS)

    view_embeddings = view_embeddings.reshape(
        -1, n_views, view_embeddings.shape[-1]
    )
    view_probabilities = view_probabilities.reshape(
        -1, n_views, view_probabilities.shape[-1]
    )

    weights = VIEW_WEIGHTS / VIEW_WEIGHTS.sum()

    embeddings = np.average(
        view_embeddings,
        axis=1,
        weights=weights,
    )
    embeddings /= np.clip(
        np.linalg.norm(embeddings, axis=1, keepdims=True),
        1e-12,
        None,
    )

    probabilities = np.average(
        view_probabilities,
        axis=1,
        weights=weights,
    )

    view_predictions = np.argmax(
        view_probabilities,
        axis=2,
    )

    return (
        embeddings.astype(np.float32),
        probabilities.astype(np.float32),
        view_predictions.astype(np.int64),
    )


def embed_items(model, items, label):
    embeddings = []
    probabilities = []
    view_predictions = []
    labels = []
    paths = []

    for start in range(0, len(items), SOURCE_BATCH_SIZE):
        source_batch = items[start:start + SOURCE_BATCH_SIZE]
        tensors = []
        valid_items = []

        for path, class_index in source_batch:
            try:
                views = build_views(path)
                tensors.extend(transform(view) for view in views)
                valid_items.append((path, class_index))
            except Exception as error:
                print(f"Skipped: {path} | {error}")

        if valid_items:
            batch = torch.stack(tensors, dim=0).to(DEVICE)

            view_emb, view_prob = extract_view_outputs(
                model,
                batch,
            )

            emb, prob, view_pred = aggregate_outputs(
                view_emb,
                view_prob,
            )

            embeddings.append(emb)
            probabilities.append(prob)
            view_predictions.append(view_pred)
            labels.extend(class_index for _, class_index in valid_items)
            paths.extend(str(path) for path, _ in valid_items)

        processed = min(start + SOURCE_BATCH_SIZE, len(items))
        print(f"{label}: {processed}/{len(items)}")

    if not embeddings:
        raise RuntimeError(f"No readable images remained in {label}.")

    return {
        "embeddings": np.vstack(embeddings).astype(np.float32),
        "probabilities": np.vstack(probabilities).astype(np.float32),
        "view_predictions": np.vstack(view_predictions).astype(np.int64),
        "labels": np.asarray(labels, dtype=np.int64),
        "paths": np.asarray(paths),
    }


def topk_mean_similarity(queries, bank, k):
    k = int(min(max(1, k), bank.shape[0]))
    similarities = queries @ bank.T
    topk = np.partition(
        similarities,
        similarities.shape[1] - k,
        axis=1,
    )[:, -k:]
    return topk.mean(axis=1)


def mahalanobis_squared(values, mean, precision):
    delta = values - mean
    return np.einsum(
        "ni,ij,nj->n",
        delta,
        precision,
        delta,
    )


def robust_location_scale(values):
    median = np.median(values, axis=0)
    mad = np.median(
        np.abs(values - median),
        axis=0,
    )
    scale = 1.4826 * mad

    fallback = np.std(values, axis=0)
    scale = np.where(scale > 1e-6, scale, fallback)
    scale = np.where(scale > 1e-6, scale, 1e-3)

    return median.astype(np.float32), scale.astype(np.float32)


def compute_components(
    embeddings,
    probabilities,
    view_predictions,
    class_index,
    reference_embeddings,
    reference_labels,
    centroids,
    pca_values,
    class_pca_mean,
    class_precision,
):
    centroid_similarities = embeddings @ centroids.T

    true_centroid_similarity = centroid_similarities[:, class_index]
    sorted_centroid = np.sort(centroid_similarities, axis=1)
    centroid_gap = sorted_centroid[:, -1] - sorted_centroid[:, -2]

    class_bank = reference_embeddings[
        reference_labels == class_index
    ]

    topk_similarity = topk_mean_similarity(
        embeddings,
        class_bank,
        TOPK_REFERENCE_K,
    )

    mahalanobis = mahalanobis_squared(
        pca_values,
        class_pca_mean,
        class_precision,
    ) / max(pca_values.shape[1], 1)

    view_agreement = np.mean(
        view_predictions == class_index,
        axis=1,
    )

    class_confidence = probabilities[:, class_index]

    components = np.column_stack([
        1.0 - true_centroid_similarity,
        1.0 - topk_similarity,
        mahalanobis,
        np.maximum(0.0, 0.10 - centroid_gap),
        1.0 - view_agreement,
        1.0 - class_confidence,
    ])

    return components.astype(np.float32)


def build_profile(reference_root, calibration_root=None):
    reference_items = collect_labeled_images(reference_root)

    if calibration_root:
        calibration_items = collect_labeled_images(
            calibration_root
        )
        split_description = (
            "separate known-domain calibration folder"
        )
    else:
        reference_items, calibration_items = stratified_split(
            reference_items
        )
        split_description = "automatic 80/20 known-only split"

    reference_counts = np.bincount(
        [label for _, label in reference_items],
        minlength=len(CLASSES),
    )
    calibration_counts = np.bincount(
        [label for _, label in calibration_items],
        minlength=len(CLASSES),
    )

    print("\nSplit:", split_description)
    print("\nReference / calibration counts:")

    insufficient = []

    for index, class_name in enumerate(CLASSES):
        ref_count = int(reference_counts[index])
        cal_count = int(calibration_counts[index])

        print(
            f"  {class_name}: "
            f"reference={ref_count}, calibration={cal_count}"
        )

        if ref_count < MIN_IMAGES_PER_CLASS or cal_count < 3:
            insufficient.append(class_name)

    if insufficient:
        raise RuntimeError(
            "Insufficient known-only data for: "
            f"{sorted(set(insufficient))}. "
            "Use at least 8 reference and 3 calibration images per class."
        )

    model = load_model()

    reference = embed_items(
        model,
        reference_items,
        "Reference",
    )
    calibration = embed_items(
        model,
        calibration_items,
        "Calibration",
    )

    reference_embeddings = reference["embeddings"]
    reference_labels = reference["labels"]

    n_components = min(
        MAX_PCA_COMPONENTS,
        reference_embeddings.shape[0] - len(CLASSES),
        reference_embeddings.shape[1],
    )
    n_components = max(8, n_components)

    pca = PCA(
        n_components=n_components,
        random_state=SEED,
    )
    reference_pca = pca.fit_transform(
        reference_embeddings
    ).astype(np.float32)
    calibration_pca = pca.transform(
        calibration["embeddings"]
    ).astype(np.float32)

    centroids = []
    class_pca_means = []
    class_precisions = []

    for class_index in range(len(CLASSES)):
        class_reference = reference_embeddings[
            reference_labels == class_index
        ]

        centroid = class_reference.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-12
        centroids.append(centroid.astype(np.float32))

        class_reference_pca = reference_pca[
            reference_labels == class_index
        ]

        covariance = LedoitWolf().fit(class_reference_pca)
        class_pca_means.append(
            covariance.location_.astype(np.float32)
        )
        class_precisions.append(
            covariance.precision_.astype(np.float32)
        )

    centroids = np.vstack(centroids).astype(np.float32)
    class_pca_means = np.vstack(class_pca_means).astype(np.float32)
    class_precisions = np.stack(class_precisions).astype(np.float32)

    component_medians = []
    component_scales = []
    score_thresholds = []

    print("\nDomain-calibrated known-only thresholds:")

    for class_index, class_name in enumerate(CLASSES):
        mask = calibration["labels"] == class_index

        components = compute_components(
            calibration["embeddings"][mask],
            calibration["probabilities"][mask],
            calibration["view_predictions"][mask],
            class_index,
            reference_embeddings,
            reference_labels,
            centroids,
            calibration_pca[mask],
            class_pca_means[class_index],
            class_precisions[class_index],
        )

        median, scale = robust_location_scale(components)

        z = np.maximum(
            0.0,
            (components - median) / scale,
        )
        scores = z @ SCORE_WEIGHTS

        threshold = float(
            max(
                np.percentile(scores, 99.0) + 0.25,
                3.0,
            )
        )

        component_medians.append(median)
        component_scales.append(scale)
        score_thresholds.append(threshold)

        print(
            f"  {class_name}: score <= {threshold:.3f} "
            f"(median={np.median(scores):.3f}, "
            f"max={np.max(scores):.3f})"
        )

    output_path = Path(OUTPUT_PROFILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        class_names=np.asarray(CLASSES),
        reference_embeddings=reference_embeddings,
        reference_labels=reference_labels,
        centroids=centroids,
        pca_mean=pca.mean_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        class_pca_means=class_pca_means,
        class_precisions=class_precisions,
        component_medians=np.vstack(component_medians).astype(np.float32),
        component_scales=np.vstack(component_scales).astype(np.float32),
        score_thresholds=np.asarray(score_thresholds, dtype=np.float32),
        score_weights=SCORE_WEIGHTS,
        topk_reference_k=np.asarray([TOPK_REFERENCE_K], dtype=np.int64),
        confidence_floor=np.asarray([CONFIDENCE_FLOOR], dtype=np.float32),
        profile_version=np.asarray(["DOMAIN_KNOWN_ONLY_V1"]),
        split_description=np.asarray([split_description]),
        reference_paths=reference["paths"],
        calibration_paths=calibration["paths"],
    )

    print("\nSaved profile:")
    print(output_path)
    print("No unknown spectra were used.")


def select_folder(title):
    return QFileDialog.getExistingDirectory(
        None,
        title,
        "",
    )


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    app = QApplication.instance() or QApplication(sys.argv)

    reference_root = select_folder(
        "Select the main six-drug reference dataset"
    )

    if not reference_root:
        sys.exit(0)

    calibration_root = select_folder(
        "Select a separate KNOWN six-drug calibration folder "
        "(Cancel to use automatic 80/20 split)"
    )

    try:
        build_profile(
            reference_root,
            calibration_root or None,
        )

        QMessageBox.information(
            None,
            "Profile created",
            (
                "Known-only domain-calibrated profile created:\n"
                f"{OUTPUT_PROFILE}\n\n"
                "No unknown spectra were used."
            ),
        )

    except Exception as error:
        QMessageBox.critical(
            None,
            "Profile error",
            str(error),
        )
        raise
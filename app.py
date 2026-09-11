import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from PyQt6.QtCore import Qt, QTimer, QSizeF, QMarginsF
from PyQt6.QtGui import QColor, QFont, QPixmap, QImage, QPainter, QPdfWriter, QPageSize, QPageLayout
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from torchvision import transforms


# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = r"./models/best_by_acc_12092026.pt"

CLASSES = [
    "AMX",
    "CHL",
    "CIP",
    "IBU",
    "PAR",
    "TET",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 320

# Classification acceptance rule.
# The input-type gate is checked first. A valid spectrum is classified when
# the ensemble top-1 confidence reaches 45%.
MIN_ACCEPT_CONFIDENCE = 0.50

# Known-only open-set profile. This profile is built exclusively from the
# six drug classes and their held-out known calibration images.
KNOWN_ONLY_PROFILE_PATH = os.path.join(
    os.path.dirname(MODEL_PATH),
    "raman_known_only_domain_profile.npz",
)

# Multi-view inference improves robustness to faint, thin, or low-contrast
# spectral curves without changing the trained model.
VIEW_WEIGHTS = np.array([0.40, 0.25, 0.20, 0.15], dtype=np.float32)

APP_VERSION = "DOMAIN-CALIBRATED 50 v5"

# Single-sample high-resolution PDF export.
PDF_DPI = 1200
PDF_PAGE_WIDTH_MM = 330
AUTO_SAVE_PDF = True
AUTO_SAVE_DELAY_MS = 900

# Strict input-type screening.
# These thresholds are designed to accept thin line-spectrum plots and reject
# filled charts, bar charts, blank images, screenshots, and other unsupported inputs.
MIN_IMAGE_STD = 3.0
MIN_WHITE_PIXEL_RATIO = 0.30
MIN_DARK_PIXEL_RATIO = 0.0002
MAX_DARK_PIXEL_RATIO = 0.50
MIN_EDGE_PIXEL_RATIO = 0.0008

MAX_INK_PIXEL_RATIO = 0.35
MAX_DENSE_COLUMN_RATIO = 0.20
MAX_MEDIAN_INK_COLUMN_FRACTION = 0.15
MIN_EDGE_TO_INK_RATIO = 0.25
MAX_SATURATED_PIXEL_RATIO = 0.08
MAX_LARGEST_SATURATED_COMPONENT_RATIO = 0.08
MAX_LARGE_FILLED_RECTANGLES = 0

# Raman plot-geometry gate.
# A valid input must contain a large x/y-axis plot region and a single
# peak-rich signal concentrated near the lower part of the plotting area.
MIN_PLOT_WIDTH_RATIO = 0.45
MIN_PLOT_HEIGHT_RATIO = 0.35
MIN_SIGNAL_COLUMN_COVERAGE = 0.50
MIN_SIGNAL_MEDIAN_Y_RATIO = 0.70
MIN_SIGNAL_BOTTOM_TOP_RATIO = 1.80
MIN_SIGNAL_DYNAMIC_RANGE = 0.07
MIN_SIGNAL_TURN_COUNT = 5
MAX_SIGNAL_TREND_CORRELATION = 0.75


# ============================================================
# PREPROCESSING
# ============================================================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def read_cv_image(path, flags=cv2.IMREAD_COLOR):
    """Unicode-safe OpenCV image reader for Windows paths."""
    data = np.fromfile(str(path), dtype=np.uint8)

    if data.size == 0:
        return None

    return cv2.imdecode(data, flags)


def write_cv_image(path, image):
    """Unicode-safe OpenCV image writer for Windows paths."""
    extension = os.path.splitext(str(path))[1].lower() or ".png"

    if extension not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        extension = ".png"

    success, encoded = cv2.imencode(extension, image)

    if not success:
        return False

    encoded.tofile(str(path))
    return True


def build_inference_views(img_path):
    """Create conservative views for faint or low-contrast Raman plots."""
    original = Image.open(img_path).convert("RGB")
    gray = ImageOps.grayscale(original)

    auto_gray = ImageOps.autocontrast(
        gray,
        cutoff=1,
    )
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

    return [
        original,
        auto_rgb,
        contrast_rgb,
        clahe_rgb,
    ]


def preprocess_image(img_path):
    """Return the original image and a batch of enhanced inference views."""
    views = build_inference_views(img_path)
    batch = torch.stack(
        [transform(view) for view in views],
        dim=0,
    ).to(DEVICE)

    return views[0], batch, views


@torch.no_grad()
def predict_multiview(model, batch):
    """Weighted ensemble over original and contrast-enhanced views."""
    logits = model(batch)
    view_probabilities = torch.softmax(
        logits,
        dim=1,
    ).cpu().numpy()

    weights = VIEW_WEIGHTS.astype(np.float64)
    weights = weights / weights.sum()

    probabilities = np.average(
        view_probabilities,
        axis=0,
        weights=weights,
    )

    probabilities = probabilities / probabilities.sum()
    predicted_index = int(np.argmax(probabilities))

    agreeing_views = np.flatnonzero(
        np.argmax(view_probabilities, axis=1) == predicted_index
    )

    if len(agreeing_views) > 0:
        best_view_index = int(
            agreeing_views[
                np.argmax(
                    view_probabilities[
                        agreeing_views,
                        predicted_index,
                    ]
                )
            ]
        )
    else:
        best_view_index = int(
            np.argmax(
                view_probabilities[:, predicted_index]
            )
        )

    return probabilities, view_probabilities, best_view_index


@torch.no_grad()
def extract_open_set_view_embeddings(model, batch):
    """Extract normalized penultimate EfficientNet-B3 embeddings."""
    features = model.extract_features(batch)
    pooled = model._avg_pooling(features)
    embeddings = pooled.flatten(start_dim=1)

    embeddings = F.normalize(
        embeddings,
        p=2,
        dim=1,
    )

    return embeddings.cpu().numpy().astype(np.float32)


def load_known_only_profile():
    profile_path = Path(KNOWN_ONLY_PROFILE_PATH)

    if not profile_path.exists():
        raise FileNotFoundError(
            "Known-only domain profile was not found:\n"
            f"{profile_path}\n\n"
            "Run build_raman_known_only_domain_calibrated_profile.py first."
        )

    data = np.load(profile_path, allow_pickle=False)

    profile_classes = [
        str(value)
        for value in data["class_names"].tolist()
    ]

    if profile_classes != CLASSES:
        raise ValueError(
            f"Profile classes: {profile_classes}\n"
            f"App classes: {CLASSES}"
        )

    version = str(data["profile_version"][0])

    if version != "DOMAIN_KNOWN_ONLY_V1":
        raise ValueError(
            f"Wrong profile version: {version}"
        )

    profile = {
        "reference_embeddings": data[
            "reference_embeddings"
        ].astype(np.float32),
        "reference_labels": data[
            "reference_labels"
        ].astype(np.int64),
        "centroids": data[
            "centroids"
        ].astype(np.float32),
        "pca_mean": data["pca_mean"].astype(np.float32),
        "pca_components": data[
            "pca_components"
        ].astype(np.float32),
        "class_pca_means": data[
            "class_pca_means"
        ].astype(np.float32),
        "class_precisions": data[
            "class_precisions"
        ].astype(np.float32),
        "component_medians": data[
            "component_medians"
        ].astype(np.float32),
        "component_scales": data[
            "component_scales"
        ].astype(np.float32),
        "score_thresholds": data[
            "score_thresholds"
        ].astype(np.float32),
        "score_weights": data[
            "score_weights"
        ].astype(np.float32),
        "topk_reference_k": int(
            data["topk_reference_k"][0]
        ),
        "confidence_floor": float(
            data["confidence_floor"][0]
        ),
    }

    print("Domain-calibrated profile loaded:", profile_path)
    print(
        "Reference embeddings:",
        profile["reference_embeddings"].shape,
    )

    return profile


def aggregate_open_set_embedding(
    view_embeddings,
):
    weights = VIEW_WEIGHTS.astype(np.float64)
    weights = weights / weights.sum()

    embedding = np.average(
        view_embeddings,
        axis=0,
        weights=weights,
    )

    embedding /= (
        np.linalg.norm(embedding) + 1e-12
    )

    return embedding.astype(np.float32)


def transform_profile_pca(
    embedding,
    profile,
):
    centered = (
        embedding
        - profile["pca_mean"]
    )

    return (
        centered
        @ profile["pca_components"].T
    ).astype(np.float32)


def assess_known_only_open_set(
    probabilities,
    view_probabilities,
    view_embeddings,
    predicted_index,
    profile,
):
    embedding = aggregate_open_set_embedding(
        view_embeddings
    )

    centroid_similarities = (
        profile["centroids"] @ embedding
    )

    predicted_centroid_similarity = float(
        centroid_similarities[predicted_index]
    )

    sorted_centroids = np.sort(
        centroid_similarities
    )

    centroid_gap = float(
        sorted_centroids[-1]
        - sorted_centroids[-2]
    )

    class_reference = profile[
        "reference_embeddings"
    ][
        profile["reference_labels"]
        == predicted_index
    ]

    reference_similarities = (
        class_reference @ embedding
    )

    k = min(
        profile["topk_reference_k"],
        len(reference_similarities),
    )

    topk_similarity = float(
        np.partition(
            reference_similarities,
            len(reference_similarities) - k,
        )[-k:].mean()
    )

    pca_value = transform_profile_pca(
        embedding,
        profile,
    )

    delta = (
        pca_value
        - profile["class_pca_means"][
            predicted_index
        ]
    )

    precision = profile[
        "class_precisions"
    ][predicted_index]

    mahalanobis = float(
        delta @ precision @ delta
        / max(len(pca_value), 1)
    )

    view_predictions = np.argmax(
        view_probabilities,
        axis=1,
    )

    view_agreement = float(
        np.mean(
            view_predictions
            == predicted_index
        )
    )

    confidence = float(
        probabilities[predicted_index]
    )

    components = np.asarray([
        1.0 - predicted_centroid_similarity,
        1.0 - topk_similarity,
        mahalanobis,
        max(0.0, 0.10 - centroid_gap),
        1.0 - view_agreement,
        1.0 - confidence,
    ], dtype=np.float32)

    median = profile[
        "component_medians"
    ][predicted_index]

    scale = profile[
        "component_scales"
    ][predicted_index]

    z = np.maximum(
        0.0,
        (components - median) / scale,
    )

    score = float(
        z @ profile["score_weights"]
    )

    threshold = float(
        profile["score_thresholds"][
            predicted_index
        ]
    )

    confidence_floor = max(
        MIN_ACCEPT_CONFIDENCE,
        profile["confidence_floor"],
    )

    accepted = bool(
        confidence >= confidence_floor
        and score <= threshold
    )

    return {
        "accepted": accepted,
        "confidence_ok": confidence >= confidence_floor,
        "score_ok": score <= threshold,
        "score": score,
        "score_threshold": threshold,
        "centroid_similarity": predicted_centroid_similarity,
        "topk_reference_similarity": topk_similarity,
        "mahalanobis_distance": mahalanobis,
        "centroid_gap": centroid_gap,
        "view_agreement": int(
            round(view_agreement * len(view_probabilities))
        ),
        "confidence_floor": confidence_floor,
    }


def _merge_vertical_segments(segments, x_tolerance=9, gap_tolerance=28):
    merged = []

    for x, y1, y2 in sorted(segments, key=lambda item: (item[0], item[1])):
        matched = False

        for index, (mx, my1, my2) in enumerate(merged):
            close_x = abs(x - mx) <= x_tolerance
            overlapping_y = y1 <= my2 + gap_tolerance

            if close_x and overlapping_y:
                merged[index] = (
                    int(round((mx + x) / 2)),
                    min(my1, y1),
                    max(my2, y2),
                )
                matched = True
                break

        if not matched:
            merged.append((x, y1, y2))

    return merged


def locate_plot_region(image):
    """Locate a large scientific-plot region using the x and y axes."""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    minimum_length = int(
        max(80, min(height, width) * 0.18)
    )

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(35, int(min(height, width) * 0.06)),
        minLineLength=minimum_length,
        maxLineGap=22,
    )

    if lines is None:
        return None, {
            "horizontal_axis_candidates": 0,
            "vertical_axis_candidates": 0,
        }

    horizontal = []
    vertical = []

    for line in lines[:, 0]:
        x1, y1, x2, y2 = map(int, line)
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if (
            dy <= 4
            and dx >= 0.35 * width
            and min(y1, y2) >= 0.45 * height
        ):
            horizontal.append(
                (
                    min(x1, x2),
                    int(round((y1 + y2) / 2)),
                    max(x1, x2),
                )
            )

        if (
            dx <= 4
            and dy >= 0.22 * height
            and int(round((x1 + x2) / 2)) <= 0.40 * width
        ):
            vertical.append(
                (
                    int(round((x1 + x2) / 2)),
                    min(y1, y2),
                    max(y1, y2),
                )
            )

    vertical = _merge_vertical_segments(vertical)

    candidates = []

    for horizontal_start, horizontal_y, horizontal_end in horizontal:
        for vertical_x, vertical_top, vertical_bottom in vertical:
            aligned_bottom = abs(horizontal_y - vertical_bottom) <= 28
            crosses_vertical = (
                horizontal_start - 24
                <= vertical_x
                <= horizontal_end + 24
            )

            if not aligned_bottom or not crosses_vertical:
                continue

            plot_width = horizontal_end - vertical_x
            plot_height = horizontal_y - vertical_top

            width_ratio = plot_width / max(width, 1)
            height_ratio = plot_height / max(height, 1)

            if (
                width_ratio >= MIN_PLOT_WIDTH_RATIO
                and height_ratio >= MIN_PLOT_HEIGHT_RATIO
            ):
                candidates.append(
                    (
                        plot_width * plot_height,
                        vertical_x,
                        vertical_top,
                        horizontal_end,
                        horizontal_y,
                        width_ratio,
                        height_ratio,
                    )
                )

    metrics = {
        "horizontal_axis_candidates": len(horizontal),
        "vertical_axis_candidates": len(vertical),
    }

    if not candidates:
        return None, metrics

    best = max(candidates, key=lambda item: item[0])

    _, left, top, right, bottom, width_ratio, height_ratio = best

    metrics.update({
        "plot_width_ratio": float(width_ratio),
        "plot_height_ratio": float(height_ratio),
    })

    return (left, top, right, bottom), metrics


def measure_spectral_curve_geometry(image, plot_region):
    """Measure whether the plot contains a peak-rich Raman-like line."""
    left, top, right, bottom = plot_region
    plot = image[top:bottom, left:right]

    gray = cv2.cvtColor(plot, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(plot, cv2.COLOR_BGR2HSV)

    plot_height, plot_width = gray.shape[:2]

    foreground = (
        (gray <= 220)
        | (hsv[:, :, 1] >= 45)
    ).astype(np.uint8)

    border = max(
        3,
        int(round(min(plot_height, plot_width) * 0.012)),
    )

    foreground[:border, :] = 0
    foreground[-border:, :] = 0
    foreground[:, :border] = 0
    foreground[:, -border:] = 0

    # Remove long grid lines and remaining axis fragments.
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(20, plot_width // 3), 1),
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(20, plot_height // 3)),
    )

    horizontal_lines = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )
    vertical_lines = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        vertical_kernel,
    )

    signal_mask = foreground.copy()
    signal_mask[horizontal_lines > 0] = 0
    signal_mask[vertical_lines > 0] = 0

    row_density = signal_mask.mean(axis=1)

    upper_mass = float(
        row_density[:max(1, int(0.40 * plot_height))].sum()
    )
    lower_mass = float(
        row_density[int(0.60 * plot_height):].sum()
    )

    bottom_top_ratio = lower_mass / max(upper_mass, 1e-9)

    curve_y = np.full(plot_width, np.nan, dtype=np.float64)

    for column in range(plot_width):
        rows = np.flatnonzero(signal_mask[:, column])

        if rows.size:
            # The 75th percentile follows the lower baseline while retaining
            # upward Raman peaks and is less sensitive to text or isolated dots.
            curve_y[column] = float(np.percentile(rows, 75))

    valid_columns = np.isfinite(curve_y)
    coverage = float(valid_columns.mean())

    metrics = {
        "signal_column_coverage": coverage,
        "signal_bottom_top_ratio": float(bottom_top_ratio),
    }

    if valid_columns.sum() < max(10, int(0.20 * plot_width)):
        metrics.update({
            "signal_median_y_ratio": 0.0,
            "signal_dynamic_range": 0.0,
            "signal_turn_count": 0,
            "signal_trend_correlation": 1.0,
        })
        return metrics

    valid_x = np.flatnonzero(valid_columns)

    curve_y = np.interp(
        np.arange(plot_width),
        valid_x,
        curve_y[valid_columns],
    )

    median_y_ratio = float(
        np.median(curve_y) / max(plot_height, 1)
    )

    dynamic_range = float(
        (
            np.quantile(curve_y, 0.90)
            - np.quantile(curve_y, 0.10)
        )
        / max(plot_height, 1)
    )

    smoothing_width = max(5, plot_width // 55)

    if smoothing_width % 2 == 0:
        smoothing_width += 1

    kernel = np.ones(
        smoothing_width,
        dtype=np.float64,
    ) / smoothing_width

    smooth_curve = np.convolve(
        curve_y,
        kernel,
        mode="same",
    )

    derivative = np.diff(smooth_curve)
    derivative_threshold = max(
        0.45,
        0.004 * plot_height,
    )

    derivative_sign = np.sign(
        np.where(
            np.abs(derivative) >= derivative_threshold,
            derivative,
            0.0,
        )
    )

    nonzero_sign = derivative_sign[
        derivative_sign != 0
    ]

    if nonzero_sign.size >= 2:
        turn_count = int(
            np.sum(
                nonzero_sign[1:]
                != nonzero_sign[:-1]
            )
        )
    else:
        turn_count = 0

    if np.std(smooth_curve) > 1e-9:
        trend_correlation = float(
            np.corrcoef(
                np.arange(plot_width),
                smooth_curve,
            )[0, 1]
        )
    else:
        trend_correlation = 1.0

    metrics.update({
        "signal_median_y_ratio": median_y_ratio,
        "signal_dynamic_range": dynamic_range,
        "signal_turn_count": turn_count,
        "signal_trend_correlation": trend_correlation,
    })

    return metrics


def validate_spectrum_image(img_path):
    """Reject screenshots and plots that are not Raman line spectra."""
    image = read_cv_image(img_path)

    if image is None:
        return False, {}, [
            "The selected file cannot be read as an image."
        ]

    height, width = image.shape[:2]

    if min(height, width) < 160:
        return False, {}, [
            "Image resolution is too small."
        ]

    y0 = int(round(0.04 * height))
    y1 = int(round(0.96 * height))
    x0 = int(round(0.04 * width))
    x1 = int(round(0.96 * width))

    roi = image[y0:y1, x0:x1]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    roi_height, roi_width = gray.shape[:2]

    image_std = float(gray.std())
    white_ratio = float(np.mean(gray >= 235))
    dark_ratio = float(np.mean(gray <= 80))

    edge_gray = cv2.createCLAHE(
        clipLimit=1.6,
        tileGridSize=(8, 8),
    ).apply(gray)

    edge_mask = cv2.Canny(
        edge_gray,
        30,
        100,
    ) > 0

    edge_ratio = float(np.mean(edge_mask))

    ink_mask = gray <= 220
    ink_ratio = float(np.mean(ink_mask))

    edge_to_ink_ratio = float(
        edge_ratio / max(ink_ratio, 1e-8)
    )

    ink_per_column = (
        ink_mask.sum(axis=0)
        / max(roi_height, 1)
    )

    dense_column_ratio = float(
        np.mean(ink_per_column >= 0.25)
    )

    median_ink_column_fraction = float(
        np.median(ink_per_column)
    )

    saturation_mask = hsv[:, :, 1] >= 50

    saturated_pixel_ratio = float(
        np.mean(saturation_mask)
    )

    component_count, _, component_stats, _ = (
        cv2.connectedComponentsWithStats(
            saturation_mask.astype(np.uint8),
            connectivity=8,
        )
    )

    largest_saturated_component_ratio = 0.0
    largest_saturated_component_extent = 0.0

    if component_count > 1:
        component_areas = component_stats[
            1:,
            cv2.CC_STAT_AREA,
        ]

        largest_index = int(
            np.argmax(component_areas)
        ) + 1

        largest_area = float(
            component_stats[
                largest_index,
                cv2.CC_STAT_AREA,
            ]
        )

        largest_width = float(
            component_stats[
                largest_index,
                cv2.CC_STAT_WIDTH,
            ]
        )

        largest_height = float(
            component_stats[
                largest_index,
                cv2.CC_STAT_HEIGHT,
            ]
        )

        largest_saturated_component_ratio = (
            largest_area
            / (roi_height * roi_width)
        )

        largest_saturated_component_extent = (
            largest_area
            / max(
                largest_width * largest_height,
                1.0,
            )
        )

    binary_ink = np.uint8(ink_mask) * 255

    binary_ink = cv2.morphologyEx(
        binary_ink,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 3),
        ),
        iterations=1,
    )

    contours, _ = cv2.findContours(
        binary_ink,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    large_filled_rectangles = 0

    for contour in contours:
        area = float(cv2.contourArea(contour))

        if area <= 0:
            continue

        contour_ratio = (
            area / (roi_height * roi_width)
        )

        if contour_ratio < 0.008:
            continue

        _, _, box_width, box_height = (
            cv2.boundingRect(contour)
        )

        bounding_area = float(
            max(box_width * box_height, 1)
        )

        hull = cv2.convexHull(contour)
        hull_area = float(
            max(cv2.contourArea(hull), 1.0)
        )

        solidity = area / hull_area
        extent = area / bounding_area

        if (
            solidity >= 0.75
            and extent >= 0.65
        ):
            large_filled_rectangles += 1

    plot_region, plot_metrics = locate_plot_region(
        image
    )

    if plot_region is None:
        signal_metrics = {
            "signal_column_coverage": 0.0,
            "signal_bottom_top_ratio": 0.0,
            "signal_median_y_ratio": 0.0,
            "signal_dynamic_range": 0.0,
            "signal_turn_count": 0,
            "signal_trend_correlation": 1.0,
        }
    else:
        signal_metrics = measure_spectral_curve_geometry(
            image,
            plot_region,
        )

    reasons = []

    if image_std < MIN_IMAGE_STD:
        reasons.append(
            "The image has insufficient contrast."
        )

    if white_ratio < MIN_WHITE_PIXEL_RATIO:
        reasons.append(
            "The image background does not resemble a spectrum plot."
        )

    if not (
        MIN_DARK_PIXEL_RATIO
        <= dark_ratio
        <= MAX_DARK_PIXEL_RATIO
    ):
        reasons.append(
            "The amount of line content is outside the expected range."
        )

    if edge_ratio < MIN_EDGE_PIXEL_RATIO:
        reasons.append(
            "No clear spectral curve or axes were detected."
        )

    if ink_ratio > MAX_INK_PIXEL_RATIO:
        reasons.append(
            "The image contains too much filled graphical content."
        )

    if dense_column_ratio > MAX_DENSE_COLUMN_RATIO:
        reasons.append(
            "Large filled columns were detected."
        )

    if (
        median_ink_column_fraction
        > MAX_MEDIAN_INK_COLUMN_FRACTION
    ):
        reasons.append(
            "The graphical content is too dense for a thin spectrum line."
        )

    if edge_to_ink_ratio < MIN_EDGE_TO_INK_RATIO:
        reasons.append(
            "The image contains filled regions rather than a thin spectrum line."
        )

    if saturated_pixel_ratio > MAX_SATURATED_PIXEL_RATIO:
        reasons.append(
            "A large colored area was detected."
        )

    if (
        largest_saturated_component_ratio
        > MAX_LARGEST_SATURATED_COMPONENT_RATIO
        and largest_saturated_component_extent > 0.35
    ):
        reasons.append(
            "A large filled colored region was detected."
        )

    if (
        large_filled_rectangles
        > MAX_LARGE_FILLED_RECTANGLES
    ):
        reasons.append(
            "Filled rectangular bars were detected."
        )

    if plot_region is None:
        reasons.append(
            "A valid Raman-style x/y plotting region was not detected."
        )
    else:
        if (
            signal_metrics["signal_column_coverage"]
            < MIN_SIGNAL_COLUMN_COVERAGE
        ):
            reasons.append(
                "No continuous spectrum line was detected across the plot."
            )

        if (
            signal_metrics["signal_median_y_ratio"]
            < MIN_SIGNAL_MEDIAN_Y_RATIO
        ):
            reasons.append(
                "The plotted curves do not have a Raman-spectrum baseline pattern."
            )

        if (
            signal_metrics["signal_bottom_top_ratio"]
            < MIN_SIGNAL_BOTTOM_TOP_RATIO
        ):
            reasons.append(
                "The plot is not dominated by a lower spectral baseline."
            )

        if (
            signal_metrics["signal_dynamic_range"]
            < MIN_SIGNAL_DYNAMIC_RANGE
        ):
            reasons.append(
                "The detected signal does not contain sufficient peak variation."
            )

        if (
            signal_metrics["signal_turn_count"]
            < MIN_SIGNAL_TURN_COUNT
        ):
            reasons.append(
                "The detected curve is not sufficiently peak-rich."
            )

        if (
            signal_metrics["signal_trend_correlation"]
            > MAX_SIGNAL_TREND_CORRELATION
        ):
            reasons.append(
                "A monotonic performance curve was detected instead of a Raman spectrum."
            )

    metrics = {
        "image_std": image_std,
        "white_ratio": white_ratio,
        "dark_ratio": dark_ratio,
        "edge_ratio": edge_ratio,
        "ink_ratio": ink_ratio,
        "edge_to_ink_ratio": edge_to_ink_ratio,
        "dense_column_ratio": dense_column_ratio,
        "median_ink_column_fraction": (
            median_ink_column_fraction
        ),
        "saturated_pixel_ratio": (
            saturated_pixel_ratio
        ),
        "largest_saturated_component_ratio": (
            largest_saturated_component_ratio
        ),
        "largest_saturated_component_extent": (
            largest_saturated_component_extent
        ),
        "large_filled_rectangles": (
            large_filled_rectangles
        ),
        **plot_metrics,
        **signal_metrics,
    }

    print("Input screening metrics:", metrics)

    return len(reasons) == 0, metrics, reasons


def assess_prediction(probabilities):
    """Accept valid spectrum inputs when top-1 confidence is at least 45%."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()

    ranked = np.argsort(probabilities)[::-1]
    top_index = int(ranked[0])
    second_index = int(ranked[1])

    confidence = float(probabilities[top_index])
    second_confidence = float(probabilities[second_index])
    margin = confidence - second_confidence

    safe_probabilities = np.clip(
        probabilities,
        1e-12,
        1.0,
    )
    entropy = float(
        -np.sum(
            safe_probabilities
            * np.log(safe_probabilities)
        )
        / np.log(len(safe_probabilities))
    )

    return {
        "accepted": bool(
            confidence >= MIN_ACCEPT_CONFIDENCE
        ),
        "top_index": top_index,
        "second_index": second_index,
        "confidence": confidence,
        "second_confidence": second_confidence,
        "margin": margin,
        "entropy": entropy,
    }


# ============================================================
# MODEL
# ============================================================
def load_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found:\n{MODEL_PATH}")

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
    model = model.to(DEVICE)
    model.eval()

    return model


model = load_model()
known_only_profile = load_known_only_profile()

print("Model loaded successfully.")
print("App version:", APP_VERSION)
print("Confidence floor:", f"{MIN_ACCEPT_CONFIDENCE * 100:.0f}%")
print("Device:", DEVICE)
print("Classes:", CLASSES)


# ============================================================
# IMAGE UTILITIES
# ============================================================
def sharpen_bgr(image, sigma=0.9, strength=0.55):
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharpened = cv2.addWeighted(
        image,
        1.0 + strength,
        blurred,
        -strength,
        0,
    )
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def save_display_copy(img_path):
    image = read_cv_image(img_path)

    if image is None:
        raise ValueError("Cannot read the selected Raman image.")

    sharpened = sharpen_bgr(image)
    out_path = os.path.join(
        tempfile.gettempdir(),
        "raman_input_display.png",
    )

    if not write_cv_image(out_path, sharpened):
        raise RuntimeError("Could not create the display image.")

    return out_path


# ============================================================
# GRAD-CAM
# ============================================================
def compute_gradcam(model, tensor, class_idx):
    activations = None
    gradients = None

    def save_activations(module, inputs, output):
        nonlocal activations
        activations = output.detach().clone()

    def save_gradients(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach().clone()

    forward_handle = model._conv_head.register_forward_hook(
        save_activations
    )
    backward_handle = model._conv_head.register_full_backward_hook(
        save_gradients
    )

    try:
        model.zero_grad(set_to_none=True)
        output = model(tensor)
        output[0, class_idx].backward()

        if activations is None or gradients is None:
            raise RuntimeError("Could not capture Grad-CAM features.")

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = (weights * activations).sum(dim=1).squeeze(0)
        heatmap = torch.relu(heatmap).cpu().numpy()

        maximum = float(heatmap.max())
        if maximum > 0:
            heatmap /= maximum

        return heatmap

    finally:
        forward_handle.remove()
        backward_handle.remove()


def overlay_gradcam_on_image(img_path, heatmap, alpha=0.32):
    image = read_cv_image(img_path)

    if image is None:
        raise ValueError("Cannot read the selected Raman image.")

    resized_heatmap = cv2.resize(
        heatmap,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )

    colored_heatmap = cv2.applyColorMap(
        np.uint8(np.clip(resized_heatmap, 0, 1) * 255),
        cv2.COLORMAP_JET,
    )

    overlay = cv2.addWeighted(
        image,
        1.0 - alpha,
        colored_heatmap,
        alpha,
        0,
    )

    return sharpen_bgr(overlay, strength=0.40)



def add_rejected_gradcam_banner(image, predicted_class, confidence):
    """Mark Grad-CAM as explanatory only for a rejected candidate."""
    output = image.copy()
    height, width = output.shape[:2]

    banner_height = max(34, int(round(height * 0.11)))

    cv2.rectangle(
        output,
        (0, 0),
        (width, banner_height),
        (18, 52, 140),
        thickness=-1,
    )

    main_text = (
        f"REJECTED CANDIDATE: {predicted_class} "
        f"({confidence:.1f}%)"
    )

    secondary_text = "Attention map only - not a valid identification"

    main_scale = max(0.45, min(0.82, width / 900))
    secondary_scale = max(0.34, min(0.58, width / 1200))

    cv2.putText(
        output,
        main_text,
        (12, max(20, int(banner_height * 0.45))),
        cv2.FONT_HERSHEY_SIMPLEX,
        main_scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        secondary_text,
        (12, max(31, int(banner_height * 0.82))),
        cv2.FONT_HERSHEY_SIMPLEX,
        secondary_scale,
        (235, 240, 255),
        1,
        cv2.LINE_AA,
    )

    return output


# ============================================================
# UI HELPERS — SINGLE-SPECTRUM PAPER MODE
# ============================================================
UI_FONT_SCALE = 3.0
EXPORT_SCALE = 4


def fs(base):
    """Large Arial UI font for paper-ready single-sample display."""
    return int(round(base * UI_FONT_SCALE))


def add_shadow(widget):
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(12)
    shadow.setColor(QColor(23, 43, 77, 18))
    shadow.setOffset(0, 2)
    widget.setGraphicsEffect(shadow)


def content_crop_box(image, threshold=248, pad_ratio=0.025):
    """
    Find the tight visible-content box of the original spectrum image.
    Used only for DISPLAY/EXPORT, never for model inference.
    """
    if image is None or image.size == 0:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = (gray < threshold) | (hsv[:, :, 1] > 22)
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    h, w = image.shape[:2]
    pad_x = max(8, int(round((x1 - x0 + 1) * pad_ratio)))
    pad_y = max(8, int(round((y1 - y0 + 1) * pad_ratio)))

    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w, x1 + pad_x + 1)
    y1 = min(h, y1 + pad_y + 1)

    if x1 <= x0 or y1 <= y0:
        return None

    return (x0, y0, x1, y1)


def crop_with_box(image, box):
    if box is None:
        return image
    x0, y0, x1, y1 = box
    return image[y0:y1, x0:x1]


class LargeImagePanel(QFrame):
    """Large scientific panel for one sample."""

    def __init__(self, title, placeholder):
        super().__init__()
        self.setObjectName("scienceCard")
        self._pixmap = None

        self.title_label = QLabel(title)
        self.title_label.setObjectName("scienceTitle")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.image_label = QLabel(placeholder)
        self.image_label.setObjectName("imageCanvas")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(360, 420)
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 7, 8, 8)
        layout.setSpacing(4)
        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label, 1)

    def _refresh_pixmap(self):
        if self._pixmap is None or self._pixmap.isNull():
            return

        self.image_label.setPixmap(
            self._pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def set_image(self, image_path):
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            raise ValueError(f"Cannot display image:\n{image_path}")

        self._pixmap = pixmap
        self.image_label.setText("")
        self._refresh_pixmap()

    def reset(self, placeholder):
        self._pixmap = None
        self.image_label.clear()
        self.image_label.setText(placeholder)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()


# ============================================================
# MAIN APPLICATION — ONE SAMPLE PER TEST
# ============================================================
class RamanApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.current_image_path = None
        self.current_result = None

        self.setWindowTitle("Raman Antibiotic Analyzer")

        screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.resize(
                int(available.width() * 0.995),
                int(available.height() * 0.965),
            )
        else:
            self.resize(1900, 980)

        self.setMinimumSize(1280, 720)

        # ----------------------------------------------------
        # PAPER-READY ARIAL STYLE
        # ----------------------------------------------------
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: #F4F7FB;
            }}

            QWidget {{
                font-family: Arial;
                color: #17233C;
            }}

            QFrame#resultHeader,
            QFrame#scienceCard {{
                background-color: #FFFFFF;
                border: 1px solid #DDE5EF;
                border-radius: 10px;
            }}

            QLabel#predictionClass {{
                color: #0D6EFD;
                font-family: Arial;
                font-size: {fs(28)}px;
                font-weight: 900;
            }}

            QLabel#predictionConfidence {{
                color: #162A46;
                font-family: Arial;
                font-size: {fs(25)}px;
                font-weight: 900;
            }}

            QLabel#statusReady,
            QLabel#statusRunning,
            QLabel#statusAccepted,
            QLabel#statusRejected {{
                border-radius: 10px;
                padding: 5px 12px;
                font-family: Arial;
                font-size: {fs(10)}px;
                font-weight: 900;
            }}

            QLabel#statusReady {{
                color: #52647E;
                background-color: #EEF2F7;
            }}

            QLabel#statusRunning {{
                color: #735B00;
                background-color: #FFF4CC;
            }}

            QLabel#statusAccepted {{
                color: #0F6F4B;
                background-color: #E7F8F0;
            }}

            QLabel#statusRejected {{
                color: #9A5B00;
                background-color: #FFF0D8;
            }}

            QLabel#scienceTitle {{
                color: #172844;
                font-family: Arial;
                font-size: {fs(14)}px;
                font-weight: 900;
            }}

            QLabel#imageCanvas {{
                background-color: #FFFFFF;
                border: 1px solid #D8E2EE;
                border-radius: 8px;
                color: #8896AA;
                font-family: Arial;
                font-size: {fs(11)}px;
                font-weight: 800;
                padding: 0px;
            }}

            QPushButton#primaryButton {{
                background-color: #0B78D0;
                color: white;
                border: none;
                border-radius: 10px;
                min-height: 52px;
                padding: 0 20px;
                font-family: Arial;
                font-size: {fs(11)}px;
                font-weight: 900;
            }}

            QPushButton#primaryButton:hover {{
                background-color: #0967B3;
            }}

            QPushButton#secondaryButton {{
                background-color: #FFFFFF;
                color: #43516A;
                border: 1px solid #CED7E5;
                border-radius: 10px;
                min-height: 52px;
                padding: 0 18px;
                font-family: Arial;
                font-size: {fs(10)}px;
                font-weight: 900;
            }}

            QPushButton#secondaryButton:hover {{
                background-color: #F5F8FC;
            }}
        """)

        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # Exported area: compact result header + 3 scientific panels.
        self.figure_widget = QWidget()
        figure_layout = QVBoxLayout(self.figure_widget)
        figure_layout.setContentsMargins(0, 0, 0, 0)
        figure_layout.setSpacing(6)

        figure_layout.addWidget(self.build_result_header())

        panels = QHBoxLayout()
        panels.setContentsMargins(0, 0, 0, 0)
        panels.setSpacing(6)

        self.input_panel = LargeImagePanel(
            "Input spectrum",
            "Select a Raman image",
        )
        self.gradcam_panel = LargeImagePanel(
            "Model attention",
            "Grad-CAM",
        )
        self.chart_panel = LargeImagePanel(
            "Prediction confidence",
            "Confidence",
        )

        panels.addWidget(self.input_panel, 1)
        panels.addWidget(self.gradcam_panel, 1)
        panels.addWidget(self.chart_panel, 1)

        figure_layout.addLayout(panels, 1)

        root.addWidget(self.figure_widget, 1)
        root.addLayout(self.build_action_bar())

    # ========================================================
    # RESULT HEADER — ONLY ESSENTIAL INFORMATION
    # ========================================================
    def build_result_header(self):
        card = QFrame()
        card.setObjectName("resultHeader")
        add_shadow(card)

        self.prediction_class = QLabel("Waiting")
        self.prediction_class.setObjectName("predictionClass")

        self.prediction_confidence = QLabel("")
        self.prediction_confidence.setObjectName("predictionConfidence")

        self.status_label = QLabel("READY")
        self.status_label.setObjectName("statusReady")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QHBoxLayout(card)
        layout.setContentsMargins(14, 7, 14, 7)
        layout.setSpacing(12)
        layout.addWidget(self.prediction_class)
        layout.addWidget(self.prediction_confidence)
        layout.addStretch()
        layout.addWidget(self.status_label)

        return card

    # ========================================================
    # ACTIONS
    # ========================================================
    def build_action_bar(self):
        save_button = QPushButton("Save PDF")
        save_button.setObjectName("secondaryButton")
        save_button.clicked.connect(self.save_pdf_dialog)

        reset_button = QPushButton("Reset")
        reset_button.setObjectName("secondaryButton")
        reset_button.clicked.connect(self.reset_view)

        select_button = QPushButton("Select Raman image")
        select_button.setObjectName("primaryButton")
        select_button.clicked.connect(self.open_image)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addStretch()
        layout.addWidget(save_button)
        layout.addWidget(reset_button)
        layout.addWidget(select_button)
        return layout

    # ========================================================
    # SINGLE IMAGE SELECTION
    # ========================================================
    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Raman spectrum image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )

        if not file_path:
            return

        self.current_image_path = file_path
        self.current_result = None

        self.set_running()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:
            QApplication.processEvents()
            self.current_result = self.run_inference(file_path)

            if self.current_result and self.current_result.get("accepted"):
                self.schedule_auto_save()

        except Exception as error:
            self.set_rejected("ERROR")
            QMessageBox.critical(self, "Analysis error", str(error))

        finally:
            QApplication.restoreOverrideCursor()

    # ========================================================
    # STATUS
    # ========================================================
    def _set_status_object(self, object_name):
        self.status_label.setObjectName(object_name)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def set_running(self):
        self.prediction_class.setText("Analyzing…")
        self.prediction_confidence.setText("")
        self.status_label.setText("RUNNING")
        self._set_status_object("statusRunning")
        QApplication.processEvents()

    def set_accepted(self, predicted_class, confidence):
        self.prediction_class.setText(predicted_class)
        self.prediction_confidence.setText(f"{confidence:.1f}%")
        self.status_label.setText("ACCEPTED")
        self._set_status_object("statusAccepted")

    def set_rejected(self, label="REJECTED"):
        self.prediction_class.setText(label)
        self.prediction_confidence.setText("")
        self.status_label.setText("REJECTED")
        self._set_status_object("statusRejected")

    # ========================================================
    # SINGLE-SAMPLE INFERENCE
    # ========================================================
    def run_inference(self, img_path):
        image = read_cv_image(img_path)
        if image is None:
            self.set_rejected("INVALID INPUT")
            return {
                "accepted": False,
                "reason": "Unreadable image",
            }

        # Tight display crop only. Model still receives the untouched image.
        crop_box = content_crop_box(image)

        display_image = crop_with_box(
            sharpen_bgr(image),
            crop_box,
        )

        display_path = os.path.join(
            tempfile.gettempdir(),
            "raman_single_input_display.png",
        )

        if not write_cv_image(display_path, display_image):
            raise RuntimeError("Could not create the display image.")

        self.input_panel.set_image(display_path)

        # ----------------------------------------------------
        # INPUT SCREENING
        # ----------------------------------------------------
        is_valid_image, _, image_reasons = validate_spectrum_image(img_path)

        if not is_valid_image:
            self.gradcam_panel.reset("Unsupported")
            self.chart_panel.reset("Unsupported")
            self.set_rejected("UNSUPPORTED")

            return {
                "accepted": False,
                "reason": " ".join(image_reasons),
            }

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------
        _, inference_batch, _ = preprocess_image(img_path)

        probabilities, view_probabilities, best_view_index = predict_multiview(
            model,
            inference_batch,
        )

        decision = assess_prediction(probabilities)
        predicted_index = decision["top_index"]
        predicted_class = CLASSES[predicted_index]
        confidence = decision["confidence"] * 100.0

        # ----------------------------------------------------
        # KNOWN-ONLY OPEN-SET DECISION
        # ----------------------------------------------------
        view_embeddings = extract_open_set_view_embeddings(
            model,
            inference_batch,
        )

        open_set_decision = assess_known_only_open_set(
            probabilities,
            view_probabilities,
            view_embeddings,
            predicted_index,
            known_only_profile,
        )

        # ----------------------------------------------------
        # CREATE CONFIDENCE CHART EVEN FOR A VALID CANDIDATE
        # ----------------------------------------------------
        chart_path = self.create_confidence_chart(probabilities)
        self.chart_panel.set_image(chart_path)

        # ----------------------------------------------------
        # GRAD-CAM
        # ----------------------------------------------------
        gradcam_tensor = inference_batch[best_view_index].unsqueeze(0)

        heatmap = compute_gradcam(
            model,
            gradcam_tensor,
            predicted_index,
        )

        gradcam = overlay_gradcam_on_image(
            img_path,
            heatmap,
        )

        # Same tight crop as the input display.
        gradcam = crop_with_box(
            gradcam,
            crop_box,
        )

        gradcam_path = os.path.join(
            tempfile.gettempdir(),
            f"raman_single_gradcam_{predicted_class}.png",
        )

        if not write_cv_image(gradcam_path, gradcam):
            raise RuntimeError("Could not save Grad-CAM.")

        self.gradcam_panel.set_image(gradcam_path)

        # ----------------------------------------------------
        # FINAL DECISION
        # ----------------------------------------------------
        if not open_set_decision["accepted"]:
            self.set_rejected("UNKNOWN")

            return {
                "accepted": False,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "decision": open_set_decision,
            }

        self.set_accepted(
            predicted_class,
            confidence,
        )

        return {
            "accepted": True,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "decision": open_set_decision,
        }

    # ========================================================
    # CONFIDENCE CHART — EXTRA LARGE ARIAL TEXT / NUMBERS
    # ========================================================
    def create_confidence_chart(self, probabilities):
        chart_path = os.path.join(
            tempfile.gettempdir(),
            "raman_single_confidence.png",
        )

        with plt.rc_context({
            "font.family": "Arial",
            "font.size": 32,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }):
            fig, ax = plt.subplots(
                figsize=(9.2, 7.0),
                dpi=520,
            )

            y_positions = np.arange(len(CLASSES))

            bars = ax.barh(
                y_positions,
                probabilities * 100,
                height=0.70,
                color="#1AA6C9",
                edgecolor="#087EA4",
                linewidth=1.6,
            )

            ax.set_yticks(y_positions)
            ax.set_yticklabels(
                CLASSES,
                fontsize=42,
                fontweight="bold",
                fontfamily="Arial",
            )

            ax.invert_yaxis()
            ax.set_xlim(0, 108)
            ax.set_xticks([0, 20, 40, 60, 80, 100])

            ax.set_xlabel(
                "Confidence (%)",
                fontsize=42,
                fontweight="bold",
                fontfamily="Arial",
                labelpad=10,
            )

            ax.bar_label(
                bars,
                labels=[f"{p * 100:.1f}%" for p in probabilities],
                padding=9,
                fontsize=38,
                fontweight="bold",
                fontfamily="Arial",
            )

            ax.tick_params(
                axis="x",
                labelsize=31,
                width=1.4,
                length=7,
            )

            ax.tick_params(
                axis="y",
                labelsize=42,
                width=1.4,
                length=4,
            )

            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontfamily("Arial")

            ax.grid(
                axis="x",
                linestyle="--",
                linewidth=0.9,
                alpha=0.16,
            )

            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#9AA6B5")
            ax.spines["bottom"].set_color("#9AA6B5")
            ax.spines["left"].set_linewidth(1.3)
            ax.spines["bottom"].set_linewidth(1.3)

            # Use almost all of the canvas. No unnecessary white margin.
            fig.subplots_adjust(
                left=0.18,
                right=0.94,
                top=0.97,
                bottom=0.16,
            )

            fig.savefig(
                chart_path,
                dpi=720,
                bbox_inches="tight",
                pad_inches=0.015,
                facecolor="white",
            )

            plt.close(fig)

        return chart_path

    # ========================================================
    # PDF EXPORT — MAXIMUM PRACTICAL QUALITY
    # ========================================================
    def _export_directory(self):
        if self.current_image_path:
            base_directory = Path(self.current_image_path).parent
        else:
            base_directory = Path(MODEL_PATH).parent

        output_directory = base_directory / "interface_exports"
        output_directory.mkdir(parents=True, exist_ok=True)
        return output_directory

    def _build_filename(self):
        if self.current_result and self.current_result.get("accepted"):
            class_name = self.current_result.get("predicted_class", "UNK")
            confidence = self.current_result.get("confidence", 0.0)
            tag = f"{class_name}_{confidence:.1f}pct"
        else:
            tag = "result"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"raman_{tag}_{timestamp}.pdf"

    def _render_figure_to_highres_image(self, scale=EXPORT_SCALE):
        scale = max(1, int(scale))

        QApplication.processEvents()
        self.figure_widget.repaint()
        self.input_panel.repaint()
        self.gradcam_panel.repaint()
        self.chart_panel.repaint()
        QApplication.processEvents()

        logical_width = self.figure_widget.width()
        logical_height = self.figure_widget.height()

        if logical_width <= 0 or logical_height <= 0:
            raise RuntimeError("Invalid figure size.")

        pixel_width = logical_width * scale
        pixel_height = logical_height * scale

        image = QImage(
            pixel_width,
            pixel_height,
            QImage.Format.Format_ARGB32_Premultiplied,
        )

        image.setDevicePixelRatio(float(scale))
        image.fill(QColor("#F4F7FB"))

        painter = QPainter(image)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            self.figure_widget.render(painter)
        finally:
            painter.end()

        image.setDevicePixelRatio(1.0)
        return image

    def export_pdf(self, save_path):
        save_path = Path(save_path)
        if save_path.suffix.lower() != ".pdf":
            save_path = save_path.with_suffix(".pdf")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        image = self._render_figure_to_highres_image(
            scale=EXPORT_SCALE
        )

        image_width = image.width()
        image_height = image.height()

        if image_width <= 0 or image_height <= 0:
            raise RuntimeError("Could not render the scientific figure.")

        pdf_width_mm = float(PDF_PAGE_WIDTH_MM)
        pdf_height_mm = pdf_width_mm * image_height / image_width

        writer = QPdfWriter(str(save_path))
        writer.setResolution(PDF_DPI)

        writer.setPageSize(
            QPageSize(
                QSizeF(pdf_width_mm, pdf_height_mm),
                QPageSize.Unit.Millimeter,
                "Raman single-sample result",
            )
        )

        writer.setPageMargins(
            QMarginsF(0, 0, 0, 0),
            QPageLayout.Unit.Millimeter,
        )

        painter = QPainter(writer)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

            page_rect = writer.pageLayout().paintRectPixels(
                writer.resolution()
            )

            painter.drawImage(
                page_rect,
                image,
            )

        finally:
            painter.end()

        print("=" * 80)
        print("SINGLE-SAMPLE PDF SAVED")
        print("=" * 80)
        print("Render scale   :", f"{EXPORT_SCALE}x")
        print("Rendered image :", f"{image_width} x {image_height} px")
        print("PDF resolution :", f"{PDF_DPI} dpi")
        print("PDF page       :", f"{pdf_width_mm:.1f} x {pdf_height_mm:.1f} mm")
        print("Path           :", save_path)
        print("=" * 80)

        return save_path

    def save_pdf_dialog(self):
        output_directory = self._export_directory()
        default_path = output_directory / self._build_filename()

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Raman result",
            str(default_path),
            "PDF document (*.pdf)",
        )

        if not save_path:
            return

        try:
            saved_path = self.export_pdf(save_path)
            QMessageBox.information(
                self,
                "PDF saved",
                f"Saved successfully.\n\n{saved_path}",
            )
        except Exception as error:
            QMessageBox.critical(self, "Save error", str(error))

    def auto_save_pdf(self):
        if not AUTO_SAVE_PDF or not self.current_image_path:
            return

        try:
            output_directory = self._export_directory()
            save_path = output_directory / self._build_filename()
            self.export_pdf(save_path)
        except Exception as error:
            print("Automatic PDF save failed:", error)

    def schedule_auto_save(self):
        if AUTO_SAVE_PDF:
            QTimer.singleShot(
                AUTO_SAVE_DELAY_MS,
                self.auto_save_pdf,
            )

    # ========================================================
    # RESET
    # ========================================================
    def reset_view(self):
        self.current_image_path = None
        self.current_result = None

        self.prediction_class.setText("Waiting")
        self.prediction_confidence.setText("")
        self.status_label.setText("READY")
        self._set_status_object("statusReady")

        self.input_panel.reset("Select a Raman image")
        self.gradcam_panel.reset("Grad-CAM")
        self.chart_panel.reset("Confidence")


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Arial"))

    window = RamanApp()
    window.showMaximized()

    sys.exit(app.exec())

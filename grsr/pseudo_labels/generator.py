#!/usr/bin/env python3
"""Generate the physics-inspired initial pseudo masks of GRSR.

The implementation follows Sec. III-A of the manuscript:
  1. crop a point-centred target region;
  2. suppress the background with a morphological top-hat transform;
  3. threshold at 0.5, retain the component containing the point, and skeletonize;
  4. generate skeleton-distance Gaussian responses;
  5. select ``(r, sigma)`` by minimizing ``E_fit-lambda_c*E_ctr+lambda_r*E_reg``.

Parameters not fixed by the manuscript are exposed as CLI options and labelled
as optimization space. They are implementation choices, not extra modules.
"""

import argparse
import json
import os
from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


DEFAULT_OUTPUT_SUBDIR = ("supervision_maps_grsr", "initial")


def resolve_layout(data_root: str) -> Tuple[str, str]:
    """Resolve the two dataset layouts already used by this repository."""
    if os.path.isdir(os.path.join(data_root, "IRSTD1k_Img")):
        return (
            os.path.join(data_root, "IRSTD1k_Img"),
            os.path.join(data_root, "IRSTD1k_Label"),
        )
    return os.path.join(data_root, "images"), os.path.join(data_root, "masks")


def load_or_build_annotations(data_root: str, mask_dir: str):
    """Load point annotations or simulate one centroid click per GT component.

    A few legacy SIRST-v1 files contain a mask at a different raster size than
    its image (e.g. ``Misc_111``).  Component centroids must be expressed in
    image coordinates because the generator crops the image; therefore the
    mask is explicitly resized with nearest-neighbour interpolation before
    extracting clicks.  The canonical mask file is never modified.
    """
    from scipy import ndimage

    json_path = os.path.join(data_root, "point_annotations.json")
    if os.path.isfile(json_path):
        with open(json_path) as handle:
            return json.load(handle)

    image_dir, _ = resolve_layout(data_root)
    annotations = {}
    for filename in tqdm(sorted(os.listdir(mask_dir)), desc="points from masks"):
        if not filename.lower().endswith(".png"):
            continue
        with Image.open(os.path.join(mask_dir, filename)) as mask_image:
            mask_image = mask_image.convert("L")
            image_path = os.path.join(image_dir, filename)
            if os.path.isfile(image_path):
                with Image.open(image_path) as source_image:
                    if mask_image.size != source_image.size:
                        mask_image = mask_image.resize(source_image.size, Image.NEAREST)
            mask = np.array(mask_image)
        components, count = ndimage.label(mask > 128)
        targets = []
        for component_id in range(1, count + 1):
            coords = np.argwhere(components == component_id)
            cy, cx = coords.mean(axis=0)
            targets.append({
                "centroid_y": float(cy),
                "centroid_x": float(cx),
                "size": int(len(coords)),
            })
        stem = os.path.splitext(filename)[0]
        annotations[stem] = {
            "image": filename,
            "num_targets": len(targets),
            "targets": targets,
        }

    with open(json_path, "w") as handle:
        json.dump(annotations, handle, indent=2)
    print(f"Wrote {json_path} ({len(annotations)} images)")
    return annotations


def normalize01(array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = array.astype(np.float32)
    low = float(values.min())
    span = float(values.max()) - low
    if span <= eps:
        return np.zeros_like(values, dtype=np.float32)
    return (values - low) / span


def percentile_normalize01(
    array: np.ndarray,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Normalize a local response robustly against isolated outliers.

    Percentiles are computed from the current point-centred patch only, so
    this optional path does not use dense masks or any additional supervision.
    The legacy min--max behavior remains the default for reproducibility.
    """
    values = array.astype(np.float32)
    low = float(np.percentile(values, low_percentile))
    high = float(np.percentile(values, high_percentile))
    span = high - low
    if span <= eps:
        return normalize01(values, eps)
    return np.clip((values - low) / span, 0.0, 1.0).astype(np.float32)


def normalize_for_generation(array: np.ndarray, mode: str) -> np.ndarray:
    if mode == "percentile":
        return percentile_normalize01(array)
    return normalize01(array)


def crop_point_window(image: np.ndarray, px: float, py: float, side: int):
    """Crop an odd square window and pad at image borders."""
    if side < 3 or side % 2 == 0:
        raise ValueError("--window-size must be an odd integer >= 3")

    half = side // 2
    ix, iy = int(round(px)), int(round(py))
    height, width = image.shape
    x0, x1 = max(0, ix - half), min(width, ix + half + 1)
    y0, y1 = max(0, iy - half), min(height, iy + half + 1)
    pad_left, pad_right = max(0, half - ix), max(0, ix + half + 1 - width)
    pad_top, pad_bottom = max(0, half - iy), max(0, iy + half + 1 - height)
    patch = np.pad(
        image[y0:y1, x0:x1],
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="edge",
    )
    point_x = ix - x0 + pad_left
    point_y = iy - y0 + pad_top
    placement = (x0, x1, y0, y1, pad_left, pad_right, pad_top, pad_bottom)
    return patch, point_x, point_y, placement


def morphological_skeleton(binary: np.ndarray) -> np.ndarray:
    """Morphological thinning using only the OpenCV dependency already present."""
    work = (binary > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(work)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(work):
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(work, opened))
        work = cv2.erode(work, element)
    return skeleton > 0


def prune_skeleton_spurs(skeleton: np.ndarray, iterations: int, keep_y: int, keep_x: int) -> np.ndarray:
    """Remove short skeleton branches by peeling end points, never the annotated point."""
    pruned = skeleton.copy()
    kernel = np.ones((3, 3), np.float32)
    kernel[1, 1] = 0.0
    for _ in range(int(iterations)):
        neighbours = cv2.filter2D(pruned.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
        ends = pruned & (neighbours <= 1.0)
        ends[keep_y, keep_x] = False
        if not ends.any():
            break
        remaining = pruned & ~ends
        if not remaining.any():
            break
        pruned = remaining
    return pruned


def extract_skeleton(
    patch: np.ndarray,
    point_x: int,
    point_y: int,
    coarse_threshold: float,
    tophat_kernel: int,
    tophat_kernels: Sequence[int] = (),
    normalization: str = "minmax",
    prune_spurs: int = 0,
    return_enhanced: bool = False,
):
    """Top-hat, point-component selection, and thinning.

    Optional multi-scale top-hat responses and robust normalization preserve
    the original skeleton-guided point-centered construction while reducing
    sensitivity to one kernel size or one saturated local pixel.
    """
    if tophat_kernel < 3 or tophat_kernel % 2 == 0:
        raise ValueError("--tophat-kernel must be an odd integer >= 3")
    kernels = tuple(tophat_kernels) if tophat_kernels else (tophat_kernel,)
    source = np.clip(patch, 0, 255).astype(np.uint8)
    enhanced_maps = []
    for kernel_size in kernels:
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("all top-hat kernels must be odd integers >= 3")
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        enhanced = cv2.morphologyEx(source, cv2.MORPH_TOPHAT, kernel)
        enhanced_maps.append(normalize_for_generation(enhanced, normalization))
    # Equalize each scale before fusion so a larger raw dynamic range does not
    # win merely because of its kernel size.
    enhanced_norm = np.maximum.reduce(enhanced_maps)
    coarse = enhanced_norm >= coarse_threshold

    point_y = int(np.clip(point_y, 0, coarse.shape[0] - 1))
    point_x = int(np.clip(point_x, 0, coarse.shape[1] - 1))
    if not coarse[point_y, point_x]:
        coarse[point_y, point_x] = True

    count, labels = cv2.connectedComponents(coarse.astype(np.uint8), connectivity=8)
    component_id = int(labels[point_y, point_x])
    if count <= 1 or component_id == 0:
        component = np.zeros_like(coarse, dtype=bool)
        component[point_y, point_x] = True
    else:
        component = labels == component_id

    skeleton = morphological_skeleton(component)
    if not skeleton.any():
        skeleton[point_y, point_x] = True
    if prune_spurs > 0:
        skeleton = prune_skeleton_spurs(skeleton, prune_spurs, point_y, point_x)
    if return_enhanced:
        return component, skeleton, enhanced_norm
    return component, skeleton


def skeleton_distance(skeleton: np.ndarray) -> np.ndarray:
    """Euclidean distance from every pixel to its nearest skeleton pixel."""
    non_skeleton = (~skeleton).astype(np.uint8)
    return cv2.distanceTransform(non_skeleton, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def gaussian_response(distance: np.ndarray, radius: float, sigma: float) -> np.ndarray:
    shifted = np.maximum(0.0, distance.astype(np.float32) - float(radius))
    return np.exp(-(shifted * shifted) / (2.0 * float(sigma) ** 2)).astype(np.float32)


def scale_regularization(sigma: float, sigma_min: float, sigma_max: float) -> float:
    if sigma > sigma_max:
        return float((sigma - sigma_max) ** 2)
    if sigma < sigma_min:
        return float((sigma_min - sigma) ** 2)
    return 0.0


def energy_terms(
    image_patch: np.ndarray,
    coarse_mask: np.ndarray,
    response: np.ndarray,
    sigma: float,
    support_threshold: float,
    sigma_reg_min: float,
    sigma_reg_max: float,
    normalization: str = "minmax",
    eps: float = 1e-8,
):
    """Compute E_fit, E_ctr, and E_reg from Eqs. (3)-(7)."""
    image_norm = normalize_for_generation(image_patch, normalization)
    response_norm = normalize_for_generation(response, normalization)
    support = coarse_mask | (response_norm >= support_threshold)
    if support.any():
        fit = float(np.mean((response_norm[support] - image_norm[support]) ** 2))
    else:
        fit = float(np.mean((response_norm - image_norm) ** 2))

    fg_den = float(response_norm.sum()) + eps
    bg_weight = 1.0 - response_norm
    bg_den = float(bg_weight.sum()) + eps
    intensity = image_patch.astype(np.float32)
    foreground = float((response_norm * intensity).sum()) / fg_den
    background = float((bg_weight * intensity).sum()) / bg_den
    contrast = (foreground - background) / (float(intensity.max()) + eps)
    regularization = scale_regularization(sigma, sigma_reg_min, sigma_reg_max)
    return fit, contrast, regularization


def radius_candidates(coarse_mask: np.ndarray, distance: np.ndarray, scales: Iterable[float]):
    inside_distances = distance[coarse_mask]
    base_radius = float(np.median(inside_distances)) if inside_distances.size else 0.0
    # A one-pixel-wide component has median distance zero. The 0.5 floor only
    # keeps the four-value search non-degenerate; its value remains tunable.
    base_radius = max(base_radius, 0.5)
    return [base_radius * float(scale) for scale in scales]


def select_response(
    image_patch: np.ndarray,
    coarse_mask: np.ndarray,
    skeleton: np.ndarray,
    r_scales: Sequence[float],
    sigma_values: Sequence[float],
    lambda_contrast: float,
    lambda_regularization: float,
    support_threshold: float,
    sigma_reg_min: float,
    sigma_reg_max: float,
    normalization: str = "minmax",
    support_consistency: float = 0.0,
):
    """Minimize the manuscript's multi-dimensional energy over (r, sigma)."""
    distance = skeleton_distance(skeleton)
    candidates_r = radius_candidates(coarse_mask, distance, r_scales)
    best_energy = float("inf")
    best_response = None
    best_parameters = None

    for radius in candidates_r:
        for sigma in sigma_values:
            response = gaussian_response(distance, radius, sigma)
            fit, contrast, regularization = energy_terms(
                image_patch,
                coarse_mask,
                response,
                sigma,
                support_threshold,
                sigma_reg_min,
                sigma_reg_max,
                normalization,
            )
            energy = fit - lambda_contrast * contrast + lambda_regularization * regularization
            if support_consistency > 0.0:
                # The contrast term alone favours ever narrower responses; this
                # term ties the response extent to the observed target support.
                predicted = response >= 0.5
                overlap = float(np.count_nonzero(predicted & coarse_mask))
                total = float(np.count_nonzero(predicted) + np.count_nonzero(coarse_mask))
                dice = 2.0 * overlap / total if total > 0 else 0.0
                energy = energy + support_consistency * (1.0 - dice)
            if energy < best_energy:
                best_energy = energy
                best_response = response
                best_parameters = (radius, float(sigma), fit, contrast, regularization)

    return best_response, best_parameters, best_energy


def place_patch(full: np.ndarray, patch: np.ndarray, placement) -> None:
    x0, x1, y0, y1, pad_left, pad_right, pad_top, pad_bottom = placement
    patch_y1 = patch.shape[0] - pad_bottom if pad_bottom else patch.shape[0]
    patch_x1 = patch.shape[1] - pad_right if pad_right else patch.shape[1]
    visible = patch[pad_top:patch_y1, pad_left:patch_x1]
    full[y0:y1, x0:x1] = np.maximum(full[y0:y1, x0:x1], visible)


def generate_initial_masks(data_root: str, image_dir: str, annotations, args) -> str:
    output_dir = args.output_dir or os.path.join(data_root, *DEFAULT_OUTPUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)
    r_scales = tuple(float(value) for value in args.r_scales.split(","))
    if len(r_scales) != 4:
        raise ValueError("The manuscript fixes four candidate r values; --r-scales needs 4 entries")
    sigma_values = np.linspace(
        args.sigma_search_min, args.sigma_search_max, args.sigma_candidates
    )
    tophat_kernels = tuple(
        int(value.strip()) for value in args.tophat_kernels.split(",") if value.strip()
    )

    print("[GRSR stage 1] Physics-Inspired Initial Pseudo Mask Generation")
    print(
        "  paper-fixed: coarse_threshold=%.2f, r_candidates=4, sigma_candidates=%d, "
        "lambda_c=%.2f, lambda_r=%.2f"
        % (
            args.coarse_threshold,
            args.sigma_candidates,
            args.lambda_contrast,
            args.lambda_regularization,
        )
    )
    print(
        "  optimization space: window=%d, tophat_kernel=%d, r_scales=%s, "
        "sigma_search=[%.2f, %.2f], sigma_regularization=[%.2f, %.2f], "
        "normalization=%s, tophat_kernels=%s"
        % (
            args.window_size,
            args.tophat_kernel,
            r_scales,
            args.sigma_search_min,
            args.sigma_search_max,
            args.sigma_reg_min,
            args.sigma_reg_max,
            args.normalization,
            tophat_kernels if tophat_kernels else (args.tophat_kernel,),
        )
    )

    for _, info in tqdm(annotations.items(), desc="GRSR initial masks"):
        filename = info["image"]
        output_path = os.path.join(output_dir, filename)
        if os.path.isfile(output_path) and not args.overwrite:
            continue

        image = np.array(Image.open(os.path.join(image_dir, filename)).convert("L"))
        heatmap = np.zeros(image.shape, dtype=np.float32)
        for target in info["targets"]:
            patch, point_x, point_y, placement = crop_point_window(
                image,
                target["centroid_x"],
                target["centroid_y"],
                args.window_size,
            )
            coarse_mask, skeleton, enhanced = extract_skeleton(
                patch,
                point_x,
                point_y,
                args.coarse_threshold,
                args.tophat_kernel,
                tophat_kernels,
                args.normalization,
                prune_spurs=int(getattr(args, "prune_spurs", 0)),
                return_enhanced=True,
            )
            response, _, _ = select_response(
                patch,
                coarse_mask,
                skeleton,
                r_scales,
                sigma_values,
                args.lambda_contrast,
                args.lambda_regularization,
                args.support_threshold,
                args.sigma_reg_min,
                args.sigma_reg_max,
                args.normalization,
                support_consistency=float(getattr(args, "support_consistency", 0.0)),
            )
            # Optional confidence calibration of the continuous Gaussian
            # response.  A power above one sharpens weak tails while keeping
            # the same skeleton-guided Gaussian construction; power=1.0 is
            # the legacy behavior.
            if args.response_power != 1.0:
                response = np.power(
                    np.clip(response, 0.0, 1.0), args.response_power
                ).astype(np.float32)
            eta = float(getattr(args, "radiance_eta", 0.0))
            if eta > 0.0:
                # Gaussian prior as the spatial envelope, observed radiance as the
                # shape inside it: a prior-times-observation response.
                peak = float(response.max())
                fused = np.power(np.clip(response, 0.0, 1.0), 1.0 - eta) * np.power(
                    np.clip(enhanced, 0.0, 1.0), eta
                )
                if float(fused.max()) > 0.0 and peak > 0.0:
                    response = (fused / float(fused.max()) * peak).astype(np.float32)
            place_patch(heatmap, response, placement)

        Image.fromarray((np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8)).save(
            output_path
        )

    print(f"Done -> {output_dir}")
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="GRSR stage 1: physics-inspired initial pseudo-mask generation"
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--overwrite", action="store_true")

    fixed = parser.add_argument_group("Paper-fixed settings")
    fixed.add_argument("--coarse_threshold", type=float, default=0.5)
    fixed.add_argument("--sigma_candidates", type=int, default=25)
    fixed.add_argument("--lambda_contrast", type=float, default=1.0)
    fixed.add_argument("--lambda_regularization", type=float, default=0.5)

    tuning = parser.add_argument_group("Optimization space (not fixed in the manuscript)")
    tuning.add_argument("--window_size", type=int, default=31)
    tuning.add_argument("--tophat_kernel", type=int, default=5)
    tuning.add_argument(
        "--tophat_kernels",
        default="",
        help="Optional comma-separated multi-scale top-hat kernels, e.g. 3,5,7",
    )
    tuning.add_argument(
        "--normalization",
        choices=("minmax", "percentile"),
        default="minmax",
        help="Local response normalization; percentile is an optional robust variant",
    )
    tuning.add_argument(
        "--response_power",
        type=float,
        default=1.0,
        help="Power calibration for Gaussian confidence tails; 1.0 is legacy",
    )
    tuning.add_argument("--r_scales", default="0.5,1.0,1.5,2.0")
    tuning.add_argument("--sigma_search_min", type=float, default=0.5)
    tuning.add_argument("--sigma_search_max", type=float, default=6.5)
    tuning.add_argument("--sigma_reg_min", type=float, default=1.0)
    tuning.add_argument("--sigma_reg_max", type=float, default=4.0)
    tuning.add_argument("--support_threshold", type=float, default=0.5)
    tuning.add_argument("--prune_spurs", type=int, default=0, help="Peel skeleton end points this many times; 0 keeps the legacy skeleton")
    tuning.add_argument("--radiance_eta", type=float, default=0.0, help="Weight of observed radiance in the prior-times-observation response; 0 is legacy")
    tuning.add_argument("--support_consistency", type=float, default=0.0, help="Weight of the support-consistency energy term; 0 is legacy")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.response_power <= 0.0:
        raise ValueError("--response_power must be positive")
    image_dir, mask_dir = resolve_layout(args.data_root)
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
    annotations = load_or_build_annotations(args.data_root, mask_dir)
    generate_initial_masks(args.data_root, image_dir, annotations, args)


if __name__ == "__main__":
    main()

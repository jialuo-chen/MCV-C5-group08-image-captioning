from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

DEFAULT_ANNOTATION_FILE = Path(
    "outputs/synthetic_data/annotations/synthetic_multicap_qwen36_aug3.json"
)
DEFAULT_SOURCE_IMAGE_DIR = Path("outputs/synthetic_data/images_aug3")
DEFAULT_SUBSET_FRACTION = 0.2

GENERIC_UNCLEAR_CAPTIONS = (
    "The image is too unclear to identify the object.",
    "The photo quality is too poor to tell what is shown.",
    "I cannot tell what this is because the image is unclear.",
    "The object cannot be identified from this picture.",
    "The image is too distorted to recognize the item.",
    "The picture is unclear, so the object cannot be identified.",
)

UNCLEAR_CAPTION_BANKS: dict[str, tuple[str, ...]] = {
    "too_dark": (
        "The image is too dark to identify the object.",
        "The photo is too dark to tell what is shown.",
        "Poor lighting makes the object impossible to identify.",
        "The scene is too dim to recognize the item.",
        "The picture is too dark to tell what the object is.",
    ),
    "overexposed": (
        "The image is too bright to identify the object.",
        "The photo is overexposed, so the object cannot be recognized.",
        "Glare washes out the scene and makes the object unclear.",
        "The picture is too bright to tell what is shown.",
        "The lighting is blown out, so the item cannot be identified.",
    ),
    "heavy_blur": (
        "The image is too blurry to identify the object.",
        "The photo is out of focus, so I cannot tell what this is.",
        "The picture is too blurred to recognize the item.",
        "Blur makes the object impossible to identify.",
        "The object cannot be identified because the image is too blurry.",
    ),
    "motion_blur": (
        "The image has too much motion blur to identify the object.",
        "The photo is blurred by movement, so the object is unclear.",
        "Motion blur makes it impossible to tell what is shown.",
        "The picture is smeared by motion and the item cannot be recognized.",
        "The object cannot be identified because the image is blurred by motion.",
    ),
    "occluded": (
        "The object is too blocked to identify clearly.",
        "Part of the scene is covered, so the object cannot be recognized.",
        "The item is too obstructed to tell what it is.",
        "An occlusion blocks the object and makes it hard to identify.",
        "The picture is partially covered, so the object is unclear.",
    ),
    "off_frame": (
        "The object is cut off, so it cannot be identified.",
        "Only part of the object is visible, making it hard to tell what it is.",
        "The subject is mostly out of frame, so the object is unclear.",
        "The image is cropped too tightly to identify the object.",
        "The item is only partially visible, so it cannot be recognized.",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a low-quality synthetic subset with cannot-identify captions and a "
            "merged VizWiz-style annotation file."
        )
    )
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=DEFAULT_ANNOTATION_FILE,
        help="Input VizWiz-style annotation file.",
    )
    parser.add_argument(
        "--source-image-dir",
        type=Path,
        default=DEFAULT_SOURCE_IMAGE_DIR,
        help="Directory containing source synthetic images.",
    )
    parser.add_argument(
        "--output-image-dir",
        type=Path,
        default=None,
        help="Output directory containing the merged original images and unclear variants.",
    )
    parser.add_argument(
        "--output-ann",
        type=Path,
        default=None,
        help="Output merged VizWiz-style annotation file.",
    )
    parser.add_argument(
        "--subset-ann",
        type=Path,
        default=None,
        help="Output VizWiz-style annotation file containing only the unclear variants.",
    )
    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=None,
        help="Manifest describing selected source images, degradations, and captions.",
    )
    parser.add_argument(
        "--subset-fraction",
        type=float,
        default=DEFAULT_SUBSET_FRACTION,
        help="Fraction of eligible source images to turn into unclear examples.",
    )
    parser.add_argument(
        "--subset-count",
        type=int,
        default=None,
        help="Absolute number of eligible source images to select. Overrides subset-fraction.",
    )
    parser.add_argument(
        "--variants-per-image",
        type=int,
        default=1,
        help="How many different unclear variants to create for each selected source image.",
    )
    parser.add_argument(
        "--captions-per-image",
        type=int,
        default=5,
        help="How many cannot-identify captions to assign to each unclear variant.",
    )
    parser.add_argument(
        "--include-generated-variants",
        action="store_true",
        help=(
            "Allow previously generated variants such as *_augA.png to be selected as "
            "sources for the unclear subset."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, max(1, (os.cpu_count() or 1))),
        help="Number of parallel worker threads.",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        default=1,
        help="PNG compression level from 0 to 9. Lower is faster.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed used for selection and degradation parameters.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for small dry runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate output image and annotation artifacts from scratch.",
    )
    return parser.parse_args()


def normalize_caption(text: str) -> str:
    text = " ".join(str(text).replace("\n", " ").split()).strip()
    if not text:
        return ""
    if text[0].islower():
        text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


def build_default_output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    if args.subset_count is None:
        label = f"unclear{int(round(args.subset_fraction * 100)):02d}"
    else:
        label = f"unclear{args.subset_count}"
    output_image_dir = args.output_image_dir or (
        args.source_image_dir.parent / f"{args.source_image_dir.name}_{label}"
    )
    output_ann = args.output_ann or (
        args.annotation_file.parent / f"{args.annotation_file.stem}_{label}.json"
    )
    subset_ann = args.subset_ann or (
        args.annotation_file.parent / f"{args.annotation_file.stem}_{label}_subset.json"
    )
    manifest_file = args.manifest_file or (
        args.annotation_file.parent / f"{args.annotation_file.stem}_{label}_manifest.json"
    )
    return output_image_dir, output_ann, subset_ann, manifest_file


def validate_args(args: argparse.Namespace) -> None:
    if args.subset_count is None and not (0.0 < args.subset_fraction <= 1.0):
        raise ValueError("subset-fraction must be in the interval (0, 1].")
    if args.subset_count is not None and args.subset_count < 1:
        raise ValueError("subset-count must be at least 1.")
    if args.variants_per_image < 1:
        raise ValueError("variants-per-image must be at least 1.")
    if args.variants_per_image > len(UNCLEAR_CAPTION_BANKS):
        raise ValueError(
            "variants-per-image exceeds the number of available unclear degradations."
        )
    if args.captions_per_image < 1:
        raise ValueError("captions-per-image must be at least 1.")
    if args.workers < 1:
        raise ValueError("workers must be at least 1.")
    if not 0 <= args.png_compression <= 9:
        raise ValueError("png-compression must be between 0 and 9.")


def load_dataset(annotation_file: Path) -> dict[str, Any]:
    dataset = json.loads(annotation_file.read_text())
    if not isinstance(dataset, dict):
        raise ValueError(f"Expected a dict in {annotation_file}.")
    if not isinstance(dataset.get("images"), list):
        raise ValueError("Annotation file must contain an images list.")
    if not isinstance(dataset.get("annotations"), list):
        raise ValueError("Annotation file must contain an annotations list.")
    return dataset


def build_items(
    dataset: dict[str, Any],
    source_image_dir: Path,
    limit: int | None,
) -> list[dict[str, Any]]:
    captions_by_image: dict[int, list[str]] = defaultdict(list)
    for annotation in dataset["annotations"]:
        image_id = int(annotation["image_id"])
        caption = normalize_caption(annotation.get("caption", ""))
        if caption and caption not in captions_by_image[image_id]:
            captions_by_image[image_id].append(caption)

    items = []
    for image in sorted(dataset["images"], key=lambda item: int(item["id"])):
        image_id = int(image["id"])
        file_name = str(image["file_name"])
        source_path = source_image_dir / file_name
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source image: {source_path}")
        items.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "width": int(image.get("width", 0)),
                "height": int(image.get("height", 0)),
                "source_path": source_path,
                "source_captions": captions_by_image.get(image_id, []),
            }
        )
    if limit is not None:
        return items[:limit]
    return items


def build_base_dataset(
    dataset: dict[str, Any],
    allowed_image_ids: set[int],
) -> dict[str, Any]:
    images = [
        dict(image)
        for image in dataset["images"]
        if int(image["id"]) in allowed_image_ids
    ]
    annotations = [
        dict(annotation)
        for annotation in dataset["annotations"]
        if int(annotation["image_id"]) in allowed_image_ids
    ]
    return {"images": images, "annotations": annotations}


def is_generated_variant_name(file_name: str) -> bool:
    stem = Path(file_name).stem.lower()
    return bool(re.search(r"_aug[a-z]+$", stem)) or "_unclear" in stem


def select_source_indices(
    items: list[dict[str, Any]],
    *,
    subset_fraction: float,
    subset_count: int | None,
    include_generated_variants: bool,
    seed: int,
) -> tuple[list[int], list[int]]:
    candidate_indices = [
        index
        for index, item in enumerate(items)
        if include_generated_variants
        or not is_generated_variant_name(str(item["file_name"]))
    ]
    if not candidate_indices:
        raise ValueError(
            "No eligible source images found. Use --include-generated-variants to allow generated source files."
        )

    if subset_count is None:
        selection_size = int(round(len(candidate_indices) * subset_fraction))
        if subset_fraction > 0.0:
            selection_size = max(1, selection_size)
    else:
        selection_size = subset_count
    selection_size = min(selection_size, len(candidate_indices))
    chooser = random.Random(seed)
    selected_indices = sorted(chooser.sample(candidate_indices, k=selection_size))
    return selected_indices, candidate_indices


def ensure_original_in_output(source_path: Path, target_path: Path) -> str:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return "existing"
    try:
        target_path.symlink_to(source_path.resolve())
        return "symlink"
    except OSError:
        shutil.copy2(source_path, target_path)
        return "copy"


def load_image_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image_rgb(path: Path, image: np.ndarray, png_compression: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(
        str(path),
        bgr_image,
        [cv2.IMWRITE_PNG_COMPRESSION, png_compression],
    )
    if not ok:
        raise RuntimeError(f"Could not write image: {path}")


def clamp_image(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    normalized = np.clip(image.astype(np.float32) / 255.0, 0.0, 1.0)
    corrected = np.power(normalized, gamma)
    return corrected * 255.0


def adjust_saturation(image: np.ndarray, saturation: float) -> np.ndarray:
    hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)


def add_gaussian_noise(
    image: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    noise = rng.normal(0.0, sigma, size=image.shape)
    return image.astype(np.float32) + noise


def add_soft_highlight(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
    strength: float,
) -> np.ndarray:
    height, width = image.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    mask = np.exp(
        -(
            ((xx - center_x) ** 2) / (2.0 * max(radius_x, 1.0) ** 2)
            + ((yy - center_y) ** 2) / (2.0 * max(radius_y, 1.0) ** 2)
        )
    )
    highlight = 255.0 * strength * mask[..., None]
    return image.astype(np.float32) + highlight


def build_motion_blur_kernel(length: int, angle_deg: float) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    center = ((length - 1) / 2.0, (length - 1) / 2.0)
    rotation = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (length, length))
    kernel_sum = float(kernel.sum())
    if kernel_sum <= 0.0:
        kernel[length // 2, length // 2] = 1.0
        kernel_sum = 1.0
    return kernel / kernel_sum


def make_too_dark_params(width: int, height: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    return {
        "kind": "too_dark",
        "brightness_shift": round(rng.uniform(-110.0, -62.0), 4),
        "contrast": round(rng.uniform(0.72, 0.96), 4),
        "gamma": round(rng.uniform(1.45, 2.1), 4),
        "saturation": round(rng.uniform(0.55, 0.88), 4),
        "noise_sigma": round(rng.uniform(12.0, 30.0), 4),
        "width": width,
        "height": height,
    }


def make_overexposed_params(width: int, height: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    center_x = rng.uniform(width * 0.2, width * 0.8)
    center_y = rng.uniform(height * 0.15, height * 0.85)
    return {
        "kind": "overexposed",
        "brightness_shift": round(rng.uniform(52.0, 106.0), 4),
        "contrast": round(rng.uniform(0.76, 0.98), 4),
        "gamma": round(rng.uniform(0.42, 0.76), 4),
        "saturation": round(rng.uniform(0.8, 1.04), 4),
        "highlight_strength": round(rng.uniform(0.22, 0.42), 4),
        "highlight_center_x": round(center_x, 4),
        "highlight_center_y": round(center_y, 4),
        "highlight_radius_x": round(rng.uniform(width * 0.22, width * 0.52), 4),
        "highlight_radius_y": round(rng.uniform(height * 0.22, height * 0.52), 4),
        "width": width,
        "height": height,
    }


def make_heavy_blur_params(width: int, height: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    blur_sigma = rng.uniform(2.6, 4.8)
    kernel_size = int(max(5, 2 * math.ceil(blur_sigma * 2.0) + 1))
    return {
        "kind": "heavy_blur",
        "blur_sigma": round(blur_sigma, 4),
        "kernel_size": kernel_size,
        "downsample_scale": round(rng.uniform(0.22, 0.45), 4),
        "brightness_shift": round(rng.uniform(-12.0, 10.0), 4),
        "contrast": round(rng.uniform(0.82, 1.02), 4),
        "noise_sigma": round(rng.uniform(1.0, 6.0), 4),
        "width": width,
        "height": height,
    }


def make_motion_blur_params(width: int, height: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    blur_length = rng.randrange(17, 37, 2)
    return {
        "kind": "motion_blur",
        "blur_length": blur_length,
        "angle_deg": round(rng.uniform(0.0, 180.0), 4),
        "brightness_shift": round(rng.uniform(-18.0, 6.0), 4),
        "contrast": round(rng.uniform(0.8, 1.0), 4),
        "gamma": round(rng.uniform(0.92, 1.24), 4),
        "noise_sigma": round(rng.uniform(1.0, 5.0), 4),
        "width": width,
        "height": height,
    }


def make_occluded_params(width: int, height: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    occ_width = int(rng.uniform(width * 0.28, width * 0.56))
    occ_height = int(rng.uniform(height * 0.24, height * 0.52))
    x0 = int(rng.uniform(0, max(0, width - occ_width)))
    y0 = int(rng.uniform(0, max(0, height - occ_height)))
    return {
        "kind": "occluded",
        "x0": x0,
        "y0": y0,
        "occ_width": occ_width,
        "occ_height": occ_height,
        "fill_value": round(rng.uniform(10.0, 90.0), 4),
        "opacity": round(rng.uniform(0.82, 0.96), 4),
        "mask_blur_sigma": round(rng.uniform(4.0, 9.0), 4),
        "width": width,
        "height": height,
    }


def make_off_frame_params(width: int, height: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    crop_width = int(rng.uniform(width * 0.5, width * 0.76))
    crop_height = int(rng.uniform(height * 0.5, height * 0.76))
    anchors = [
        (0, 0),
        (max(0, width - crop_width), 0),
        (0, max(0, height - crop_height)),
        (max(0, width - crop_width), max(0, height - crop_height)),
    ]
    x0, y0 = anchors[rng.randrange(len(anchors))]
    return {
        "kind": "off_frame",
        "x0": x0,
        "y0": y0,
        "crop_width": crop_width,
        "crop_height": crop_height,
        "brightness_shift": round(rng.uniform(-10.0, 8.0), 4),
        "contrast": round(rng.uniform(0.82, 1.02), 4),
        "gamma": round(rng.uniform(0.92, 1.18), 4),
        "width": width,
        "height": height,
    }


UNCLEAR_BUILDERS: dict[str, Any] = {
    "too_dark": make_too_dark_params,
    "overexposed": make_overexposed_params,
    "heavy_blur": make_heavy_blur_params,
    "motion_blur": make_motion_blur_params,
    "occluded": make_occluded_params,
    "off_frame": make_off_frame_params,
}


def build_variant_params(
    image_id: int,
    width: int,
    height: int,
    variants_per_image: int,
    seed: int,
) -> list[dict[str, Any]]:
    chooser = random.Random(seed + image_id * 1009 + width * 17 + height * 31)
    selected_kinds = chooser.sample(list(UNCLEAR_BUILDERS), k=variants_per_image)
    params = []
    for variant_index, kind in enumerate(selected_kinds):
        local_seed = (
            seed
            + image_id * 1009
            + variant_index * 9173
            + sum(ord(char) for char in kind)
        )
        entry = UNCLEAR_BUILDERS[kind](width, height, local_seed)
        entry["variant_index"] = variant_index
        entry["variant_suffix"] = chr(ord("A") + variant_index)
        params.append(entry)
    return params


def apply_too_dark(
    image: np.ndarray,
    params: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    working = apply_gamma(image, float(params["gamma"]))
    working = (working - 127.5) * float(params["contrast"]) + 127.5
    working = working + float(params["brightness_shift"])
    working = clamp_image(working)
    working = adjust_saturation(working, float(params["saturation"]))
    working = add_gaussian_noise(working, float(params["noise_sigma"]), rng)
    return clamp_image(working)


def apply_overexposed(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    working = apply_gamma(image, float(params["gamma"]))
    working = (working - 127.5) * float(params["contrast"]) + 127.5
    working = working + float(params["brightness_shift"])
    working = clamp_image(working)
    working = adjust_saturation(working, float(params["saturation"]))
    working = add_soft_highlight(
        working,
        center_x=float(params["highlight_center_x"]),
        center_y=float(params["highlight_center_y"]),
        radius_x=float(params["highlight_radius_x"]),
        radius_y=float(params["highlight_radius_y"]),
        strength=float(params["highlight_strength"]),
    )
    return clamp_image(working)


def apply_heavy_blur(
    image: np.ndarray,
    params: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    height, width = image.shape[:2]
    scale = float(params["downsample_scale"])
    reduced = cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_LINEAR,
    )
    working = cv2.resize(reduced, (width, height), interpolation=cv2.INTER_LINEAR)
    working = cv2.GaussianBlur(
        working,
        (int(params["kernel_size"]), int(params["kernel_size"])),
        sigmaX=float(params["blur_sigma"]),
        sigmaY=float(params["blur_sigma"]),
    ).astype(np.float32)
    working = (working - 127.5) * float(params["contrast"]) + 127.5
    working = working + float(params["brightness_shift"])
    working = add_gaussian_noise(working, float(params["noise_sigma"]), rng)
    return clamp_image(working)


def apply_motion_blur(
    image: np.ndarray,
    params: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    kernel = build_motion_blur_kernel(
        int(params["blur_length"]), float(params["angle_deg"])
    )
    working = cv2.filter2D(image.astype(np.float32), -1, kernel)
    working = apply_gamma(working, float(params["gamma"]))
    working = (working - 127.5) * float(params["contrast"]) + 127.5
    working = working + float(params["brightness_shift"])
    working = add_gaussian_noise(working, float(params["noise_sigma"]), rng)
    return clamp_image(working)


def apply_occluded(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    height, width = image.shape[:2]
    x0 = int(params["x0"])
    y0 = int(params["y0"])
    x1 = min(width, x0 + int(params["occ_width"]))
    y1 = min(height, y0 + int(params["occ_height"]))
    mask = np.zeros((height, width), dtype=np.float32)
    mask[y0:y1, x0:x1] = 1.0
    blur_sigma = float(params["mask_blur_sigma"])
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    alpha = np.clip(mask * float(params["opacity"]), 0.0, 1.0)[..., None]
    fill_value = float(params["fill_value"])
    fill = np.full_like(image, fill_value, dtype=np.float32)
    working = image.astype(np.float32) * (1.0 - alpha) + fill * alpha
    return clamp_image(working)


def apply_off_frame(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    height, width = image.shape[:2]
    x0 = int(params["x0"])
    y0 = int(params["y0"])
    crop_width = max(1, int(params["crop_width"]))
    crop_height = max(1, int(params["crop_height"]))
    x1 = min(width, x0 + crop_width)
    y1 = min(height, y0 + crop_height)
    cropped = image[y0:y1, x0:x1]
    working = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
    working = apply_gamma(working, float(params["gamma"]))
    working = (working - 127.5) * float(params["contrast"]) + 127.5
    working = working + float(params["brightness_shift"])
    return clamp_image(working)


def apply_variant(
    image: np.ndarray,
    params: dict[str, Any],
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if params["kind"] == "too_dark":
        return apply_too_dark(image, params, rng)
    if params["kind"] == "overexposed":
        return apply_overexposed(image, params)
    if params["kind"] == "heavy_blur":
        return apply_heavy_blur(image, params, rng)
    if params["kind"] == "motion_blur":
        return apply_motion_blur(image, params, rng)
    if params["kind"] == "occluded":
        return apply_occluded(image, params)
    if params["kind"] == "off_frame":
        return apply_off_frame(image, params)
    raise ValueError(f"Unsupported variant kind: {params['kind']}")


def build_unclear_captions(
    kind: str,
    captions_per_image: int,
    seed: int,
) -> list[str]:
    bank = []
    for caption in UNCLEAR_CAPTION_BANKS.get(kind, ()) + GENERIC_UNCLEAR_CAPTIONS:
        normalized = normalize_caption(caption)
        if normalized and normalized not in bank:
            bank.append(normalized)
    if captions_per_image > len(bank):
        raise ValueError(
            f"Requested {captions_per_image} captions for kind={kind}, but only {len(bank)} unique captions are available."
        )
    chooser = random.Random(seed + sum(ord(char) for char in kind))
    return chooser.sample(bank, k=captions_per_image)


def compute_variant_image_id(
    selected_order: int,
    variant_index: int,
    first_variant_image_id: int,
    variants_per_image: int,
) -> int:
    return first_variant_image_id + selected_order * variants_per_image + variant_index


def process_item(
    item_index: int,
    item: dict[str, Any],
    *,
    output_image_dir: Path,
    selected_orders: dict[int, int],
    variants_per_image: int,
    captions_per_image: int,
    seed: int,
    png_compression: int,
    first_variant_image_id: int,
) -> dict[str, Any]:
    output_original = output_image_dir / item["file_name"]
    link_mode = ensure_original_in_output(item["source_path"], output_original)
    selected_order = selected_orders.get(item_index)
    if selected_order is None:
        return {
            "item_index": item_index,
            "source_image_id": item["image_id"],
            "source_file_name": item["file_name"],
            "source_caption_count": len(item["source_captions"]),
            "output_original_mode": link_mode,
            "variants": [],
        }

    source_image = load_image_rgb(item["source_path"])
    height, width = source_image.shape[:2]
    variant_entries = []
    for params in build_variant_params(
        image_id=item["image_id"],
        width=width,
        height=height,
        variants_per_image=variants_per_image,
        seed=seed,
    ):
        variant_index = int(params["variant_index"])
        kind = str(params["kind"])
        stem = Path(item["file_name"]).stem
        variant_name = f"{stem}_unclear_{kind}_{variant_index + 1}.png"
        variant_path = output_image_dir / variant_name
        variant_seed = seed + item["image_id"] * 1009 + variant_index * 9173
        if not variant_path.exists():
            variant_image = apply_variant(source_image, params, variant_seed)
            save_image_rgb(variant_path, variant_image, png_compression)
        variant_entries.append(
            {
                "image_id": compute_variant_image_id(
                    selected_order=selected_order,
                    variant_index=variant_index,
                    first_variant_image_id=first_variant_image_id,
                    variants_per_image=variants_per_image,
                ),
                "file_name": variant_name,
                "kind": kind,
                "captions": build_unclear_captions(
                    kind=kind,
                    captions_per_image=captions_per_image,
                    seed=variant_seed,
                ),
                "params": params,
                "width": width,
                "height": height,
            }
        )

    return {
        "item_index": item_index,
        "source_image_id": item["image_id"],
        "source_file_name": item["file_name"],
        "source_caption_count": len(item["source_captions"]),
        "output_original_mode": link_mode,
        "variants": variant_entries,
    }


def build_outputs(
    base_dataset: dict[str, Any],
    items: list[dict[str, Any]],
    *,
    output_image_dir: Path,
    selected_indices: list[int],
    variants_per_image: int,
    captions_per_image: int,
    seed: int,
    workers: int,
    png_compression: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    selected_orders = {
        item_index: selected_order
        for selected_order, item_index in enumerate(selected_indices)
    }
    first_variant_image_id = max(item["image_id"] for item in items) + 1 if items else 1

    cv2.setNumThreads(1)
    results: list[dict[str, Any] | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_item,
                item_index,
                item,
                output_image_dir=output_image_dir,
                selected_orders=selected_orders,
                variants_per_image=variants_per_image,
                captions_per_image=captions_per_image,
                seed=seed,
                png_compression=png_compression,
                first_variant_image_id=first_variant_image_id,
            ): item_index
            for item_index, item in enumerate(items)
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Creating unclear subset",
        ):
            result = future.result()
            results[result["item_index"]] = result

    images = [dict(image) for image in base_dataset["images"]]
    annotations = [dict(annotation) for annotation in base_dataset["annotations"]]
    subset_images = []
    subset_annotations = []
    annotation_id = (
        max((int(annotation.get("id", 0)) for annotation in annotations), default=0) + 1
    )
    subset_annotation_id = 1
    kind_counts: dict[str, int] = defaultdict(int)
    manifest_items = []

    for item_index, item in enumerate(items):
        result = results[item_index]
        if result is None:
            raise RuntimeError(
                f"Missing unclear-subset result for image {item['image_id']}"
            )
        if not result["variants"]:
            continue
        manifest_variants = []
        for variant in result["variants"]:
            image_entry = {
                "id": variant["image_id"],
                "file_name": variant["file_name"],
                "width": variant["width"],
                "height": variant["height"],
            }
            images.append(image_entry)
            subset_images.append(dict(image_entry))
            kind_counts[str(variant["kind"])] += 1
            for caption in variant["captions"]:
                merged_ann = {
                    "id": annotation_id,
                    "image_id": variant["image_id"],
                    "caption": caption,
                    "is_rejected": False,
                    "is_precanned": False,
                }
                annotations.append(merged_ann)
                annotation_id += 1
                subset_annotations.append(
                    {
                        "id": subset_annotation_id,
                        "image_id": variant["image_id"],
                        "caption": caption,
                        "is_rejected": False,
                        "is_precanned": False,
                    }
                )
                subset_annotation_id += 1
            manifest_variants.append(
                {
                    "image_id": variant["image_id"],
                    "file_name": variant["file_name"],
                    "kind": variant["kind"],
                    "captions": variant["captions"],
                    "params": variant["params"],
                }
            )
        manifest_items.append(
            {
                "source_image_id": result["source_image_id"],
                "source_file_name": result["source_file_name"],
                "source_caption_count": result["source_caption_count"],
                "output_original_mode": result["output_original_mode"],
                "variants": manifest_variants,
            }
        )

    merged_annotation = {"images": images, "annotations": annotations}
    subset_annotation = {"images": subset_images, "annotations": subset_annotations}
    manifest = {
        "source_images": len(items),
        "selected_source_images": len(selected_indices),
        "variants_per_image": variants_per_image,
        "captions_per_image": captions_per_image,
        "total_images": len(images),
        "total_annotations": len(annotations),
        "unclear_images": len(subset_images),
        "unclear_annotations": len(subset_annotations),
        "kind_counts": dict(sorted(kind_counts.items())),
        "items": manifest_items,
    }
    return merged_annotation, subset_annotation, manifest


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_image_dir, output_ann, subset_ann, manifest_file = build_default_output_paths(
        args
    )

    if args.overwrite and output_image_dir.exists():
        shutil.rmtree(output_image_dir)
    if args.overwrite and output_ann.exists():
        output_ann.unlink()
    if args.overwrite and subset_ann.exists():
        subset_ann.unlink()
    if args.overwrite and manifest_file.exists():
        manifest_file.unlink()

    dataset = load_dataset(args.annotation_file)
    items = build_items(dataset, args.source_image_dir, args.limit)
    if not items:
        raise ValueError("No source images were found to process.")

    allowed_image_ids = {item["image_id"] for item in items}
    base_dataset = build_base_dataset(dataset, allowed_image_ids)
    selected_indices, candidate_indices = select_source_indices(
        items,
        subset_fraction=args.subset_fraction,
        subset_count=args.subset_count,
        include_generated_variants=args.include_generated_variants,
        seed=args.seed,
    )

    output_image_dir.mkdir(parents=True, exist_ok=True)
    merged_annotation, subset_annotation, manifest = build_outputs(
        base_dataset,
        items,
        output_image_dir=output_image_dir,
        selected_indices=selected_indices,
        variants_per_image=args.variants_per_image,
        captions_per_image=args.captions_per_image,
        seed=args.seed,
        workers=args.workers,
        png_compression=args.png_compression,
    )

    expected_unclear_images = len(selected_indices) * args.variants_per_image
    expected_unclear_annotations = expected_unclear_images * args.captions_per_image
    if len(subset_annotation["images"]) != expected_unclear_images:
        raise RuntimeError(
            f"Expected {expected_unclear_images} unclear images, found {len(subset_annotation['images'])}."
        )
    if len(subset_annotation["annotations"]) != expected_unclear_annotations:
        raise RuntimeError(
            "Unexpected unclear annotation count: "
            f"expected {expected_unclear_annotations}, found {len(subset_annotation['annotations'])}."
        )

    output_ann.parent.mkdir(parents=True, exist_ok=True)
    subset_ann.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    output_ann.write_text(json.dumps(merged_annotation, indent=2), encoding="utf-8")
    subset_ann.write_text(json.dumps(subset_annotation, indent=2), encoding="utf-8")

    manifest["annotation_file"] = str(args.annotation_file)
    manifest["source_image_dir"] = str(args.source_image_dir)
    manifest["output_image_dir"] = str(output_image_dir)
    manifest["output_annotation_file"] = str(output_ann)
    manifest["subset_annotation_file"] = str(subset_ann)
    manifest["manifest_file"] = str(manifest_file)
    manifest["seed"] = args.seed
    manifest["workers"] = args.workers
    manifest["png_compression"] = args.png_compression
    manifest["include_generated_variants"] = args.include_generated_variants
    manifest["eligible_source_images"] = len(candidate_indices)
    manifest["selected_fraction_requested"] = args.subset_fraction
    manifest["selected_fraction_effective"] = round(
        len(selected_indices) / max(1, len(candidate_indices)), 6
    )
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Eligible source images: {len(candidate_indices)}")
    print(f"Selected source images: {len(selected_indices)}")
    print(f"Saved merged image directory to: {output_image_dir}")
    print(f"Saved merged annotations to: {output_ann}")
    print(f"Saved unclear-only annotations to: {subset_ann}")
    print(f"Saved unclear manifest to: {manifest_file}")


if __name__ == "__main__":
    main()

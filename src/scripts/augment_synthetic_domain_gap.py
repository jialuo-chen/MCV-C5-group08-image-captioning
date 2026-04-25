from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

DEFAULT_ANNOTATION_FILE = Path(
    "outputs/synthetic_data/annotations/synthetic_multicap_qwen36.json"
)
DEFAULT_SOURCE_IMAGE_DIR = Path("outputs/synthetic_data/images")
DEFAULT_OUTPUT_IMAGE_DIR = Path("outputs/synthetic_data/images_aug3")
DEFAULT_OUTPUT_ANN = Path(
    "outputs/synthetic_data/annotations/synthetic_multicap_qwen36_aug3.json"
)
DEFAULT_MANIFEST_FILE = Path(
    "outputs/synthetic_data/annotations/synthetic_aug3_manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create controlled perturbed variants for synthetic training images."
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
        help="Directory containing original synthetic images.",
    )
    parser.add_argument(
        "--output-image-dir",
        type=Path,
        default=DEFAULT_OUTPUT_IMAGE_DIR,
        help="Output directory containing originals plus variants.",
    )
    parser.add_argument(
        "--output-ann",
        type=Path,
        default=DEFAULT_OUTPUT_ANN,
        help="Output merged VizWiz-style annotation file.",
    )
    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=DEFAULT_MANIFEST_FILE,
        help="Manifest describing generated variants and parameters.",
    )
    parser.add_argument(
        "--variants-per-image",
        type=int,
        default=3,
        help="Number of augmented variants to create per original image.",
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
        help="Deterministic seed used to sample augmentation parameters.",
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
        caption = normalize_caption(annotation.get("caption", ""))
        if caption:
            captions_by_image[int(annotation["image_id"])] += [caption]

    items = []
    for image in sorted(dataset["images"], key=lambda item: int(item["id"])):
        image_id = int(image["id"])
        file_name = str(image["file_name"])
        source_path = source_image_dir / file_name
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source image: {source_path}")
        captions = []
        for caption in captions_by_image.get(image_id, []):
            if caption not in captions:
                captions.append(caption)
        if not captions:
            continue
        items.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "width": int(image.get("width", 0)),
                "height": int(image.get("height", 0)),
                "source_path": source_path,
                "captions": captions,
            }
        )
    if limit is not None:
        return items[:limit]
    return items


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
    image: np.ndarray, sigma: float, rng: np.random.Generator
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


def make_noise_photometric_params(
    width: int,
    height: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    exposure_mode = "dark" if rng.random() < 0.5 else "bright"
    brightness_shift = (
        rng.uniform(-38.0, -12.0) if exposure_mode == "dark" else rng.uniform(10.0, 34.0)
    )
    return {
        "kind": "noise_photometric",
        "brightness_shift": round(brightness_shift, 4),
        "contrast": round(rng.uniform(0.92, 1.12), 4),
        "gamma": round(rng.uniform(0.85, 1.18), 4),
        "saturation": round(rng.uniform(0.9, 1.08), 4),
        "noise_sigma": round(rng.uniform(6.0, 18.0), 4),
        "height": height,
        "width": width,
        "exposure_mode": exposure_mode,
    }


def make_blur_exposure_params(
    width: int,
    height: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    blur_sigma = rng.uniform(0.9, 2.4)
    kernel_size = int(max(3, 2 * math.ceil(blur_sigma * 1.5) + 1))
    center_x = rng.uniform(width * 0.2, width * 0.8)
    center_y = rng.uniform(height * 0.15, height * 0.85)
    return {
        "kind": "blur_exposure",
        "blur_sigma": round(blur_sigma, 4),
        "kernel_size": kernel_size,
        "brightness_shift": round(rng.uniform(-16.0, 22.0), 4),
        "contrast": round(rng.uniform(0.9, 1.1), 4),
        "gamma": round(rng.uniform(0.82, 1.28), 4),
        "highlight_strength": round(rng.uniform(0.08, 0.22), 4),
        "highlight_center_x": round(center_x, 4),
        "highlight_center_y": round(center_y, 4),
        "highlight_radius_x": round(rng.uniform(width * 0.18, width * 0.45), 4),
        "highlight_radius_y": round(rng.uniform(height * 0.18, height * 0.45), 4),
        "height": height,
        "width": width,
    }


def make_very_dark_params(
    width: int,
    height: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    return {
        "kind": "very_dark",
        "brightness_shift": round(rng.uniform(-92.0, -52.0), 4),
        "contrast": round(rng.uniform(0.8, 0.98), 4),
        "gamma": round(rng.uniform(1.35, 1.9), 4),
        "saturation": round(rng.uniform(0.62, 0.9), 4),
        "noise_sigma": round(rng.uniform(10.0, 28.0), 4),
        "height": height,
        "width": width,
    }


def make_very_bright_params(
    width: int,
    height: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    center_x = rng.uniform(width * 0.2, width * 0.8)
    center_y = rng.uniform(height * 0.15, height * 0.85)
    return {
        "kind": "very_bright",
        "brightness_shift": round(rng.uniform(38.0, 88.0), 4),
        "contrast": round(rng.uniform(0.88, 1.06), 4),
        "gamma": round(rng.uniform(0.52, 0.82), 4),
        "saturation": round(rng.uniform(0.9, 1.08), 4),
        "highlight_strength": round(rng.uniform(0.16, 0.34), 4),
        "highlight_center_x": round(center_x, 4),
        "highlight_center_y": round(center_y, 4),
        "highlight_radius_x": round(rng.uniform(width * 0.2, width * 0.48), 4),
        "highlight_radius_y": round(rng.uniform(height * 0.2, height * 0.48), 4),
        "height": height,
        "width": width,
    }


def make_motion_blur_params(
    width: int,
    height: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    blur_length = rng.randrange(9, 28, 2)
    return {
        "kind": "motion_blur",
        "blur_length": blur_length,
        "angle_deg": round(rng.uniform(0.0, 180.0), 4),
        "brightness_shift": round(rng.uniform(-20.0, 8.0), 4),
        "contrast": round(rng.uniform(0.88, 1.04), 4),
        "gamma": round(rng.uniform(0.92, 1.18), 4),
        "noise_sigma": round(rng.uniform(1.5, 7.5), 4),
        "height": height,
        "width": width,
    }


AUGMENTATION_BUILDERS: dict[str, Any] = {
    "noise_photometric": make_noise_photometric_params,
    "blur_exposure": make_blur_exposure_params,
    "very_dark": make_very_dark_params,
    "very_bright": make_very_bright_params,
    "motion_blur": make_motion_blur_params,
}


def build_variant_params(
    image_id: int,
    width: int,
    height: int,
    variants_per_image: int,
    seed: int,
) -> list[dict[str, Any]]:
    if variants_per_image > len(AUGMENTATION_BUILDERS):
        raise ValueError(
            f"variants-per-image={variants_per_image} exceeds available augmentation types={len(AUGMENTATION_BUILDERS)}."
        )
    chooser = random.Random(seed + image_id * 1009 + width * 17 + height * 31)
    selected_kinds = chooser.sample(list(AUGMENTATION_BUILDERS), k=variants_per_image)
    params = []
    for variant_index, kind in enumerate(selected_kinds):
        local_seed = (
            seed
            + image_id * 1009
            + variant_index * 9173
            + sum(ord(char) for char in kind)
        )
        entry = AUGMENTATION_BUILDERS[kind](width, height, local_seed)
        entry["variant_index"] = variant_index
        entry["variant_suffix"] = chr(ord("A") + variant_index)
        params.append(entry)
    return params


def apply_noise_photometric(
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


def apply_blur_exposure(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    working = cv2.GaussianBlur(
        image,
        (int(params["kernel_size"]), int(params["kernel_size"])),
        sigmaX=float(params["blur_sigma"]),
        sigmaY=float(params["blur_sigma"]),
    ).astype(np.float32)
    working = apply_gamma(working, float(params["gamma"]))
    working = (working - 127.5) * float(params["contrast"]) + 127.5
    working = working + float(params["brightness_shift"])
    working = add_soft_highlight(
        working,
        center_x=float(params["highlight_center_x"]),
        center_y=float(params["highlight_center_y"]),
        radius_x=float(params["highlight_radius_x"]),
        radius_y=float(params["highlight_radius_y"]),
        strength=float(params["highlight_strength"]),
    )
    return clamp_image(working)


def apply_very_dark(
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


def apply_very_bright(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
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


def apply_variant(
    image: np.ndarray,
    params: dict[str, Any],
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if params["kind"] == "noise_photometric":
        return apply_noise_photometric(image, params, rng)
    if params["kind"] == "blur_exposure":
        return apply_blur_exposure(image, params)
    if params["kind"] == "very_dark":
        return apply_very_dark(image, params, rng)
    if params["kind"] == "very_bright":
        return apply_very_bright(image, params)
    if params["kind"] == "motion_blur":
        return apply_motion_blur(image, params, rng)
    raise ValueError(f"Unsupported variant kind: {params['kind']}")


def compute_variant_image_id(
    image_index: int,
    variant_index: int,
    first_variant_image_id: int,
    variants_per_image: int,
) -> int:
    return first_variant_image_id + image_index * variants_per_image + variant_index


def process_item(
    item_index: int,
    item: dict[str, Any],
    *,
    output_image_dir: Path,
    variants_per_image: int,
    seed: int,
    png_compression: int,
    first_variant_image_id: int,
) -> dict[str, Any]:
    output_original = output_image_dir / item["file_name"]
    link_mode = ensure_original_in_output(item["source_path"], output_original)

    source_image = load_image_rgb(item["source_path"])
    variant_entries = []
    for params in build_variant_params(
        image_id=item["image_id"],
        width=item["width"],
        height=item["height"],
        variants_per_image=variants_per_image,
        seed=seed,
    ):
        variant_index = int(params["variant_index"])
        suffix = params["variant_suffix"]
        stem = Path(item["file_name"]).stem
        variant_name = f"{stem}_aug{suffix}.png"
        variant_path = output_image_dir / variant_name
        variant_seed = seed + item["image_id"] * 1009 + variant_index * 9173
        if not variant_path.exists():
            variant_image = apply_variant(source_image, params, variant_seed)
            save_image_rgb(variant_path, variant_image, png_compression)
        variant_entries.append(
            {
                "image_id": compute_variant_image_id(
                    image_index=item_index,
                    variant_index=variant_index,
                    first_variant_image_id=first_variant_image_id,
                    variants_per_image=variants_per_image,
                ),
                "file_name": variant_name,
                "params": params,
            }
        )

    return {
        "item_index": item_index,
        "source_image_id": item["image_id"],
        "source_file_name": item["file_name"],
        "width": item["width"],
        "height": item["height"],
        "captions": item["captions"],
        "output_original_mode": link_mode,
        "variants": variant_entries,
    }


def build_outputs(
    items: list[dict[str, Any]],
    output_image_dir: Path,
    variants_per_image: int,
    seed: int,
    workers: int,
    png_compression: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    images = []
    annotations = []
    manifest_items = []
    annotation_id = 1
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
                variants_per_image=variants_per_image,
                seed=seed,
                png_compression=png_compression,
                first_variant_image_id=first_variant_image_id,
            ): item_index
            for item_index, item in enumerate(items)
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Augmenting images",
        ):
            result = future.result()
            results[result["item_index"]] = result

    for item_index, item in enumerate(items):
        result = results[item_index]
        if result is None:
            raise RuntimeError(
                f"Missing augmentation result for image {item['image_id']}"
            )

        images.append(
            {
                "id": item["image_id"],
                "file_name": item["file_name"],
                "width": item["width"],
                "height": item["height"],
            }
        )
        for caption in item["captions"]:
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": item["image_id"],
                    "caption": caption,
                    "is_rejected": False,
                    "is_precanned": False,
                }
            )
            annotation_id += 1

        for variant in result["variants"]:
            images.append(
                {
                    "id": variant["image_id"],
                    "file_name": variant["file_name"],
                    "width": item["width"],
                    "height": item["height"],
                }
            )
            for caption in item["captions"]:
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": variant["image_id"],
                        "caption": caption,
                        "is_rejected": False,
                        "is_precanned": False,
                    }
                )
                annotation_id += 1

        manifest_items.append(
            {
                "source_image_id": item["image_id"],
                "source_file_name": item["file_name"],
                "output_original_mode": result["output_original_mode"],
                "captions": item["captions"],
                "variants": result["variants"],
            }
        )

    annotation = {"images": images, "annotations": annotations}
    manifest = {
        "variants_per_image": variants_per_image,
        "workers": workers,
        "png_compression": png_compression,
        "source_images": len(items),
        "total_images": len(images),
        "total_annotations": len(annotations),
        "items": manifest_items,
    }
    return annotation, manifest


def main() -> None:
    args = parse_args()
    if args.variants_per_image < 1:
        raise ValueError("variants-per-image must be at least 1.")
    if args.workers < 1:
        raise ValueError("workers must be at least 1.")
    if not 0 <= args.png_compression <= 9:
        raise ValueError("png-compression must be between 0 and 9.")

    if args.overwrite and args.output_image_dir.exists():
        shutil.rmtree(args.output_image_dir)
    if args.overwrite and args.output_ann.exists():
        args.output_ann.unlink()
    if args.overwrite and args.manifest_file.exists():
        args.manifest_file.unlink()

    dataset = load_dataset(args.annotation_file)
    items = build_items(dataset, args.source_image_dir, args.limit)
    args.output_image_dir.mkdir(parents=True, exist_ok=True)

    annotation, manifest = build_outputs(
        items=items,
        output_image_dir=args.output_image_dir,
        variants_per_image=args.variants_per_image,
        seed=args.seed,
        workers=args.workers,
        png_compression=args.png_compression,
    )

    expected_images = len(items) * (1 + args.variants_per_image)
    if len(annotation["images"]) != expected_images:
        raise RuntimeError(
            f"Expected {expected_images} images in output annotation, found {len(annotation['images'])}."
        )
    if not annotation["annotations"]:
        raise RuntimeError("Output annotation has no captions.")

    args.output_ann.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_ann.write_text(json.dumps(annotation, indent=2), encoding="utf-8")
    args.manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved augmented image directory to: {args.output_image_dir}")
    print(f"Saved merged annotations to: {args.output_ann}")
    print(f"Saved augmentation manifest to: {args.manifest_file}")


if __name__ == "__main__":
    main()

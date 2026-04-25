from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import re
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from tqdm import tqdm

DEFAULT_API_BASE = "http://158.109.8.111:8000/v1"
DEFAULT_ANNOTATION_FILE = Path("outputs/synthetic_data/annotations/synthetic.json")
DEFAULT_IMAGE_DIR = Path("outputs/synthetic_data/images")
DEFAULT_OUTPUT_ANN = Path(
    "outputs/synthetic_data/annotations/synthetic_multicap_qwen36.json"
)
DEFAULT_REPORT_FILE = Path(
    "outputs/synthetic_data/annotations/synthetic_multicap_qwen36_report.json"
)
DEFAULT_CACHE_FILE = Path(
    "outputs/synthetic_data/annotations/synthetic_multicap_qwen36_cache.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 4 additional captions per synthetic image using a vision LLM."
    )
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=DEFAULT_ANNOTATION_FILE,
        help="Input VizWiz-style synthetic annotation file.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directory containing source synthetic images.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=DEFAULT_API_BASE,
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model id. If omitted, the first served model is used.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY"),
        help="Optional API key. Defaults to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--target-captions",
        type=int,
        default=5,
        help="Total captions to keep per image.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for caption generation.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling value.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=220,
        help="Maximum completion tokens.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum API attempts per image.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Maximum number of in-flight caption requests.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=25,
        help="Append cache rows and refresh the report every N completed images.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for small dry runs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed recorded in the report.",
    )
    parser.add_argument(
        "--output-ann",
        type=Path,
        default=DEFAULT_OUTPUT_ANN,
        help="Output VizWiz-style annotation file.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=DEFAULT_REPORT_FILE,
        help="Output JSON report file.",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=DEFAULT_CACHE_FILE,
        help="Cache file used for resume support.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite cache and outputs from scratch.",
    )
    return parser.parse_args()


def normalize_caption(text: str) -> str:
    text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    text = text.strip(" \t\r\n-*")
    if not text:
        return ""
    if text[0].islower():
        text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


def caption_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def is_caption_valid(text: str) -> bool:
    if not text:
        return False
    words = caption_key(text).split()
    return 3 <= len(words) <= 20 and 12 <= len(text) <= 140


def is_diverse(candidate: str, existing: list[str], threshold: float = 0.88) -> bool:
    candidate_key = caption_key(candidate)
    if not candidate_key:
        return False
    for current in existing:
        current_key = caption_key(current)
        if candidate_key == current_key:
            return False
        similarity = SequenceMatcher(None, candidate_key, current_key).ratio()
        if similarity >= threshold:
            return False
        if candidate_key in current_key or current_key in candidate_key:
            return False
    return True


def encode_image_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_client(api_base: str, api_key: str | None, timeout: int) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=api_base.rstrip("/"),
        api_key=api_key or "EMPTY",
        timeout=timeout,
        max_retries=0,
    )


async def discover_model(client: AsyncOpenAI) -> str:
    data = (await client.models.list()).data
    if not data:
        raise RuntimeError("No models returned by /models.")
    model_id = data[0].id
    if not model_id:
        raise RuntimeError("Could not resolve model id from /models response.")
    return model_id


def parse_captions_from_response(content: str | None) -> list[str]:
    if not content:
        return []
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    captions = payload.get("captions", []) if isinstance(payload, dict) else []
    if not isinstance(captions, list):
        return []
    return [str(item) for item in captions]


def load_vizwiz_annotations(annotation_file: Path) -> dict[str, Any]:
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
    image_dir: Path,
    limit: int | None,
) -> list[dict[str, Any]]:
    image_map = {int(image["id"]): image for image in dataset["images"]}
    captions_by_image: dict[int, list[str]] = defaultdict(list)
    for annotation in dataset["annotations"]:
        image_id = int(annotation["image_id"])
        caption = normalize_caption(str(annotation.get("caption", "")))
        if caption:
            captions_by_image[image_id].append(caption)

    items: list[dict[str, Any]] = []
    for image_id in sorted(image_map):
        image = image_map[image_id]
        file_name = str(image["file_name"])
        image_path = image_dir / file_name
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image file: {image_path}")
        captions = []
        for caption in captions_by_image.get(image_id, []):
            if is_diverse(caption, captions, threshold=0.995):
                captions.append(caption)
        if not captions:
            continue
        items.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "width": int(image.get("width", 0)),
                "height": int(image.get("height", 0)),
                "image_path": image_path,
                "captions": captions,
            }
        )
    if limit is not None:
        return items[:limit]
    return items


def build_prompt(existing_captions: list[str], missing_count: int) -> str:
    existing_block = "\n".join(f"- {caption}" for caption in existing_captions)
    return (
        "You are writing extra reference captions for a VizWiz-style image captioning dataset.\n"
        f"Return exactly {missing_count} short captions as JSON with this shape: "
        '{"captions": ["...", "..."]}.\n'
        "Rules:\n"
        "- Each caption must be a short literal sentence.\n"
        "- Mention only what is visible in the image.\n"
        "- Keep each caption different from the existing ones and from each other.\n"
        "- Avoid lists, explanations, uncertainty, OCR guesses, and extra fields.\n"
        "- Stay concise, similar to VizWiz references.\n"
        "Existing captions to avoid copying:\n"
        f"{existing_block}"
    )


def generate_fallback_captions(
    existing_captions: list[str],
    missing_count: int,
) -> list[str]:
    base = existing_captions[0].rstrip(".!?")
    lowered = base[0].lower() + base[1:] if base else base
    templates = [
        f"A photo of {lowered}.",
        f"This image shows {lowered}.",
        f"There is {lowered}.",
        f"The picture shows {lowered}.",
        f"A close view of {lowered}.",
        f"A brief view of {lowered}.",
        f"The scene contains {lowered}.",
        f"Seen here is {lowered}.",
        f"A close-up image of {lowered}.",
        f"Visible here is {lowered}.",
    ]
    results: list[str] = []
    used_keys = {caption_key(caption) for caption in existing_captions}
    for candidate in templates:
        candidate = normalize_caption(candidate)
        candidate_key = caption_key(candidate)
        if (
            is_caption_valid(candidate)
            and candidate_key not in used_keys
            and is_diverse(candidate, existing_captions + results, threshold=0.96)
        ):
            results.append(candidate)
            used_keys.add(candidate_key)
        if len(results) >= missing_count:
            break

    if len(results) < missing_count:
        openers = [
            "A photo showing",
            "This image contains",
            "The picture includes",
            "In the image is",
            "Shown here is",
            "The scene shows",
            "A close-up of",
            "Visible in the image is",
        ]
        endings = [
            "",
            " in view",
            " shown here",
            " in the scene",
            " on display",
            " visible here",
        ]
        for opener in openers:
            for ending in endings:
                candidate = normalize_caption(f"{opener} {lowered}{ending}.")
                candidate_key = caption_key(candidate)
                if candidate_key in used_keys:
                    continue
                if is_caption_valid(candidate):
                    results.append(candidate)
                    used_keys.add(candidate_key)
                if len(results) >= missing_count:
                    break
            if len(results) >= missing_count:
                break
    return results[:missing_count]


async def request_caption_candidates(
    client: AsyncOpenAI,
    *,
    model: str,
    image_data_url: str,
    existing_captions: list[str],
    missing_count: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> list[str]:
    completion = await client.chat.completions.create(
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        extra_body={
            "chat_template_kwargs": {
                "enable_reasoning": False,
                "enable_thinking": False,
            },
            "top_p": 1.0,
            "top_k": 50,
            "min_p": 0.05,
            "presence_penalty": 0.7,
            "repetition_penalty": 1.05,
            "frequency_penalty": 0.4,
        },
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Return JSON only."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": build_prompt(existing_captions, missing_count),
                    },
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    )
    if not completion.choices:
        return []
    return parse_captions_from_response(completion.choices[0].message.content)


async def request_additional_captions(
    *,
    client: AsyncOpenAI,
    model: str,
    image_data_url: str,
    existing_captions: list[str],
    target_captions: int,
    max_retries: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> tuple[list[str], int, bool, dict[str, int]]:
    captions = list(existing_captions)
    retries_used = 0
    used_fallback = False
    stats: dict[str, int] = defaultdict(int)

    for attempt in range(max_retries + 1):
        missing_count = target_captions - len(captions)
        if missing_count <= 0:
            break
        try:
            candidates = await request_caption_candidates(
                client,
                model=model,
                image_data_url=image_data_url,
                existing_captions=captions,
                missing_count=missing_count,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        except (APIConnectionError, APITimeoutError):
            stats["connection_errors"] += 1
            if attempt >= max_retries:
                break
            retries_used += 1
            await asyncio.sleep(min(10, 2**attempt))
            continue
        except RateLimitError:
            stats["rate_limit_errors"] += 1
            if attempt >= max_retries:
                break
            retries_used += 1
            await asyncio.sleep(min(10, 2**attempt))
            continue
        except APIStatusError:
            stats["api_status_errors"] += 1
            if attempt >= max_retries:
                break
            retries_used += 1
            await asyncio.sleep(min(10, 2**attempt))
            continue

        stats["api_calls"] += 1
        if not candidates:
            stats["empty_generations"] += 1
            if attempt >= max_retries:
                continue
            retries_used += 1
            continue

        for candidate in candidates:
            candidate = normalize_caption(candidate)
            if not is_caption_valid(candidate):
                stats["invalid_candidates"] += 1
                continue
            if not is_diverse(candidate, captions):
                stats["duplicate_candidates"] += 1
                continue
            captions.append(candidate)
            if len(captions) >= target_captions:
                break

    missing_count = target_captions - len(captions)
    if missing_count > 0:
        used_fallback = True
        fallback = generate_fallback_captions(captions, missing_count)
        stats["fallback_captions"] += len(fallback)
        captions.extend(fallback)

    if len(captions) < target_captions:
        raise RuntimeError(
            f"Could not build {target_captions} captions for image after retries and fallback."
        )
    return captions[:target_captions], retries_used, used_fallback, dict(stats)


def append_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_cache(cache_file: Path) -> dict[int, dict[str, Any]]:
    if not cache_file.is_file():
        return {}
    rows: dict[int, dict[str, Any]] = {}
    with cache_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[int(row["image_id"])] = row
    return rows


def build_output_annotation(
    items: list[dict[str, Any]],
    cache: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    images = []
    annotations = []
    annotation_id = 1
    for item in items:
        cached = cache.get(item["image_id"])
        if not cached:
            continue
        images.append(
            {
                "id": item["image_id"],
                "file_name": item["file_name"],
                "width": item["width"],
                "height": item["height"],
            }
        )
        for caption in cached["captions"]:
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
    return {"images": images, "annotations": annotations}


def merge_stats(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] += value


def write_report(
    output_report: Path,
    *,
    annotation_file: Path,
    image_dir: Path,
    output_ann: Path,
    cache_file: Path,
    api_base: str,
    model: str,
    seed: int,
    concurrency: int,
    target_captions: int,
    items: list[dict[str, Any]],
    cache: dict[int, dict[str, Any]],
    stats: dict[str, int],
    started_at: float,
    failures: list[dict[str, Any]],
) -> None:
    output_report.parent.mkdir(parents=True, exist_ok=True)
    processed_images = sum(1 for item in items if item["image_id"] in cache)
    report = {
        "annotation_file": str(annotation_file),
        "image_dir": str(image_dir),
        "output_annotation_file": str(output_ann),
        "cache_file": str(cache_file),
        "api_base": api_base,
        "model": model,
        "seed": seed,
        "concurrency": concurrency,
        "target_captions_per_image": target_captions,
        "requested_images": len(items),
        "processed_images": processed_images,
        "runtime_seconds": round(time.time() - started_at, 3),
        "stats": stats,
        "failures": failures,
    }
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")


async def process_item(
    client: AsyncOpenAI,
    item: dict[str, Any],
    *,
    model: str,
    target_captions: int,
    max_retries: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> tuple[int, dict[str, Any], dict[str, int]]:
    image_data_url = await asyncio.to_thread(encode_image_to_data_url, item["image_path"])
    captions, retries_used, used_fallback, stats = await request_additional_captions(
        client=client,
        model=model,
        image_data_url=image_data_url,
        existing_captions=item["captions"],
        target_captions=target_captions,
        max_retries=max_retries,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    stats = defaultdict(int, stats)
    stats["processed_images"] += 1
    stats["retry_images"] += int(retries_used > 0)
    stats["fallback_images"] += int(used_fallback)
    row = {
        "image_id": item["image_id"],
        "file_name": item["file_name"],
        "captions": captions,
        "source_captions": item["captions"],
        "retries_used": retries_used,
        "used_fallback": used_fallback,
    }
    return item["image_id"], row, stats


def flush_checkpoint(
    *,
    cache_file: Path,
    pending_rows: list[dict[str, Any]],
    output_report: Path,
    annotation_file: Path,
    image_dir: Path,
    output_ann: Path,
    api_base: str,
    model: str,
    seed: int,
    concurrency: int,
    target_captions: int,
    items: list[dict[str, Any]],
    cache: dict[int, dict[str, Any]],
    stats: dict[str, int],
    started_at: float,
    failures: list[dict[str, Any]],
) -> None:
    append_jsonl_rows(cache_file, pending_rows)
    pending_rows.clear()
    write_report(
        output_report,
        annotation_file=annotation_file,
        image_dir=image_dir,
        output_ann=output_ann,
        cache_file=cache_file,
        api_base=api_base,
        model=model,
        seed=seed,
        concurrency=concurrency,
        target_captions=target_captions,
        items=items,
        cache=cache,
        stats=dict(stats),
        started_at=started_at,
        failures=failures,
    )


async def run_generation(args: argparse.Namespace) -> None:
    if args.target_captions < 2:
        raise ValueError("target-captions must be at least 2.")
    if args.flush_every < 1:
        raise ValueError("flush-every must be at least 1.")
    if args.concurrency < 1:
        raise ValueError("concurrency must be at least 1.")

    if args.overwrite and args.cache_file.exists():
        args.cache_file.unlink()

    dataset = load_vizwiz_annotations(args.annotation_file)
    items = build_items(dataset, args.image_dir, args.limit)
    cache = load_cache(args.cache_file)
    pending_items = [item for item in items if item["image_id"] not in cache]

    client = build_client(args.api_base, args.api_key, args.timeout)
    try:
        model = args.model or await discover_model(client)
        print(f"Resolved model: {model}")
        print(f"Images selected: {len(items)}")
        print(f"Pending images: {len(pending_items)} | concurrency={args.concurrency}")

        stats: dict[str, int] = defaultdict(int)
        started_at = time.time()
        pending_rows: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []

        progress = tqdm(total=len(pending_items), desc="Generating captions")
        next_index = 0
        in_flight: dict[asyncio.Task, dict[str, Any]] = {}

        while next_index < len(pending_items) and len(in_flight) < args.concurrency:
            item = pending_items[next_index]
            next_index += 1
            task = asyncio.create_task(
                process_item(
                    client,
                    item,
                    model=model,
                    target_captions=args.target_captions,
                    max_retries=args.max_retries,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                )
            )
            in_flight[task] = item

        while in_flight:
            done, _ = await asyncio.wait(
                in_flight.keys(), return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                item = in_flight.pop(task)
                try:
                    image_id, row, local_stats = task.result()
                except Exception as exc:
                    failures.append(
                        {
                            "image_id": item["image_id"],
                            "file_name": item["file_name"],
                            "error": str(exc),
                        }
                    )
                    stats["failed_images"] += 1
                else:
                    cache[image_id] = row
                    pending_rows.append(row)
                    merge_stats(stats, local_stats)
                    if len(pending_rows) >= args.flush_every:
                        flush_checkpoint(
                            cache_file=args.cache_file,
                            pending_rows=pending_rows,
                            output_report=args.output_report,
                            annotation_file=args.annotation_file,
                            image_dir=args.image_dir,
                            output_ann=args.output_ann,
                            api_base=args.api_base,
                            model=model,
                            seed=args.seed,
                            concurrency=args.concurrency,
                            target_captions=args.target_captions,
                            items=items,
                            cache=cache,
                            stats=stats,
                            started_at=started_at,
                            failures=failures,
                        )
                progress.update(1)
                if next_index < len(pending_items):
                    next_item = pending_items[next_index]
                    next_index += 1
                    next_task = asyncio.create_task(
                        process_item(
                            client,
                            next_item,
                            model=model,
                            target_captions=args.target_captions,
                            max_retries=args.max_retries,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            max_tokens=args.max_tokens,
                        )
                    )
                    in_flight[next_task] = next_item

        progress.close()

        flush_checkpoint(
            cache_file=args.cache_file,
            pending_rows=pending_rows,
            output_report=args.output_report,
            annotation_file=args.annotation_file,
            image_dir=args.image_dir,
            output_ann=args.output_ann,
            api_base=args.api_base,
            model=model,
            seed=args.seed,
            concurrency=args.concurrency,
            target_captions=args.target_captions,
            items=items,
            cache=cache,
            stats=stats,
            started_at=started_at,
            failures=failures,
        )

        output_annotation = build_output_annotation(items, cache)
        if len(output_annotation["images"]) != len(items):
            missing = len(items) - len(output_annotation["images"])
            raise RuntimeError(
                f"Output annotation is missing {missing} processed images."
            )
        expected_annotations = len(items) * args.target_captions
        if len(output_annotation["annotations"]) != expected_annotations:
            raise RuntimeError(
                f"Expected {expected_annotations} annotations, found {len(output_annotation['annotations'])}."
            )
        args.output_ann.parent.mkdir(parents=True, exist_ok=True)
        args.output_ann.write_text(
            json.dumps(output_annotation, indent=2), encoding="utf-8"
        )
        write_report(
            args.output_report,
            annotation_file=args.annotation_file,
            image_dir=args.image_dir,
            output_ann=args.output_ann,
            cache_file=args.cache_file,
            api_base=args.api_base,
            model=model,
            seed=args.seed,
            concurrency=args.concurrency,
            target_captions=args.target_captions,
            items=items,
            cache=cache,
            stats=dict(stats),
            started_at=started_at,
            failures=failures,
        )
        print(f"Saved multi-caption annotations to: {args.output_ann}")
        print(f"Saved run report to: {args.output_report}")
    finally:
        await client.close()


def main() -> None:
    asyncio.run(run_generation(parse_args()))


if __name__ == "__main__":
    main()

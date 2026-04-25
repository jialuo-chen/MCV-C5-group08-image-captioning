from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision.transforms import v2

from src.data.tokenizer import BaseTokenizer
from src.data.vizwiz import VizWiz


def get_image_transform(split: str = "train", image_size: int = 224):
    """Return image transform pipeline.

    Training adds random horizontal flip; val/test only resizes + normalizes.
    """
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    if split == "train":
        return v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize((image_size, image_size), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                normalize,
            ]
        )
    else:
        return v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize((image_size, image_size), antialias=True),
                normalize,
            ]
        )


class VizWizCaptionDataset(Dataset):
    """VizWiz image-captioning dataset.

    Each item returns ``(image_tensor, caption_ids, caption_text, image_path)``.
    During training, one caption is sampled randomly per image.
    During evaluation, all captions are available via ``get_all_captions(idx)``.

    Parameters
    ----------
    annotation_file : str | Path
        Path to the VizWiz annotation JSON.
    image_dir : str | Path
        Path to the image directory.
    tokenizer : BaseTokenizer
        Tokenizer for encoding captions.
    max_length : int
        Maximum caption length (in tokens, including SOS/EOS).
    split : str
        ``"train"`` or ``"val"`` — controls image augmentation.
    indices : list[int] | None
        If given, restrict to these image indices (for train/val splitting).
    """

    def __init__(
        self,
        annotation_file: str | Path,
        image_dir: str | Path,
        tokenizer: BaseTokenizer,
        max_length: int = 201,
        split: str = "train",
        indices: list[int] | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.transform = get_image_transform(split)

        vizwiz = VizWiz(str(annotation_file), ignore_rejected=True, ignore_precanned=True)
        self._build_image_caption_pairs(vizwiz, indices)

    def _build_image_caption_pairs(
        self, vizwiz: VizWiz, indices: list[int] | None
    ) -> None:
        """Group captions by image and store as list of dicts."""
        all_img_ids = sorted(vizwiz.imgs.keys())
        if indices is not None:
            all_img_ids = [all_img_ids[i] for i in indices if i < len(all_img_ids)]

        self.items: list[dict] = []
        for img_id in all_img_ids:
            img_info = vizwiz.imgs[img_id]
            anns = vizwiz.imgToAnns.get(img_id, [])
            captions = [ann["caption"] for ann in anns if ann.get("caption")]
            if not captions:
                continue
            self.items.append(
                {
                    "image_id": img_id,
                    "file_name": img_info["file_name"],
                    "captions": captions,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        img_path = self.image_dir / item["file_name"]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        if self.split == "train":
            caption_text = random.choice(item["captions"])
        else:
            caption_text = item["captions"][0]

        cap_ids = self.tokenizer.encode(caption_text)
        cap_ids = self.tokenizer.pad_sequence(cap_ids, self.max_length)
        cap_tensor = torch.tensor(cap_ids, dtype=torch.long)

        return img, cap_tensor, caption_text, str(img_path)

    def get_all_captions(self, idx: int) -> list[str]:
        """Return all reference captions for the image at *idx*."""
        return self.items[idx]["captions"]

    def get_all_captions_by_image_id(self, image_id: int) -> list[str]:
        """Return all reference captions for a given *image_id*."""
        for item in self.items:
            if item["image_id"] == image_id:
                return item["captions"]
        return []


def caption_collate_fn(batch):
    """Custom collate: stack images and caption tensors, collect texts and paths."""
    images, cap_tensors, cap_texts, img_paths = zip(*batch)
    images = torch.stack(images, dim=0)
    cap_tensors = torch.stack(cap_tensors, dim=0)
    return images, cap_tensors, list(cap_texts), list(img_paths)


def build_datasets(cfg, tokenizer: BaseTokenizer):
    """Build train, validation, and test datasets from config.

    - train + val: come from the training annotation file (90/10 split).
    - test: comes from the val annotation file (full set).

    Returns
    -------
    tuple[VizWizCaptionDataset, VizWizCaptionDataset, VizWizCaptionDataset]
        (train_dataset, val_dataset, test_dataset)
    """
    root = Path(cfg.dataset.root)
    train_ann = str(root / cfg.dataset.train_ann)
    val_ann = str(root / cfg.dataset.val_ann)
    train_img_dir = str(root / cfg.dataset.train_img_dir)
    val_img_dir = str(root / cfg.dataset.val_img_dir)
    max_length = cfg.tokenizer.max_length

    vizwiz_train = VizWiz(train_ann, ignore_rejected=True, ignore_precanned=True)
    all_img_ids = sorted(vizwiz_train.imgs.keys())
    valid_indices = []
    for i, img_id in enumerate(all_img_ids):
        anns = vizwiz_train.imgToAnns.get(img_id, [])
        captions = [a["caption"] for a in anns if a.get("caption")]
        if captions:
            valid_indices.append(i)

    rng = random.Random(cfg.seed)
    rng.shuffle(valid_indices)
    val_size = int(len(valid_indices) * cfg.dataset.val_split_ratio)
    val_indices = sorted(valid_indices[:val_size])
    train_indices = sorted(valid_indices[val_size:])

    train_ds = VizWizCaptionDataset(
        annotation_file=train_ann,
        image_dir=train_img_dir,
        tokenizer=tokenizer,
        max_length=max_length,
        split="train",
        indices=train_indices,
    )
    val_ds = VizWizCaptionDataset(
        annotation_file=train_ann,
        image_dir=train_img_dir,
        tokenizer=tokenizer,
        max_length=max_length,
        split="val",
        indices=val_indices,
    )
    test_ds = VizWizCaptionDataset(
        annotation_file=val_ann,
        image_dir=val_img_dir,
        tokenizer=tokenizer,
        max_length=max_length,
        split="val",
        indices=None,
    )
    return train_ds, val_ds, test_ds


class VizWizEvalDataset(Dataset):
    """VizWiz dataset for evaluating pretrained / multimodal models.

    Returns ``(PIL.Image, list[str], str)`` — image, reference captions, path.
    No tokenization or tensor transforms are applied (models bring their own
    processors).
    """

    def __init__(
        self,
        annotation_file: str | Path,
        image_dir: str | Path,
    ) -> None:
        self.image_dir = Path(image_dir)
        vizwiz = VizWiz(str(annotation_file), ignore_rejected=True, ignore_precanned=True)
        self.items: list[dict] = []
        for img_id in sorted(vizwiz.imgs.keys()):
            img_info = vizwiz.imgs[img_id]
            anns = vizwiz.imgToAnns.get(img_id, [])
            captions = [ann["caption"] for ann in anns if ann.get("caption")]
            if not captions:
                continue
            self.items.append(
                {
                    "file_name": img_info["file_name"],
                    "captions": captions,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        img_path = self.image_dir / item["file_name"]
        img = Image.open(img_path).convert("RGB")
        return img, item["captions"], str(img_path)

    def get_all_captions(self, idx: int) -> list[str]:
        return self.items[idx]["captions"]


class VizWizVisionDataset(Dataset):
    """VizWiz dataset that applies a HuggingFace image processor and tokenizer.

    Returns dict with ``pixel_values``, ``input_ids``, ``attention_mask``,
    ``labels``, ``caption_text``, ``image_path``.
    """

    def __init__(
        self,
        annotation_file: str | Path,
        image_dir: str | Path,
        image_processor,
        tokenizer,
        max_length: int = 128,
        split: str = "train",
        indices: list[int] | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split

        vizwiz = VizWiz(str(annotation_file), ignore_rejected=True, ignore_precanned=True)
        all_img_ids = sorted(vizwiz.imgs.keys())
        if indices is not None:
            all_img_ids = [all_img_ids[i] for i in indices if i < len(all_img_ids)]

        self.items: list[dict] = []
        for img_id in all_img_ids:
            img_info = vizwiz.imgs[img_id]
            anns = vizwiz.imgToAnns.get(img_id, [])
            captions = [ann["caption"] for ann in anns if ann.get("caption")]
            if not captions:
                continue
            self.items.append(
                {
                    "image_id": img_id,
                    "file_name": img_info["file_name"],
                    "captions": captions,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        img_path = self.image_dir / item["file_name"]
        img = Image.open(img_path).convert("RGB")

        pixel_values = self.image_processor(
            img, return_tensors="pt"
        ).pixel_values.squeeze(0)

        if self.split == "train":
            caption_text = random.choice(item["captions"])
        else:
            caption_text = item["captions"][0]

        # Tokenize + append EOS so the model learns when to stop generating.
        # Qwen3.5 tokenizer has add_eos_token=False by default.
        token_ids = self.tokenizer.encode(caption_text, add_special_tokens=False)
        token_ids.append(self.tokenizer.eos_token_id)
        token_ids = token_ids[: self.max_length]

        # RIGHT-pad: caption tokens first, then padding.  This is critical
        # because the visual prefix is prepended in the model's forward().
        # With right-padding the last visual token directly precedes the first
        # caption token — matching generation where there is no padding.
        pad_len = self.max_length - len(token_ids)
        input_ids = torch.tensor(
            token_ids + [self.tokenizer.pad_token_id] * pad_len, dtype=torch.long
        )
        attention_mask = torch.tensor(
            [1] * len(token_ids) + [0] * pad_len, dtype=torch.long
        )
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "caption_text": caption_text,
            "image_path": str(img_path),
        }

    def get_all_captions(self, idx: int) -> list[str]:
        return self.items[idx]["captions"]


def vision_collate_fn(batch: list[dict]) -> dict:
    """Collate for VizWizVisionDataset — stacks tensors, collects strings."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    caption_texts = [b["caption_text"] for b in batch]
    image_paths = [b["image_path"] for b in batch]
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "caption_texts": caption_texts,
        "image_paths": image_paths,
    }


def build_vision_datasets(cfg, image_processor, tokenizer):
    """Build train/val/test datasets for VisionEncoderDecoder / LoRA training.

    If ``cfg.synthetic_data.enabled`` is set, a synthetic dataset is
    concatenated to the training split via ``ConcatDataset``.
    Val and test remain VizWiz-only for fair evaluation.

    Returns ``(train_ds, val_ds, test_ds)``.
    """
    root = Path(cfg.dataset.root)
    train_ann = str(root / cfg.dataset.train_ann)
    val_ann = str(root / cfg.dataset.val_ann)
    train_img_dir = str(root / cfg.dataset.train_img_dir)
    val_img_dir = str(root / cfg.dataset.val_img_dir)
    max_length = cfg.get("tokenizer", {}).get("max_length", 128)

    vizwiz_train = VizWiz(train_ann, ignore_rejected=True, ignore_precanned=True)
    all_img_ids = sorted(vizwiz_train.imgs.keys())
    valid_indices = []
    for i, img_id in enumerate(all_img_ids):
        anns = vizwiz_train.imgToAnns.get(img_id, [])
        captions = [a["caption"] for a in anns if a.get("caption")]
        if captions:
            valid_indices.append(i)

    rng = random.Random(cfg.seed)
    rng.shuffle(valid_indices)
    val_size = int(len(valid_indices) * cfg.dataset.val_split_ratio)
    val_indices = sorted(valid_indices[:val_size])
    train_indices = sorted(valid_indices[val_size:])

    train_ds = VizWizVisionDataset(
        annotation_file=train_ann,
        image_dir=train_img_dir,
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_length=max_length,
        split="train",
        indices=train_indices,
    )

    syn_cfg = cfg.get("synthetic_data", {})
    if isinstance(syn_cfg, dict):
        syn_mode = syn_cfg.get("mode")
        syn_enabled = syn_cfg.get("enabled", False)
        if isinstance(syn_mode, bool):
            syn_mode = "append" if syn_mode else "off"
        if syn_mode is None:
            syn_mode = "append" if syn_enabled else "off"
    else:
        syn_mode = "off"

    if syn_mode not in {"off", "append", "only"}:
        raise ValueError(
            f"Unknown synthetic_data.mode '{syn_mode}'. Use 'off', 'append', or 'only'."
        )

    if syn_mode != "off":
        syn_root = Path(syn_cfg["root"])
        syn_ann = str(syn_root / syn_cfg.get("annotation", "annotations/synthetic.json"))
        syn_img_dir = str(syn_root / syn_cfg.get("image_dir", "images"))
        syn_ds = VizWizVisionDataset(
            annotation_file=syn_ann,
            image_dir=syn_img_dir,
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_length=max_length,
            split="train",
            indices=None,
        )
        print(f"Synthetic data ({syn_mode}): {len(syn_ds)} samples from {syn_root}")
        if syn_mode == "append":
            train_ds = ConcatDataset([train_ds, syn_ds])
        elif syn_mode == "only":
            train_ds = syn_ds

    val_ds = VizWizVisionDataset(
        annotation_file=train_ann,
        image_dir=train_img_dir,
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_length=max_length,
        split="val",
        indices=val_indices,
    )
    test_ds = VizWizVisionDataset(
        annotation_file=val_ann,
        image_dir=val_img_dir,
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_length=max_length,
        split="val",
        indices=None,
    )
    return train_ds, val_ds, test_ds

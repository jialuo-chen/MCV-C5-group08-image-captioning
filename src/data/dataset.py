"""VizWiz Caption Dataset for PyTorch.

Loads VizWiz images + captions using the VizWiz API class.
Handles train/val splitting (90/10 of train set) and test (val set).
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from src.data.vizwiz import VizWiz
from src.data.tokenizer import BaseTokenizer


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------

def get_image_transform(split: str = "train", image_size: int = 224):
    """Return image transform pipeline.

    Training adds random horizontal flip; val/test only resizes + normalizes.
    """
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    if split == "train":
        return v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((image_size, image_size), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            normalize,
        ])
    else:
        return v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((image_size, image_size), antialias=True),
            normalize,
        ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

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

        # Load annotations using VizWiz API
        vizwiz = VizWiz(str(annotation_file), ignore_rejected=True, ignore_precanned=True)
        self._build_image_caption_pairs(vizwiz, indices)

    def _build_image_caption_pairs(self, vizwiz: VizWiz, indices: list[int] | None) -> None:
        """Group captions by image and store as list of dicts."""
        # Build list of (image_info, [captions])
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
            self.items.append({
                "image_id": img_id,
                "file_name": img_info["file_name"],
                "captions": captions,
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        # Load image
        img_path = self.image_dir / item["file_name"]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # Pick one caption (random for train, first for eval)
        if self.split == "train":
            caption_text = random.choice(item["captions"])
        else:
            caption_text = item["captions"][0]

        # Encode caption
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


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def caption_collate_fn(batch):
    """Custom collate: stack images and caption tensors, collect texts and paths."""
    images, cap_tensors, cap_texts, img_paths = zip(*batch)
    images = torch.stack(images, dim=0)
    cap_tensors = torch.stack(cap_tensors, dim=0)
    return images, cap_tensors, list(cap_texts), list(img_paths)


# ---------------------------------------------------------------------------
# Helpers for building train / val / test datasets
# ---------------------------------------------------------------------------

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

    # Count images in training set to compute split indices
    vizwiz_train = VizWiz(train_ann, ignore_rejected=True, ignore_precanned=True)
    all_img_ids = sorted(vizwiz_train.imgs.keys())
    # Only keep images that have at least one caption
    valid_indices = []
    for i, img_id in enumerate(all_img_ids):
        anns = vizwiz_train.imgToAnns.get(img_id, [])
        captions = [a["caption"] for a in anns if a.get("caption")]
        if captions:
            valid_indices.append(i)

    # Shuffle and split
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

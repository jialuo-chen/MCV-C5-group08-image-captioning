"""Image encoder registry.

Wraps HuggingFace vision models (ResNet, VGG) and provides a uniform interface
with two output modes:

- **pooled**: single feature vector ``(batch, feature_dim)``
- **spatial**: feature map ``(batch, H*W, feature_dim)`` for attention decoders
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


# ---------------------------------------------------------------------------
# Feature dimension lookup (fallback if not in config)
# ---------------------------------------------------------------------------

_FEATURE_DIMS: dict[str, int] = {
    "microsoft/resnet-18": 512,
    "microsoft/resnet-34": 512,
    "microsoft/resnet-50": 2048,
    "google/vgg-16": 512,
    "google/vgg-19": 512,
}


class ImageEncoder(nn.Module):
    """Wraps a HuggingFace vision backbone.

    Parameters
    ----------
    pretrained : str
        HuggingFace model id (e.g. ``"microsoft/resnet-50"``).
    feature_dim : int | None
        Output feature dimension. Auto-detected if *None*.
    freeze : bool
        Whether to freeze all backbone parameters.
    """

    def __init__(
        self,
        pretrained: str = "microsoft/resnet-18",
        feature_dim: int | None = None,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(pretrained)
        self.feature_dim = feature_dim or _FEATURE_DIMS.get(pretrained, 512)
        self._is_vgg = "vgg" in pretrained.lower()

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    # ----- forward ---------------------------------------------------------

    def forward(self, images: torch.Tensor, spatial: bool = False) -> torch.Tensor:
        """Encode images.

        Parameters
        ----------
        images : Tensor
            ``(batch, 3, H, W)``
        spatial : bool
            If *True*, return spatial features ``(batch, num_positions, feature_dim)``
            for attention. Otherwise return pooled ``(batch, feature_dim)``.
        """
        outputs = self.backbone(images)

        if spatial:
            # Use last hidden state (feature map before pooling)
            feat = outputs.last_hidden_state  # (batch, feature_dim, h, w) for ResNet
            if feat.dim() == 4:
                b, c, h, w = feat.shape
                feat = feat.view(b, c, h * w).permute(0, 2, 1)  # (batch, h*w, c)
            return feat
        else:
            # Pooled feature vector
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                feat = outputs.pooler_output
                # ResNet pooler_output is (batch, feature_dim, 1, 1)
                if feat.dim() == 4:
                    feat = feat.squeeze(-1).squeeze(-1)
                return feat
            else:
                # Fallback: global avg pool over last hidden state
                feat = outputs.last_hidden_state
                if feat.dim() == 4:
                    feat = feat.mean(dim=[-2, -1])
                return feat

    @classmethod
    def from_config(cls, cfg) -> "ImageEncoder":
        """Build encoder from the ``encoder`` section of config."""
        return cls(
            pretrained=cfg.encoder.pretrained,
            feature_dim=cfg.encoder.get("feature_dim"),
            freeze=cfg.encoder.get("freeze", False),
        )

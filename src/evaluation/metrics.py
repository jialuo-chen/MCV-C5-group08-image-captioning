"""Evaluation metrics for image captioning.

Computes BLEU-1, BLEU-2, ROUGE-L, and METEOR using HuggingFace ``evaluate``.
"""

from __future__ import annotations

import evaluate


# Pre-load metrics (lazy singletons)
_bleu = None
_rouge = None
_meteor = None


def _get_metrics():
    global _bleu, _rouge, _meteor
    if _bleu is None:
        _bleu = evaluate.load("bleu")
    if _rouge is None:
        _rouge = evaluate.load("rouge")
    if _meteor is None:
        _meteor = evaluate.load("meteor")
    return _bleu, _rouge, _meteor


def compute_metrics(
    predictions: list[str],
    references: list[list[str]],
) -> dict[str, float]:
    """Compute captioning metrics.

    Parameters
    ----------
    predictions : list[str]
        Generated captions (one per image).
    references : list[list[str]]
        Reference captions (list of lists; multiple refs per image).

    Returns
    -------
    dict[str, float]
        ``{"bleu1", "bleu2", "rougeL", "meteor"}`` — all in [0, 1].
    """
    bleu_metric, rouge_metric, meteor_metric = _get_metrics()

    bleu1 = bleu_metric.compute(
        predictions=predictions, references=references, max_order=1,
    )
    bleu2 = bleu_metric.compute(
        predictions=predictions, references=references, max_order=2,
    )
    rouge = rouge_metric.compute(
        predictions=predictions, references=references,
    )
    meteor = meteor_metric.compute(
        predictions=predictions, references=references,
    )

    return {
        "bleu1": bleu1["bleu"],
        "bleu2": bleu2["bleu"],
        "rougeL": rouge["rougeL"],
        "meteor": meteor["meteor"],
    }


def format_metrics(metrics: dict[str, float]) -> str:
    """Format metrics dict as a readable string."""
    parts = []
    for k, v in metrics.items():
        parts.append(f"{k}: {v * 100:.2f}%")
    return " | ".join(parts)

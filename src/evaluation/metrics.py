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
    # Degenerate batches can happen early in training (e.g., empty generations).
    # Return safe zeros instead of crashing metric libraries.
    if not predictions or not references or len(predictions) != len(references):
        return {
            "bleu1": 0.0,
            "bleu2": 0.0,
            "rougeL": 0.0,
            "meteor": 0.0,
        }

    # Ensure each sample has at least one non-empty reference caption.
    placeholder = "<empty>"
    normalized_references: list[list[str]] = []
    for refs in references:
        if not refs:
            normalized_references.append([placeholder])
        else:
            cleaned = [r for r in refs if isinstance(r, str) and r.strip()]
            normalized_references.append(cleaned if cleaned else [placeholder])

    # BLEU in evaluate can raise ZeroDivisionError when all predictions are empty.
    # Use a single space placeholder so the metric returns ~0 rather than failing.
    normalized_predictions = [
        p if isinstance(p, str) and p.strip() else placeholder
        for p in predictions
    ]

    bleu_metric, rouge_metric, meteor_metric = _get_metrics()

    bleu1 = bleu_metric.compute(
        predictions=normalized_predictions, references=normalized_references, max_order=1,
    )
    bleu2 = bleu_metric.compute(
        predictions=normalized_predictions, references=normalized_references, max_order=2,
    )
    rouge = rouge_metric.compute(
        predictions=normalized_predictions, references=normalized_references,
    )
    meteor = meteor_metric.compute(
        predictions=normalized_predictions, references=normalized_references,
    )

    return {
        "bleu1": float(bleu1["bleu"]),
        "bleu2": float(bleu2["bleu"]),
        "rougeL": float(rouge["rougeL"]),
        "meteor": float(meteor["meteor"]),
    }


def format_metrics(metrics: dict[str, float]) -> str:
    """Format metrics dict as a readable string."""
    parts = []
    for k, v in metrics.items():
        parts.append(f"{k}: {v * 100:.2f}%")
    return " | ".join(parts)

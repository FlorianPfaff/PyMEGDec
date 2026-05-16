"""Public facade for Procrustes hyperalignment stimulus decoding."""

from __future__ import annotations

import sys

import numpy as np

from pymegdec import _stimulus_hyperalignment_legacy as _impl


def _topk_and_rank_metrics(true_labels, class_scores, score_classes):
    if class_scores.size == 0:
        return {"top2_accuracy": np.nan, "top3_accuracy": np.nan, "mean_true_label_rank": np.nan}
    ranks = np.asarray(_impl._true_label_ranks(true_labels, class_scores, score_classes), dtype=float)
    finite_ranks = ranks[np.isfinite(ranks)]
    if ranks.size == 0:
        return {"top2_accuracy": np.nan, "top3_accuracy": np.nan, "mean_true_label_rank": np.nan}
    return {
        "top2_accuracy": float(np.mean(ranks <= 2)),
        "top3_accuracy": float(np.mean(ranks <= 3)),
        "mean_true_label_rank": float(np.mean(finite_ranks)) if finite_ranks.size else np.nan,
    }


_impl._topk_and_rank_metrics = _topk_and_rank_metrics

sys.modules[__name__] = _impl
globals().update(_impl.__dict__)

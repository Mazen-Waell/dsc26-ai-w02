"""Decision metrics for a mule-risk score.

You write this file. Keep the signatures exactly as they are: the grader calls them by name.

Rules for every function:
- y_true and scores are lists or 1-D arrays. Raise ValueError if the lengths differ, the
  input is empty, a label is not 0 or 1, or a score is NaN.
- A case is flagged when score >= threshold.
- Precision is 0.0 when nothing is flagged; recall is 0.0 when there are no positives.
"""
from __future__ import annotations

import math

import numpy as np


def _prepare(y_true, scores):
    try:
        y = np.asarray(y_true)
        s = np.asarray(scores)
    except Exception as exc:
        raise ValueError("inputs must be array-like") from exc
    if y.ndim != 1 or s.ndim != 1:
        raise ValueError("inputs must be 1-D")
    if len(y) != len(s):
        raise ValueError("y_true and scores must have the same length")
    if len(y) == 0:
        raise ValueError("inputs must not be empty")
    try:
        s = s.astype(float)
        y_float = y.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("inputs must be numeric") from exc
    if np.isnan(s).any():
        raise ValueError("scores contain NaN")
    if not np.isin(y_float, [0.0, 1.0]).all():
        raise ValueError("labels must be 0 or 1")
    return y_float.astype(int), s


def _sweep(y, s):
    # scores sorted descending, so index i flags every case with score >= ss[i]
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    last = np.r_[ss[1:] != ss[:-1], True]
    return ss[last], tp[last], fp[last]


def confusion_at(y_true, scores, threshold: float) -> dict:
    """Return {"tp": int, "fp": int, "tn": int, "fn": int} at `threshold`."""
    y, s = _prepare(y_true, scores)
    flagged = s >= threshold
    return {
        "tp": int(np.sum(flagged & (y == 1))),
        "fp": int(np.sum(flagged & (y == 0))),
        "tn": int(np.sum(~flagged & (y == 0))),
        "fn": int(np.sum(~flagged & (y == 1))),
    }


def precision_recall_at(y_true, scores, threshold: float) -> tuple[float, float]:
    """Return (precision, recall) at `threshold`."""
    c = confusion_at(y_true, scores, threshold)
    flagged = c["tp"] + c["fp"]
    positives = c["tp"] + c["fn"]
    precision = c["tp"] / flagged if flagged else 0.0
    recall = c["tp"] / positives if positives else 0.0
    return float(precision), float(recall)


def precision_at_k(y_true, scores, k: int) -> float:
    """Share of positives among the k highest scores.

    ValueError if k <= 0; if k > n, use n. Rank by score descending; ties keep their
    original order.
    """
    y, s = _prepare(y_true, scores)
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(int(k), len(y))
    order = np.argsort(-s, kind="stable")
    return float(y[order[:k]].sum() / k)


def expected_cost(y_true, scores, threshold: float, *, cost_fp: float, cost_fn: float) -> float:
    """fp * cost_fp + fn * cost_fn at `threshold`."""
    c = confusion_at(y_true, scores, threshold)
    return float(c["fp"] * cost_fp + c["fn"] * cost_fn)


def best_threshold(y_true, scores, *, cost_fp: float, cost_fn: float) -> float:
    """Threshold with the lowest expected cost.

    Candidates are every distinct score plus math.inf ("flag nothing"). On a tie in cost,
    return the highest threshold (fewer flags for the same cost).
    """
    y, s = _prepare(y_true, scores)
    thresholds, tp, fp = _sweep(y, s)
    positives = int(y.sum())
    fn = positives - tp
    costs = np.r_[positives * cost_fn, fp * cost_fp + fn * cost_fn]
    candidates = np.r_[math.inf, thresholds]
    # candidates run from highest to lowest and argmin keeps the first minimum
    return float(candidates[int(np.argmin(costs))])


def threshold_for_precision(y_true, scores, min_precision: float) -> float | None:
    """Lowest distinct score whose precision is >= min_precision, or None if none reaches it."""
    y, s = _prepare(y_true, scores)
    thresholds, tp, fp = _sweep(y, s)
    precision = tp / (tp + fp)
    ok = precision >= min_precision
    if not ok.any():
        return None
    return float(thresholds[ok].min())
"""Data audit for Rawaj Pay transfer extracts.

You write this file. Keep the signature exactly as it is: the grader calls it by name.
Allowed dependencies: numpy, pandas, scikit-learn (and the standard library).
"""
from __future__ import annotations

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.metrics import roc_auc_score

MIN_ROWS = 20
AUC_HI, AUC_LO = 0.98, 0.02


def _missing_mask(s: pd.Series) -> pd.Series:
    mask = s.isna()
    if is_numeric_dtype(s) or is_bool_dtype(s):
        return mask
    blank = s.map(lambda v: isinstance(v, str) and v.strip() == "")
    return mask | blank.astype(bool)


def _extreme_auc(x, y) -> bool:
    if y.nunique() < 2:
        return False
    auc = roc_auc_score(y, x)
    return auc >= AUC_HI or auc <= AUC_LO


def _leakage_suspects(df, miss, labelled, y, skip) -> list:
    if labelled.sum() < MIN_ROWS or y[labelled].nunique() < 2:
        return []
    suspects = []
    for col in df.columns:
        if col in skip:
            continue
        # here missing counts as a value, unlike constant_columns
        if df[col].mask(miss[col]).nunique(dropna=False) <= 1:
            continue
        flagged = False
        if is_numeric_dtype(df[col]) or is_bool_dtype(df[col]):
            usable = labelled & ~miss[col]
            if usable.sum() >= MIN_ROWS:
                x = pd.to_numeric(df.loc[usable, col]).astype(float)
                flagged = _extreme_auc(x, y[usable])
        if not flagged:
            indicator = miss.loc[labelled, col].astype(int)
            flagged = _extreme_auc(indicator, y[labelled])
        if flagged:
            suspects.append(col)
    return sorted(suspects)


def audit(df: pd.DataFrame, *, target: str, id_col: str, time_col: str) -> dict:
    """Audit a transfer extract before anyone trains a model on it.

    Raises KeyError if `target`, `id_col` or `time_col` is not a column of `df`.

    Returns a dict with exactly these keys:

    - "n_rows" (int): number of rows.
    - "missing" ({column: float}): fraction missing, for every column. Missing means
      NaN/None, or a string that is empty or only whitespace.
    - "duplicate_ids" (list): sorted non-missing `id_col` values that appear more than once.
    - "duplicate_rows" (int): rows that exactly repeat an earlier row.
    - "constant_columns" (list): sorted columns other than `target` with at most one
      distinct non-missing value.
    - "unlabelled_rows" (int): rows where `target` is missing.
    - "positive_rate" (float): share of labelled rows with target == 1; 0.0 if none is labelled.
    - "bad_timestamps" (int): non-missing values in `time_col` that do not parse as dates.
    - "time_range" (list): [min, max] of the parsed timestamps as ISO strings, or [None, None].
    - "leakage_suspects" (list): sorted columns (not target, id_col or time_col) that look
      like leaks. See README.md for the exact rule.
    """
    for col in (target, id_col, time_col):
        if col not in df.columns:
            raise KeyError(col)

    n = len(df)
    miss = pd.DataFrame({c: _missing_mask(df[c]) for c in df.columns}, index=df.index)

    ids = df.loc[~miss[id_col], id_col]
    counts = ids.value_counts()
    duplicate_ids = sorted(counts[counts > 1].index.tolist())

    constant_columns = sorted(
        c for c in df.columns
        if c != target and df.loc[~miss[c], c].nunique() <= 1
    )

    labelled = ~miss[target]
    y_all = pd.to_numeric(df[target].mask(miss[target]), errors="coerce")
    y = (y_all == 1).astype(int)
    positive_rate = float(y[labelled].mean()) if labelled.any() else 0.0

    raw_time = df.loc[~miss[time_col], time_col]
    parsed = pd.to_datetime(raw_time, errors="coerce")
    bad_timestamps = int(parsed.isna().sum())
    valid = parsed.dropna()
    if valid.empty:
        time_range = [None, None]
    else:
        time_range = [valid.min().isoformat(), valid.max().isoformat()]

    suspects = _leakage_suspects(
        df, miss, labelled, y, skip={target, id_col, time_col}
    )

    return {
        "n_rows": int(n),
        "missing": {c: float(miss[c].mean()) if n else 0.0 for c in df.columns},
        "duplicate_ids": duplicate_ids,
        "duplicate_rows": int(df.duplicated().sum()),
        "constant_columns": constant_columns,
        "unlabelled_rows": int((~labelled).sum()),
        "positive_rate": positive_rate,
        "bad_timestamps": bad_timestamps,
        "time_range": time_range,
        "leakage_suspects": suspects,
    }
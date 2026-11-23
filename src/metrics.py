"""Drift and performance metrics used by the monitor."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def psi(reference, current, bins: int = 10) -> float:
    """Population stability index with bins set on the reference's deciles.

    Rule of thumb: < 0.1 stable, 0.1-0.2 watch, > 0.2 material shift.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant feature: fall back to its distinct values
        edges = np.unique(np.concatenate([reference, current]))
        edges = np.append(edges, edges[-1] + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(reference, edges)[0] / len(reference)
    cur_share = np.histogram(current, edges)[0] / len(current)
    eps = 1e-4
    ref_share, cur_share = np.clip(ref_share, eps, None), np.clip(cur_share, eps, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def top_decile_capture(y, score) -> float:
    """Share of all applicants found in the top 10% of scores."""
    y = np.asarray(y)
    k = max(1, int(len(y) * 0.10))
    top = np.argsort(-np.asarray(score), kind="stable")[:k]
    return float(y[top].sum() / max(y.sum(), 1))


def performance(y, score) -> dict:
    return {"auc": float(roc_auc_score(y, score)), "top_decile_capture": top_decile_capture(y, score)}


def psi_table(reference: pd.DataFrame, current: pd.DataFrame, columns) -> pd.Series:
    return pd.Series({c: psi(reference[c], current[c]) for c in columns})

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.metrics import cohen_kappa_score


def normalize_code_set(value: str | Iterable[str] | None) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",")]
    else:
        parts = [str(part).strip() for part in value]
    return frozenset(part for part in parts if part)


def raw_agreement(a: Iterable[str], b: Iterable[str]) -> float | None:
    pairs = list(zip(a, b))
    if not pairs:
        return None
    return sum(x == y for x, y in pairs) / len(pairs)


def safe_kappa(a: Iterable[str], b: Iterable[str]) -> float | None:
    a_list, b_list = list(a), list(b)
    if not a_list or len(a_list) != len(b_list):
        return None
    try:
        score = float(cohen_kappa_score(a_list, b_list))
    except Exception:
        return None
    return None if np.isnan(score) else score


def exact_set_agreement(a: Iterable[str], b: Iterable[str]) -> float | None:
    a_list = [normalize_code_set(x) for x in a]
    b_list = [normalize_code_set(x) for x in b]
    if not a_list or len(a_list) != len(b_list):
        return None
    return sum(x == y for x, y in zip(a_list, b_list)) / len(a_list)


def mean_jaccard(a: Iterable[str], b: Iterable[str]) -> float | None:
    a_list = [normalize_code_set(x) for x in a]
    b_list = [normalize_code_set(x) for x in b]
    if not a_list or len(a_list) != len(b_list):
        return None
    scores: list[float] = []
    for x, y in zip(a_list, b_list):
        union = x | y
        scores.append(1.0 if not union else len(x & y) / len(union))
    return sum(scores) / len(scores)


def pair_metrics(a: Iterable[str], b: Iterable[str], coding_mode: str) -> dict[str, float | None]:
    if coding_mode == "single":
        return {
            "raw_agreement": raw_agreement(a, b),
            "cohen_kappa": safe_kappa(a, b),
        }
    return {
        "exact_set_agreement": exact_set_agreement(a, b),
        "mean_jaccard": mean_jaccard(a, b),
    }

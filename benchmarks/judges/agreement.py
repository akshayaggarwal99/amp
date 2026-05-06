"""Inter-judge agreement metrics."""
from __future__ import annotations

from typing import Sequence


def kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Cohen's kappa for two binary judges.

    No sklearn dep — direct formula.
    """
    assert len(a) == len(b) and len(a) > 0
    n = len(a)
    agree = sum(1 for x, y in zip(a, b) if x == y)
    p_o = agree / n

    p_a1 = sum(a) / n
    p_b1 = sum(b) / n
    p_e = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)

    if p_e >= 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1 - p_e)


def raw_agreement(a: Sequence[int], b: Sequence[int]) -> float:
    assert len(a) == len(b) and len(a) > 0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)

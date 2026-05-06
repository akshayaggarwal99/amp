"""Bootstrap confidence intervals — no scipy dep."""
from __future__ import annotations

import random
from typing import Sequence


def bootstrap_mean_ci(
    scores: Sequence[float],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Returns (mean, ci_low, ci_high) at (1 - alpha) confidence."""
    if not scores:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    arr = list(scores)
    n = len(arr)
    means = []
    for _ in range(n_resamples):
        sample = [arr[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = int((1 - alpha / 2) * n_resamples) - 1
    return sum(arr) / n, means[lo_idx], means[hi_idx]

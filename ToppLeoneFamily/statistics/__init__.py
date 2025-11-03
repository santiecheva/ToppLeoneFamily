"""Statistical utilities for analysing Topp-Leone family datasets."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike


def empirical_ci(
    samples: ArrayLike,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Compute an empirical two-sided confidence interval."""
    data = np.asarray(samples, dtype=float)
    lower = np.percentile(data, 100 * alpha / 2)
    upper = np.percentile(data, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


__all__ = ["empirical_ci"]

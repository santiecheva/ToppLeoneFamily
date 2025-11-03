"""Estimation utilities for the Topp-Leone probability family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..distributions.base import NumberArray


@dataclass(slots=True)
class MLEFitResult:
    """Container for maximum-likelihood estimation outputs."""

    params: Mapping[str, float]
    success: bool
    message: str
    fun: float
    nfev: int | None = None
    jac: NumberArray | None = None
    hess_inv: NumberArray | None = None


__all__ = ["MLEFitResult"]

"""Core abstractions for the Topp-Leone family of continuous distributions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

NumberArray = NDArray[np.float64]


def _default_rng() -> np.random.Generator:
    """Create a reusable default pseudo-random number generator."""
    return np.random.default_rng()


class ContinuousDistribution(ABC):
    """Base interface for continuous probability distributions."""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self._rng: Final[np.random.Generator] = rng or _default_rng()

    @abstractmethod
    def pdf(self, x: ArrayLike) -> NumberArray:
        """Evaluate the probability density function on *x*."""

    @abstractmethod
    def cdf(self, x: ArrayLike) -> NumberArray:
        """Evaluate the cumulative distribution function on *x*."""

    @abstractmethod
    def ppf(self, q: ArrayLike) -> NumberArray:
        """Evaluate the percent point function (inverse CDF) on *q*."""

    def sample(
        self,
        size: int,
        *,
        random_state: np.random.Generator | None = None,
    ) -> NumberArray:
        """Generate *size* random draws from the distribution."""
        if size <= 0:
            raise ValueError("Sample size must be a positive integer.")
        rng = random_state or self._rng
        uniforms = rng.random(size=size)
        return self.ppf(uniforms)

    @staticmethod
    def _to_numpy(values: ArrayLike) -> NumberArray:
        """Convert user input into a 1-D NumPy array of floats."""
        array = np.asarray(values, dtype=float)
        if array.ndim == 0:
            array = np.expand_dims(array, axis=0)
        return array


__all__ = ["ContinuousDistribution", "NumberArray"]

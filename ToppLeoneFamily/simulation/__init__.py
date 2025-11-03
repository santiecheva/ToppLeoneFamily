"""Simulation helpers for the Topp-Leone family of distributions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..distributions.base import NumberArray


@dataclass(slots=True)
class SimulationSummary:
    """Empirical summary of Monte Carlo experiments."""

    estimates: NumberArray
    mean: NumberArray
    bias: NumberArray
    mse: NumberArray
    variance: NumberArray
    ci_lower: NumberArray
    ci_upper: NumberArray


__all__ = ["SimulationSummary"]

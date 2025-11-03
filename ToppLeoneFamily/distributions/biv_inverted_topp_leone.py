"""Bivariate Inverted Topp-Leone distribution utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import comb, gamma

from .base import NumberArray


@dataclass(slots=True)
class BivariateInvertedToppLeoneParameters:
    """Parameter bundle for the bivariate inverted Topp-Leone distribution."""

    nu1: float
    nu2: float
    xi: float


class BivariateInvertedToppLeoneDistribution:
    """Bivariate distribution with shared scale ``xi``."""

    def __init__(self, nu1: float, nu2: float, xi: float) -> None:
        self.params = BivariateInvertedToppLeoneParameters(
            nu1=float(nu1),
            nu2=float(nu2),
            xi=float(xi),
        )
        self._validate_parameters()

    @property
    def nu1(self) -> float:
        return self.params.nu1

    @property
    def nu2(self) -> float:
        return self.params.nu2

    @property
    def xi(self) -> float:
        return self.params.xi

    def normalization_constant(self) -> float:
        """Return the normalizing constant ``C(nu1, nu2, xi)``."""
        nu = self.nu1 + self.nu2
        numerator = 2.0 * gamma(nu + 1.0)
        denominator = (
            np.power(self.xi, nu) * gamma(self.nu1) * gamma(self.nu2)
        )
        return float(numerator / denominator)

    def pdf(self, x: ArrayLike, y: ArrayLike) -> NumberArray:
        """Evaluate the joint density at ``(x, y)``."""
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if x_arr.shape != y_arr.shape:
            raise ValueError("x and y must share the same shape.")

        density = np.zeros_like(x_arr, dtype=float)
        mask = (x_arr > 0) & (y_arr > 0)
        if np.any(mask):
            z = x_arr[mask] + y_arr[mask]
            coef = self.normalization_constant()
            density[mask] = (
                coef
                * np.power(x_arr[mask], self.nu1 - 1.0)
                * np.power(y_arr[mask], self.nu2 - 1.0)
                * np.power(1.0 + z / self.xi, -2.0 * (self.nu1 + self.nu2) - 1.0)
                * np.power(2.0 + z / self.xi, self.nu1 + self.nu2 - 1.0)
            )
        return density

    def correlation(self, max_j: int = 50) -> dict[str, float]:
        """Return correlation, covariance and marginal variances."""
        rho, cov, var_x, var_y = _rho_ibtl(
            self.nu1,
            self.nu2,
            self.xi,
            max_j=max_j,
        )
        return {
            "rho": float(rho),
            "covariance": float(cov),
            "var_x": float(var_x),
            "var_y": float(var_y),
        }

    @staticmethod
    def correlation_grid(
        nu1_values: Sequence[float],
        nu2_values: Sequence[float],
        xi: float,
        *,
        max_j: int = 50,
        as_dataframe: bool = False,
    ) -> NumberArray | "pd.DataFrame":
        """Evaluate the correlation grid for combinations of ``nu1`` and ``nu2``."""
        nu1_array = np.asarray(nu1_values, dtype=float)
        nu2_array = np.asarray(nu2_values, dtype=float)
        grid = np.full((nu1_array.size, nu2_array.size), np.nan, dtype=float)

        for i, nu1 in enumerate(nu1_array):
            for j, nu2 in enumerate(nu2_array):
                rho, _, _, _ = _rho_ibtl(nu1, nu2, xi, max_j=max_j)
                grid[i, j] = rho

        if as_dataframe:
            try:
                import pandas as pd
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "pandas is required when as_dataframe=True."
                ) from exc
            df = pd.DataFrame(
                grid,
                index=pd.Index(nu1_array, name="nu1"),
                columns=pd.Index(nu2_array, name="nu2"),
            )
            return df

        return grid

    def _validate_parameters(self) -> None:
        if self.nu1 <= 0 or self.nu2 <= 0 or self.xi <= 0:
            raise ValueError("nu1, nu2 and xi must be strictly positive.")


def _h2(nu: float, *, max_j: int = 50) -> float:
    total = 0.0
    for j in range(max_j + 1):
        for i in range(j + 3):
            coef = comb(j + 2, i)
            sign = (-1) ** i
            gamma_num = gamma((1.0 + i) / 2.0) * gamma(nu)
            gamma_den = gamma(nu + (1.0 + i) / 2.0)
            total += coef * sign * gamma_num / gamma_den
    return float(total)


def _rho_ibtl(
    nu1: float,
    nu2: float,
    xi: float,
    *,
    max_j: int = 50,
) -> tuple[float, float, float, float]:
    if nu1 <= 0 or nu2 <= 0 or xi <= 0:
        raise ValueError("nu1, nu2 and xi must be strictly positive.")
    nu = nu1 + nu2
    gamma_term = (np.sqrt(np.pi) * gamma(nu + 1.0)) / gamma(nu + 0.5)
    common_expr = (gamma_term - 1.0) ** 2
    xi_sq = xi**2
    nu_sq = nu**2
    h2_val = _h2(nu, max_j=max_j)

    var_x = ((nu1 * (nu1 + 1.0)) / nu) * xi_sq * h2_val - (xi_sq * nu1**2 / nu_sq) * common_expr
    var_y = ((nu2 * (nu2 + 1.0)) / nu) * xi_sq * h2_val - (xi_sq * nu2**2 / nu_sq) * common_expr
    cov_xy = ((nu1 * nu2) / nu) * xi_sq * h2_val - (xi_sq * nu1 * nu2 / nu_sq) * common_expr

    rho_xy = cov_xy / np.sqrt(var_x * var_y)
    return float(rho_xy), float(cov_xy), float(var_x), float(var_y)


__all__ = [
    "BivariateInvertedToppLeoneDistribution",
    "BivariateInvertedToppLeoneParameters",
]

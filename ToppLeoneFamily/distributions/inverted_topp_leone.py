"""Inverted Topp-Leone distribution utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy import integrate, optimize

from .base import ContinuousDistribution, NumberArray
from ..estimation import MLEFitResult
from ..statistics import empirical_ci


@dataclass(slots=True)
class InvertedToppLeoneParameters:
    """Parameter bundle for the Inverted Topp-Leone distribution."""

    nu: float
    xi: float


class InvertedToppLeoneDistribution(ContinuousDistribution):
    """Continuous distribution supported on ``(0, ∞)``."""

    def __init__(
        self,
        nu: float,
        xi: float,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(rng=rng)
        self._params = InvertedToppLeoneParameters(nu=float(nu), xi=float(xi))
        self._validate_parameters()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def params(self) -> InvertedToppLeoneParameters:
        return self._params

    @property
    def nu(self) -> float:
        return self._params.nu

    @property
    def xi(self) -> float:
        return self._params.xi

    # ------------------------------------------------------------------
    # Distribution interface
    # ------------------------------------------------------------------
    def pdf(self, x: ArrayLike) -> NumberArray:
        values = self._to_numpy(x)
        pdf = np.zeros_like(values, dtype=float)
        mask = values > 0
        if np.any(mask):
            ratio = values[mask] / self.xi
            pdf[mask] = (
                2.0
                * self.nu
                * np.power(ratio, self.nu - 1.0)
                * np.power(1.0 + ratio, -2.0 * self.nu - 1.0)
                * np.power(2.0 + ratio, self.nu - 1.0)
            )
        return pdf

    def cdf(self, x: ArrayLike) -> NumberArray:
        values = self._to_numpy(x)
        cdf = np.zeros_like(values, dtype=float)
        mask = values >= 0
        beta = self.xi / (self.xi + values[mask])
        cdf[mask] = np.power(1.0 - beta, 2.0 * self.nu)
        return cdf

    def ppf(self, q: ArrayLike) -> NumberArray:
        probabilities = self._to_numpy(q)
        if np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("Probabilities must be in the interval [0, 1].")
        bracket = np.power(1.0 - np.power(probabilities, 1.0 / self.nu), -0.5)
        return self.xi * (bracket - 1.0)

    # ------------------------------------------------------------------
    # Likelihood utilities
    # ------------------------------------------------------------------
    def raw_moment(self, order: float) -> float:
        """Return the raw moment of order ``order``."""
        if order < 0:
            raise ValueError("Moment order must be non-negative.")

        def integrand(x: float) -> float:
            return (x**order) * self.pdf(np.array([x], dtype=float))[0]

        result, _ = integrate.quad(integrand, 0.0, np.inf, limit=200)
        return float(result)

    def mean(self) -> float:
        """Return the expected value."""
        return self.raw_moment(1.0)

    def variance(self) -> float:
        """Return the variance."""
        second_moment = self.raw_moment(2.0)
        mean = self.mean()
        return float(second_moment - mean**2)

    def log_pdf(self, x: ArrayLike) -> NumberArray:
        values = self._ensure_positive(x)
        ratio = values / self.xi
        return (
            np.log(2.0)
            + np.log(self.nu)
            + (self.nu - 1.0) * (np.log(ratio) + np.log(2.0 + ratio))
            - (2.0 * self.nu + 1.0) * np.log(1.0 + ratio)
        )

    def log_likelihood(self, sample: ArrayLike) -> float:
        data = self._ensure_positive(sample)
        return float(np.sum(self.log_pdf(data)))

    def score(self, sample: ArrayLike) -> NumberArray:
        data = self._ensure_positive(sample)
        return _score_inverted_topp_leone(data, self.nu, self.xi)

    @staticmethod
    def fit_mle(
        sample: ArrayLike,
        *,
        initial_guess: Tuple[float, float] = (1.0, 1.0),
        tol: float = 1e-8,
        maxfev: int = 10_000,
    ) -> MLEFitResult:
        """Solve the MLE score equations using ``scipy.optimize.root``."""
        data = ContinuousDistribution._to_numpy(sample)
        if np.any(data <= 0):
            raise ValueError("All observations must be strictly positive.")

        def score_equations(params: ArrayLike) -> NumberArray:
            nu, xi = params
            return _score_inverted_topp_leone(data, nu, xi)

        sol = optimize.root(
            score_equations,
            x0=np.asarray(initial_guess, dtype=float),
            method="hybr",
            tol=tol,
            options={"maxfev": maxfev},
        )
        params = dict(zip(("nu", "xi"), sol.x))
        return MLEFitResult(
            params=params,
            success=bool(sol.success),
            message=sol.message,
            fun=float(np.linalg.norm(sol.fun)),
            nfev=sol.nfev,
            jac=None,
            hess_inv=None,
        )

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def simulate_inverse_transform(
        sample_size: int,
        nu: float,
        xi: float,
        *,
        rng: np.random.Generator | None = None,
    ) -> NumberArray:
        generator = rng or np.random.default_rng()
        uniforms = generator.uniform(low=0.0, high=1.0, size=sample_size)
        dist = InvertedToppLeoneDistribution(nu, xi, rng=generator)
        return dist.ppf(uniforms)

    @staticmethod
    def simulate_mle_statistics(
        nu_true: float,
        xi_true: float,
        *,
        sample_size: int,
        reps: int = 1_000,
        initial_guess: Tuple[float, float] = (1.0, 1.0),
        alpha: float = 0.05,
        rng: np.random.Generator | None = None,
    ) -> dict[str, float | tuple[float, float]]:
        generator = rng or np.random.default_rng()
        estimates = np.zeros((reps, 2), dtype=float)
        for i in range(reps):
            sample = InvertedToppLeoneDistribution.simulate_inverse_transform(
                sample_size,
                nu_true,
                xi_true,
                rng=generator,
            )
            fit = InvertedToppLeoneDistribution.fit_mle(
                sample,
                initial_guess=initial_guess,
            )
            estimates[i] = np.array([fit.params["nu"], fit.params["xi"]])

        mean_estimate = estimates.mean(axis=0)
        bias = mean_estimate - np.array([nu_true, xi_true])
        mse = np.mean((estimates - np.array([nu_true, xi_true])) ** 2, axis=0)
        variance = np.var(estimates, axis=0, ddof=1)
        ci_lower = np.zeros(2)
        ci_upper = np.zeros(2)
        for idx in range(2):
            lower, upper = empirical_ci(estimates[:, idx], alpha=alpha)
            ci_lower[idx], ci_upper[idx] = lower, upper

        return {
            "mean_nu": float(mean_estimate[0]),
            "mean_xi": float(mean_estimate[1]),
            "bias_nu": float(bias[0]),
            "bias_xi": float(bias[1]),
            "mse_nu": float(mse[0]),
            "mse_xi": float(mse[1]),
            "var_nu": float(variance[0]),
            "var_xi": float(variance[1]),
            "ci_nu": (float(ci_lower[0]), float(ci_upper[0])),
            "ci_xi": (float(ci_lower[1]), float(ci_upper[1])),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_positive(self, values: ArrayLike) -> NumberArray:
        data = self._to_numpy(values)
        if np.any(data <= 0):
            raise ValueError("All values must be strictly positive.")
        return data

    def _validate_parameters(self) -> None:
        if self.nu <= 0 or self.xi <= 0:
            raise ValueError("Both nu and xi must be positive.")


def _log_likelihood_inverted_topp_leone(
    sample: NumberArray,
    nu: float,
    xi: float,
) -> float:
    if nu <= 0 or xi <= 0:
        return -np.inf
    ratio = sample / xi
    loglike = (
        sample.size * (np.log(2.0) + np.log(nu) - nu * np.log(xi))
        + (nu - 1.0) * np.sum(np.log(sample))
        - (2.0 * nu + 1.0) * np.sum(np.log(1.0 + ratio))
        + (nu - 1.0) * np.sum(np.log(2.0 + ratio))
    )
    return float(loglike)


def _score_inverted_topp_leone(
    sample: NumberArray,
    nu: float,
    xi: float,
) -> NumberArray:
    if nu <= 0 or xi <= 0:
        raise ValueError("Parameters nu and xi must be positive.")
    ratio = sample / xi
    grad_nu = (
        sample.size / nu
        - sample.size * np.log(xi)
        + np.sum(np.log(sample))
        - 2.0 * np.sum(np.log(1.0 + ratio))
        + np.sum(np.log(2.0 + ratio))
    )
    grad_xi = (
        -(sample.size * nu) / xi
        + ((2.0 * nu + 1.0) / xi) * np.sum(ratio / (1.0 + ratio))
        - ((nu - 1.0) / xi) * np.sum(ratio / (2.0 + ratio))
    )
    return np.array([grad_nu, grad_xi], dtype=float)


__all__ = [
    "InvertedToppLeoneDistribution",
    "InvertedToppLeoneParameters",
    "_log_likelihood_inverted_topp_leone",
]

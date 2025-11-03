"""Topp-Leone distribution utilities."""

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
class ToppLeoneParameters:
    """Parameter bundle for the Topp-Leone distribution."""

    nu: float
    sigma: float


class ToppLeoneDistribution(ContinuousDistribution):
    """Univariate Topp-Leone distribution with parameters ``(nu, sigma)``."""

    def __init__(
        self,
        nu: float,
        sigma: float,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(rng=rng)
        self._params = ToppLeoneParameters(nu=float(nu), sigma=float(sigma))
        self._validate_parameters()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def params(self) -> ToppLeoneParameters:
        """Return the distribution parameters."""
        return self._params

    @property
    def nu(self) -> float:
        return self._params.nu

    @property
    def sigma(self) -> float:
        return self._params.sigma

    def pdf(self, x: ArrayLike) -> NumberArray:
        values = self._to_numpy(x)
        pdf = np.zeros_like(values, dtype=float)
        mask = (values >= 0) & (values <= self.sigma)
        if np.any(mask):
            z = values[mask] / self.sigma
            pdf[mask] = (
                (2.0 * self.nu / self.sigma)
                * np.power(z, self.nu - 1.0)
                * (1.0 - z)
                * np.power(2.0 - z, self.nu - 1.0)
            )
        return pdf

    def cdf(self, x: ArrayLike) -> NumberArray:
        values = self._to_numpy(x)
        cdf = np.zeros_like(values, dtype=float)
        mask_mid = (values >= 0) & (values < self.sigma)
        if np.any(mask_mid):
            z = values[mask_mid] / self.sigma
            cdf[mask_mid] = np.power(z * (2.0 - z), self.nu)
        cdf[values >= self.sigma] = 1.0
        return cdf

    def ppf(self, q: ArrayLike) -> NumberArray:
        probabilities = self._to_numpy(q)
        if np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("Probabilities must be in the interval [0, 1].")
        z = 1.0 - np.sqrt(1.0 - np.power(probabilities, 1.0 / self.nu))
        return self.sigma * z

    # ------------------------------------------------------------------
    # Likelihood utilities
    # ------------------------------------------------------------------
    def raw_moment(self, order: float) -> float:
        """Return the raw moment of order ``order``."""
        if order < 0:
            raise ValueError("Moment order must be non-negative.")

        def integrand(x: float) -> float:
            return (x**order) * float(self.pdf(np.array([x])))

        result, _ = integrate.quad(integrand, 0.0, self.sigma, limit=200)
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
        values = self._ensure_in_support(x)
        z = values / self.sigma
        result = (
            np.log(2.0 * self.nu / self.sigma)
            + (self.nu - 1.0) * np.log(z)
            + np.log(1.0 - z)
            + (self.nu - 1.0) * np.log(2.0 - z)
        )
        return result

    def log_likelihood(self, sample: ArrayLike) -> float:
        values = self._ensure_in_support(sample)
        return float(np.sum(self.log_pdf(values)))

    def score(self, sample: ArrayLike) -> NumberArray:
        values = self._ensure_in_support(sample)
        return _score_topp_leone(values, self.nu, self.sigma)

    @staticmethod
    def fit_mle(
        sample: ArrayLike,
        *,
        initial_guess: Tuple[float, float] | None = None,
        method: str = "L-BFGS-B",
    ) -> MLEFitResult:
        """Fit ``(nu, sigma)`` via maximum likelihood."""
        data = ToppLeoneDistribution._sanitize_sample(sample)
        if initial_guess is None:
            nu0 = 1.0
            sigma0 = max(np.max(data) * 1.1, 1.0)
            initial_guess = (nu0, sigma0)

        lower_sigma = np.max(data) * (1.0 + 1e-6)
        bounds = ((1e-6, None), (lower_sigma, None))

        def objective(params: ArrayLike) -> float:
            nu, sigma = params
            loglike = _log_likelihood_topp_leone(data, nu, sigma)
            return -loglike

        result = optimize.minimize(
            objective,
            x0=np.asarray(initial_guess, dtype=float),
            method=method,
            bounds=bounds,
        )

        params = dict(zip(("nu", "sigma"), result.x))
        return MLEFitResult(
            params=params,
            success=bool(result.success),
            message=result.message,
            fun=float(result.fun),
            nfev=result.nfev,
            jac=getattr(result, "jac", None),
            hess_inv=getattr(result, "hess_inv", None),
        )

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def simulate_mle_statistics(
        nu_true: float,
        sigma_true: float,
        *,
        sample_size: int,
        reps: int = 1_000,
        rng: np.random.Generator | None = None,
        initial_guess: Tuple[float, float] | None = None,
        alpha: float = 0.05,
    ) -> dict[str, float | tuple[float, float]]:
        """Monte Carlo study of the MLE behaviour."""
        generator = rng or np.random.default_rng()
        estimates = np.zeros((reps, 2), dtype=float)
        dist = ToppLeoneDistribution(nu_true, sigma_true, rng=generator)

        for i in range(reps):
            sample = dist.sample(sample_size, random_state=generator)
            fit = ToppLeoneDistribution.fit_mle(
                sample,
                initial_guess=initial_guess,
                method="L-BFGS-B",
            )
            estimates[i] = np.array([fit.params["nu"], fit.params["sigma"]])

        mean_estimate = estimates.mean(axis=0)
        bias = mean_estimate - np.array([nu_true, sigma_true])
        mse = np.mean((estimates - np.array([nu_true, sigma_true])) ** 2, axis=0)
        variance = np.var(estimates, axis=0, ddof=1)
        ci_lower = np.zeros(2)
        ci_upper = np.zeros(2)
        for idx, label in enumerate(("nu", "sigma")):
            lower, upper = empirical_ci(estimates[:, idx], alpha=alpha)
            ci_lower[idx], ci_upper[idx] = lower, upper

        return {
            "mean_nu": float(mean_estimate[0]),
            "mean_sigma": float(mean_estimate[1]),
            "bias_nu": float(bias[0]),
            "bias_sigma": float(bias[1]),
            "mse_nu": float(mse[0]),
            "mse_sigma": float(mse[1]),
            "var_nu": float(variance[0]),
            "var_sigma": float(variance[1]),
            "ci_nu": (float(ci_lower[0]), float(ci_upper[0])),
            "ci_sigma": (float(ci_lower[1]), float(ci_upper[1])),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize_sample(sample: ArrayLike) -> NumberArray:
        data = ContinuousDistribution._to_numpy(sample)
        if np.any(data <= 0):
            raise ValueError("Observations must be strictly positive.")
        return data

    def _ensure_in_support(self, values: ArrayLike) -> NumberArray:
        data = self._sanitize_sample(values)
        if np.any(data >= self.sigma):
            raise ValueError("All observations must be smaller than sigma.")
        return data

    def _validate_parameters(self) -> None:
        if self.nu <= 0 or self.sigma <= 0:
            raise ValueError("Both nu and sigma must be positive.")


def _log_likelihood_topp_leone(
    sample: NumberArray,
    nu: float,
    sigma: float,
) -> float:
    if nu <= 0 or sigma <= 0:
        return -np.inf
    if sigma <= np.max(sample):
        return -np.inf
    z = sample / sigma
    loglike = (
        sample.size * (np.log(2.0 * nu) - np.log(sigma))
        + (nu - 1.0) * np.sum(np.log(z))
        + np.sum(np.log(1.0 - z))
        + (nu - 1.0) * np.sum(np.log(2.0 - z))
    )
    return float(loglike)


def _score_topp_leone(
    sample: NumberArray,
    nu: float,
    sigma: float,
) -> NumberArray:
    z = sample / sigma
    grad_nu = (
        sample.size / nu
        + np.sum(np.log(z))
        + np.sum(np.log(2.0 - z))
    )
    grad_sigma = (
        -sample.size * nu / sigma
        + np.sum(sample / (sigma**2 * (1.0 - z)))
        + (nu - 1.0) * np.sum(sample / (sigma**2 * (2.0 - z)))
    )
    return np.array([grad_nu, grad_sigma], dtype=float)


__all__ = ["ToppLeoneDistribution", "ToppLeoneParameters"]

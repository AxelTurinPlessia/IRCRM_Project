"""Nelson–Siegel and Nelson–Siegel–Svensson yield calibration."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import minimum_filter
from scipy.optimize import minimize


def _yield_design(times, decay_rates):
    """Yield loadings, with the continuous extension at zero maturity."""
    times = np.asarray(times, dtype=float)
    x = times[..., None] * decay_rates
    slope = np.divide(-np.expm1(-x), x, out=np.ones_like(x), where=x != 0)
    curvature = slope - np.exp(-x)
    columns = [np.ones_like(times), slope[..., 0], curvature[..., 0]]
    if len(decay_rates) == 2:
        columns.append(curvature[..., 1])
    return np.stack(columns, axis=-1)


@dataclass(frozen=True)
class SmoothCurve:
    betas: tuple[float, ...]
    decay_rates: tuple[float, ...]

    def spot_rate(self, times):
        """Continuously compounded spot rates in decimal units."""
        return _yield_design(times, self.decay_rates) @ self.betas

    def discount(self, times):
        """Discount factors at ACT/365 times in years."""
        return np.exp(-np.asarray(times) * self.spot_rate(times))

    def forward_rate(self, times):
        """Instantaneous forward rates on the ACT/365 time axis."""
        x = np.asarray(times, dtype=float)[..., None] * self.decay_rates
        exponential = np.exp(-x)
        beta_0, beta_1, beta_2 = self.betas[:3]
        forward = beta_0 + beta_1 * exponential[..., 0] + beta_2 * x[..., 0] * exponential[..., 0]
        if len(self.decay_rates) == 2:
            forward += self.betas[3] * x[..., 1] * exponential[..., 1]
        return forward


def calibrate_curve(times, yields, decay_count: int) -> tuple[SmoothCurve, float]:
    """Fit NS (one decay) or NSS (two). Return the curve and yield RMSE in bp."""
    if decay_count not in (1, 2):
        raise ValueError("NS uses one decay rate and NSS uses two")
    yields_bp = 10_000.0 * np.asarray(yields, dtype=float)

    def profile(log_decays):
        design = _yield_design(times, np.exp(log_decays))
        betas_bp = np.linalg.lstsq(design, yields_bp, rcond=None)[0]
        residuals = design @ betas_bp - yields_bp
        return np.mean(residuals ** 2), betas_bp

    def objective(log_decays):
        return profile(log_decays)[0]

    # Search a broad finite range of positive decay rates in inverse years.
    # Refine distinct grid minima to reduce sensitivity to the starting point.
    bounds = [(np.log(1e-4), np.log(1e4))] * decay_count
    grid = np.linspace(*bounds[0], num=95)
    grid_scores = np.empty((len(grid),) * decay_count)
    for index in np.ndindex(grid_scores.shape):
        grid_scores[index] = objective(grid[list(index)])
    minima = np.argwhere(grid_scores == minimum_filter(grid_scores, size=3, mode="nearest"))
    starts = sorted(minima, key=lambda index: grid_scores[tuple(index)])[:8]
    fits = [minimize(objective, grid[index], method="Nelder-Mead", bounds=bounds,
                     options={"xatol": 1e-9, "fatol": 1e-12}) for index in starts]
    best = min((fit for fit in fits if fit.success), key=lambda fit: fit.fun)
    mse_bp, betas_bp = profile(best.x)
    curve = SmoothCurve(tuple(betas_bp / 10_000.0), tuple(np.exp(best.x)))
    return curve, float(np.sqrt(mse_bp))

"""Resolution kernels for linear historical summaries identified by an SFS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .operators import coalescent_rates


@dataclass(frozen=True)
class ResolutionResult:
    tau: np.ndarray
    target: np.ndarray
    kernel: np.ndarray
    coefficients: np.ndarray
    squared_kernel_error: float
    coefficient_norm: float
    kernel_integral: float
    negative_mass: float
    interval: tuple[float, float]
    ridge: float


def interval_projection_kernel(
    n: int,
    interval: tuple[float, float],
    *,
    horizon: float = 2.0,
    ridge: float = 1e-6,
    grid_size: int = 4001,
) -> ResolutionResult:
    """Approximate an interval-average functional using SFS moment kernels.

    The target integrates h(tau) uniformly over ``interval``. The returned
    kernel K(tau)=sum_m q_m exp(-choose(m,2) tau) solves the ridge-regularised
    L2 projection problem on [0,horizon]. The coefficient norm records noise
    amplification; negative mass records potentially misleading side-lobes.
    """
    left, right = interval
    if not (0 <= left < right <= horizon):
        raise ValueError("interval must lie inside [0,horizon]")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    rates = coalescent_rates(n)
    summed = rates[:, None] + rates[None, :]
    gram = (1.0 - np.exp(-summed * horizon)) / summed
    target_inner = (
        np.exp(-rates * left) - np.exp(-rates * right)
    ) / (rates * (right - left))
    coefficients = np.linalg.solve(
        gram + ridge * np.eye(len(rates)), target_inner
    )
    tau = np.linspace(0.0, horizon, grid_size)
    kernel = np.exp(-np.outer(tau, rates)) @ coefficients
    target = np.where((tau >= left) & (tau <= right), 1.0 / (right - left), 0.0)
    error = float(np.trapezoid((kernel - target) ** 2, tau))
    kernel_integral = float(np.trapezoid(kernel, tau))
    negative_mass = float(np.trapezoid(np.maximum(-kernel, 0.0), tau))
    return ResolutionResult(
        tau=tau,
        target=target,
        kernel=kernel,
        coefficients=coefficients,
        squared_kernel_error=error,
        coefficient_norm=float(np.linalg.norm(coefficients)),
        kernel_integral=kernel_integral,
        negative_mass=negative_mass,
        interval=interval,
        ridge=ridge,
    )

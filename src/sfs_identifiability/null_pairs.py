"""Exact finite-SFS null directions built from rational Laplace transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import mpmath as mp
import numpy as np


@dataclass(frozen=True)
class NullPairResult:
    sample_size: int
    tau: np.ndarray
    perturbation: np.ndarray
    history_plus: np.ndarray
    history_minus: np.ndarray
    poles: np.ndarray
    residues: tuple[str, ...]
    max_scaled_moment_residual: str
    sign_changes: int
    epsilon: float
    global_perturbation_bound: float = float("nan")
    positivity_margin: float = float("nan")
    positivity_certified: bool = False


def _polynomial_sup_bound(residues: list, *, subdivisions: int) -> "mp.mpf":
    """Upper bound for sup_{x in [0,1]} |sum_j r_j x^j| via Bernstein hulls.

    For each uniform subinterval [a, b], the polynomial is re-expanded around a
    (exact binomial shift of the monomial coefficients), rescaled to [0, 1] and
    converted to the Bernstein basis, whose coefficient hull bounds the local
    supremum. The returned value is the maximum over all subintervals.
    """
    from math import comb as _comb

    degree = len(residues)
    monomial = [mp.mpf(0)] + [mp.mpf(value) for value in residues]
    bound = mp.mpf(0)
    width = mp.mpf(1) / subdivisions
    for piece in range(subdivisions):
        a = piece * width
        # Coefficients of p(a + w u) in u on [0, 1].
        shifted = [mp.mpf(0)] * (degree + 1)
        for j in range(degree + 1):
            if monomial[j] == 0:
                continue
            for l in range(j + 1):
                shifted[l] += (
                    monomial[j]
                    * _comb(j, l)
                    * (a ** (j - l))
                    * (width**l)
                )
        # Bernstein coefficients: B_i = sum_{l<=i} C(i,l)/C(d,l) c_l.
        local = mp.mpf(0)
        for i in range(degree + 1):
            acc = mp.mpf(0)
            for l in range(i + 1):
                acc += (
                    shifted[l] * _comb(i, l) / _comb(degree, l)
                )
            local = max(local, abs(acc))
        bound = max(bound, local)
    return bound


def _sign_changes(values: np.ndarray, tolerance: float = 1e-10) -> int:
    signs = np.sign(values)
    signs[np.abs(values) <= tolerance] = 0
    nonzero = signs[signs != 0]
    return int(np.sum(nonzero[1:] * nonzero[:-1] < 0))


def analytic_null_pair(
    n: int = 12,
    tau: Iterable[float] | None = None,
    *,
    epsilon: float = 0.8,
    precision: int = 100,
) -> NullPairResult:
    """Construct positive histories with identical exact finite-n moments.

    Let a_m=choose(m,2), m=2,...,n, and choose poles b_j=j, j=1,...,n.
    The rational function

        G(s) = prod_m (s-a_m) / prod_j (s+b_j)

    is the Laplace transform of a finite exponential sum g(tau), and G(a_m)=0.
    After normalising g on the supplied plotting grid, h_+ = 1+epsilon*g and
    h_- = 1-epsilon*g remain positive for 0<epsilon<1 and have identical
    c_2,...,c_n, hence identical expected SFS.
    """
    if n < 3:
        raise ValueError("n must be at least 3")
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must lie in (0,1)")
    tau_arr = (
        np.linspace(0.0, 5.0, 5001)
        if tau is None
        else np.asarray(list(tau), dtype=float)
    )
    if tau_arr.ndim != 1 or np.any(tau_arr < 0):
        raise ValueError("tau must be a non-negative one-dimensional grid")

    mp.mp.dps = precision
    rates = [mp.mpf(m * (m - 1)) / 2 for m in range(2, n + 1)]
    poles = [mp.mpf(j) for j in range(1, n + 1)]
    residues: list[mp.mpf] = []
    for j, pole in enumerate(poles):
        numerator = mp.mpf(1)
        for rate in rates:
            numerator *= -pole - rate
        denominator = mp.mpf(1)
        for ell, other in enumerate(poles):
            if ell != j:
                denominator *= other - pole
        residues.append(numerator / denominator)

    values_mp = []
    for value in tau_arr:
        x = mp.mpf(str(float(value)))
        values_mp.append(sum(r * mp.exp(-b * x) for r, b in zip(residues, poles)))
    max_abs = max(abs(value) for value in values_mp)
    scaled_residues = [r / max_abs for r in residues]
    perturbation = np.asarray([float(value / max_abs) for value in values_mp])

    residuals = [
        sum(r / (rate + pole) for r, pole in zip(scaled_residues, poles))
        for rate in rates
    ]
    max_residual = max(abs(value) for value in residuals)

    plus = 1.0 + epsilon * perturbation
    minus = 1.0 - epsilon * perturbation
    if min(float(np.min(plus)), float(np.min(minus))) <= 0:
        raise RuntimeError("constructed histories are not positive")

    # Global positivity certificate over all of [0, infinity), not only the
    # plotting grid. Substituting x = e^{-tau} turns the perturbation into the
    # polynomial q(x) = sum_j r_j x^j on x in [0, 1], where the individually
    # enormous alternating residues cancel exactly. The polynomial supremum is
    # bounded rigorously by the Bernstein convex-hull property: on each of
    # ``subdivisions`` uniform subintervals, |q| <= max |Bernstein coefficient|
    # of the locally rescaled polynomial. The bound converges quadratically
    # under subdivision and respects cancellation, unlike coefficient-magnitude
    # or Lipschitz estimates.
    global_bound = float(
        _polynomial_sup_bound(scaled_residues, subdivisions=512)
    )
    margin = 1.0 - epsilon * global_bound
    certified = margin > 0.0

    return NullPairResult(
        sample_size=n,
        tau=tau_arr,
        perturbation=perturbation,
        history_plus=plus,
        history_minus=minus,
        poles=np.asarray([float(value) for value in poles]),
        residues=tuple(mp.nstr(value, 50) for value in scaled_residues),
        max_scaled_moment_residual=mp.nstr(max_residual, 12),
        sign_changes=_sign_changes(perturbation),
        epsilon=epsilon,
        global_perturbation_bound=global_bound,
        positivity_margin=margin,
        positivity_certified=certified,
    )

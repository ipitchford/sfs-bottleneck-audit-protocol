"""Generic certified linear programming for the empirical audit.

Every reported endpoint is solved by two HiGHS algorithms and carries a
solver-independent Lagrangian dual certificate evaluated in 50-digit
arithmetic. For min c'x subject to A_ub x <= b_ub, A_eq x = b_eq and finite
box bounds, any multipliers (y free, mu >= 0) give the valid lower bound

    g(y, mu) = -y'b_eq - mu'b_ub + sum_i min over [l_i, u_i] of r_i x_i,
    r = c + A_eq'y + A_ub'mu,

by weak duality. Candidate sign conventions are screened in floating point;
only the winning candidate is evaluated in high precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp
import numpy as np
from scipy.optimize import linprog

_CERTIFICATE_DPS = 50


class LPInfeasible(RuntimeError):
    """The declared class cannot reproduce the observed spectrum.

    Infeasibility is a reportable scientific outcome (class rejection at the
    declared budget and tolerance), not a numerical failure.
    """


@dataclass(frozen=True)
class CertifiedLP:
    value: float
    x: np.ndarray
    certified_bound: float
    dual_gap: float
    solver_agreement: float
    status: str


def _dual_value_float(c, a_ub, b_ub, a_eq, b_eq, lower, upper, y, mu):
    mu = np.maximum(mu, 0.0)
    reduced = c + (a_eq.T @ y if a_eq is not None else 0.0) + a_ub.T @ mu
    total = -(b_eq @ y if a_eq is not None else 0.0) - b_ub @ mu
    return total + np.minimum(reduced * lower, reduced * upper).sum()


def _dual_value_mp(c, a_ub, b_ub, a_eq, b_eq, lower, upper, y, mu):
    mu = np.maximum(mu, 0.0)
    with mp.workdps(_CERTIFICATE_DPS):
        total = mp.mpf(0)
        if a_eq is not None:
            for value, weight in zip(b_eq, y):
                total -= mp.mpf(float(weight)) * mp.mpf(float(value))
        for value, weight in zip(b_ub, mu):
            if weight != 0.0:
                total -= mp.mpf(float(weight)) * mp.mpf(float(value))
        reduced_f = c + (a_eq.T @ y if a_eq is not None else 0.0) + a_ub.T @ mu
        for i in range(len(c)):
            # Recompute the reduced cost for index i in high precision.
            r = mp.mpf(float(c[i]))
            if a_eq is not None:
                for k, weight in enumerate(y):
                    if a_eq[k, i] != 0.0:
                        r += mp.mpf(float(weight)) * mp.mpf(float(a_eq[k, i]))
            for k, weight in enumerate(mu):
                if weight != 0.0 and a_ub[k, i] != 0.0:
                    r += mp.mpf(float(weight)) * mp.mpf(float(a_ub[k, i]))
            lo = mp.mpf(float(lower[i]))
            hi = mp.mpf(float(upper[i]))
            total += min(r * lo, r * hi)
        _ = reduced_f  # float screen only; the mp path above is authoritative
        return float(total)


def certified_min(
    c: np.ndarray,
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    bounds: list[tuple[float, float]],
    a_eq: np.ndarray | None = None,
    b_eq: np.ndarray | None = None,
) -> CertifiedLP:
    """Minimise with certification. All bounds must be finite."""
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        raise ValueError("certified_min requires finite box bounds")

    kwargs = dict(
        A_ub=a_ub, b_ub=b_ub, bounds=bounds, options={
            "dual_feasibility_tolerance": 1e-9,
            "primal_feasibility_tolerance": 1e-9,
        },
    )
    if a_eq is not None:
        kwargs["A_eq"] = a_eq
        kwargs["b_eq"] = b_eq
    primary = linprog(c, method="highs-ds", **kwargs)
    if not primary.success:
        if primary.status == 2:
            raise LPInfeasible(
                "declared class rejects the observed spectrum: "
                f"{primary.message}"
            )
        raise RuntimeError(f"linear programme failed: {primary.message}")
    secondary = linprog(c, method="highs-ipm", **kwargs)
    agreement = abs(primary.fun - secondary.fun) if secondary.success else float("inf")

    y = np.asarray(primary.eqlin.marginals) if a_eq is not None else np.zeros(0)
    mu = np.asarray(primary.ineqlin.marginals)
    best = -np.inf
    best_pair = None
    for ys in (1.0, -1.0):
        for ms in (1.0, -1.0):
            candidate = _dual_value_float(
                c, a_ub, b_ub, a_eq, b_eq if a_eq is not None else None,
                lower, upper, ys * y, ms * mu,
            )
            if candidate > best:
                best = candidate
                best_pair = (ys * y, ms * mu)
    certified = _dual_value_mp(
        c, a_ub, b_ub, a_eq, b_eq if a_eq is not None else None,
        lower, upper, best_pair[0], best_pair[1],
    )
    return CertifiedLP(
        value=float(primary.fun),
        x=np.asarray(primary.x),
        certified_bound=certified,
        dual_gap=abs(float(primary.fun) - certified),
        solver_agreement=agreement,
        status=str(primary.status),
    )


def certified_interval(c, a_ub, b_ub, bounds, a_eq=None, b_eq=None):
    """Certified [min, max] of c'x: (lower CertifiedLP, upper CertifiedLP).

    The upper endpoint is the negated minimisation of -c; its certified outer
    bound is -certified_bound of that problem.
    """
    low = certified_min(c, a_ub, b_ub, bounds, a_eq, b_eq)
    negated = certified_min(-np.asarray(c), a_ub, b_ub, bounds, a_eq, b_eq)
    high = CertifiedLP(
        value=-negated.value,
        x=negated.x,
        certified_bound=-negated.certified_bound,
        dual_gap=negated.dual_gap,
        solver_agreement=negated.solver_agreement,
        status=negated.status,
    )
    return low, high

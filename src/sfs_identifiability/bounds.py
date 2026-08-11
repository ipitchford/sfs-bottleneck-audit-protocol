"""Certified discretised bounds under declared complexity restrictions.

The moment equality system is numerically rank-deficient on fine grids (the
exponential kernels are nearly dependent), so requiring every raw equality at
solver tolerance can falsely report infeasibility. This module therefore:

1. rank-reveals the row-scaled equality system by singular value decomposition
   at a declared relative tolerance, and imposes the retained directions as an
   orthonormal (condition-one) equality block;
2. verifies a feasible witness (the reference history) against the reduced
   system before solving, and verifies its objective lies inside the returned
   interval afterwards;
3. solves each endpoint with two different HiGHS algorithms (dual simplex and
   interior point) and reports their agreement;
4. evaluates solver-independent Lagrangian dual certificates in high-precision
   arithmetic. Any multiplier vector yields a valid bound by weak duality, so
   the certified enclosure does not rely on the solver's internal state.

Dropping near-null equality directions relaxes the feasible set, so the
reported interval contains the sharp interval of the fully constrained grid
problem: the certificates are conservative outer bounds for the declared
discrete class. The compatibility of every dropped direction with the target
moments is reported alongside.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mpmath as mp
import numpy as np
from scipy.optimize import linprog

from .operators import moment_grid_operator

_CERTIFICATE_DPS = 50


@dataclass(frozen=True)
class BoundResult:
    lower: float
    upper: float
    lower_history: np.ndarray
    upper_history: np.ndarray
    edges: np.ndarray
    interval: tuple[float, float]
    tv_budget: float
    lower_bound: float
    upper_bound: float
    equality_residual_lower: float
    equality_residual_upper: float
    retained_rank: int = 0
    nominal_rank: int = 0
    rank_tolerance: float = 0.0
    singular_values: np.ndarray = field(default_factory=lambda: np.array([]))
    dropped_compatibility: float = 0.0
    reduced_condition: float = 1.0
    witness_residual: float = 0.0
    witness_objective: float = 0.0
    certified_lower: float = float("-inf")
    certified_upper: float = float("inf")
    dual_gap_lower: float = float("inf")
    dual_gap_upper: float = float("inf")
    solver_agreement_lower: float = float("inf")
    solver_agreement_upper: float = float("inf")
    solver_status: str = ""


def _dual_bound(
    c: np.ndarray,
    a_eq: np.ndarray,
    b_eq: np.ndarray,
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    y: np.ndarray,
    mu: np.ndarray,
) -> float:
    """Valid lower bound on min c'x by weak Lagrangian duality, in mpmath.

    L(x, y, mu) = c'x + y'(A_eq x - b_eq) + mu'(A_ub x - b_ub) with mu >= 0
    under-estimates c'x for every feasible x, so its box minimum is a bound for
    any multipliers. All bound arrays must be finite.
    """
    with mp.workdps(_CERTIFICATE_DPS):
        mu_clipped = np.maximum(mu, 0.0)
        y_mp = [mp.mpf(float(v)) for v in y]
        mu_mp = [mp.mpf(float(v)) for v in mu_clipped]
        total = mp.mpf(0)
        for value, weight in zip(b_eq, y_mp):
            total -= weight * mp.mpf(float(value))
        for value, weight in zip(b_ub, mu_mp):
            total -= weight * mp.mpf(float(value))
        for i in range(len(c)):
            reduced = mp.mpf(float(c[i]))
            for r, weight in enumerate(y_mp):
                reduced += weight * mp.mpf(float(a_eq[r, i]))
            for r, weight in enumerate(mu_mp):
                if a_ub[r, i] != 0.0:
                    reduced += weight * mp.mpf(float(a_ub[r, i]))
            lo = mp.mpf(float(lower[i]))
            hi = mp.mpf(float(upper[i]))
            total += min(reduced * lo, reduced * hi)
        return float(total)


def _best_dual_bound(c, a_eq, b_eq, a_ub, b_ub, lower, upper, y, mu) -> float:
    """Try both sign conventions for the solver multipliers, keep the best.

    Every candidate is a valid bound, so the maximum is a valid bound.
    """
    candidates = []
    for y_sign in (1.0, -1.0):
        for mu_sign in (1.0, -1.0):
            candidates.append(
                _dual_bound(
                    c, a_eq, b_eq, a_ub, b_ub, lower, upper,
                    y_sign * y, mu_sign * mu,
                )
            )
    return max(candidates)


def interval_bounds_with_tv(
    n: int,
    edges: np.ndarray,
    reference_history: np.ndarray,
    interval: tuple[float, float],
    *,
    tv_budget: float,
    lower_bound: float = 0.05,
    upper_bound: float = 20.0,
    rank_tolerance: float = 1e-9,
    witness_tolerance: float = 1e-6,
) -> BoundResult:
    """Certified bounds on an interval average under exact reference moments.

    The tail after the final edge is assumed identical across candidate
    histories, so it cancels from the moment equalities. The equality system is
    rank-reduced at ``rank_tolerance`` (relative to the largest singular
    value); the returned interval is sharp for the reduced system and contains
    the sharp interval of the fully constrained grid problem. Certified outer
    endpoints from high-precision dual certificates are reported alongside the
    solver optima.
    """
    edges = np.asarray(edges, dtype=float)
    reference = np.asarray(reference_history, dtype=float)
    cells = len(edges) - 1
    if reference.shape != (cells,):
        raise ValueError("reference_history must have one value per grid cell")
    if tv_budget < 0:
        raise ValueError("tv_budget must be non-negative")
    left, right = interval
    if not (edges[0] <= left < right <= edges[-1]):
        raise ValueError("interval must lie within the grid")

    _, operator, _ = moment_grid_operator(n, edges)
    target_moments = operator @ reference
    row_scale = np.max(np.abs(operator), axis=1)
    scaled_operator = operator / row_scale[:, None]
    scaled_target = target_moments / row_scale

    u_mat, singular, vt_mat = np.linalg.svd(scaled_operator, full_matrices=False)
    nominal_rank = len(singular)
    retained = int(np.sum(singular > singular[0] * rank_tolerance))
    reduced_rows = vt_mat[:retained, :]
    # The target moments are constructed from the reference, so the reduced
    # right-hand side V_r' reference equals U_r' b / s_r exactly; forming it on
    # the V side avoids amplifying float rounding by 1/s at the retention edge.
    reduced_target = reduced_rows @ reference
    svd_side_target = (u_mat[:, :retained].T @ scaled_target) / singular[:retained]
    dropped_compatibility = (
        float(np.max(np.abs(u_mat[:, retained:].T @ scaled_target)))
        if retained < nominal_rank
        else 0.0
    )

    witness_residual = float(np.max(np.abs(svd_side_target - reduced_target)))
    if witness_residual > witness_tolerance:
        raise RuntimeError(
            "U-side and V-side reduced targets disagree beyond tolerance: "
            f"{witness_residual:.3e} exceeds {witness_tolerance:.3e}"
        )

    differences = cells - 1
    variables = cells + differences
    equality = np.zeros((retained, variables))
    equality[:, :cells] = reduced_rows

    inequalities: list[np.ndarray] = []
    rhs: list[float] = []
    for j in range(differences):
        row = np.zeros(variables)
        row[j + 1] = 1.0
        row[j] = -1.0
        row[cells + j] = -1.0
        inequalities.append(row)
        rhs.append(0.0)

        row = np.zeros(variables)
        row[j] = 1.0
        row[j + 1] = -1.0
        row[cells + j] = -1.0
        inequalities.append(row)
        rhs.append(0.0)

    row = np.zeros(variables)
    row[cells:] = 1.0
    inequalities.append(row)
    rhs.append(tv_budget)
    a_ub = np.asarray(inequalities)
    b_ub = np.asarray(rhs)

    overlap = np.maximum(
        0.0, np.minimum(edges[1:], right) - np.maximum(edges[:-1], left)
    )
    target_weights = overlap / (right - left)
    objective = np.concatenate([target_weights, np.zeros(differences)])
    # Explicit slack cap at tv_budget is implied by the aggregate row, so it
    # changes nothing while keeping the box compact for the dual certificate.
    box = [(lower_bound, upper_bound)] * cells + [(0.0, tv_budget)] * differences
    box_lower = np.array([b[0] for b in box])
    box_upper = np.array([b[1] for b in box])

    def solve(cost: np.ndarray, method: str):
        return linprog(
            cost,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=equality,
            b_eq=reduced_target,
            bounds=box,
            method=method,
            options={
                "dual_feasibility_tolerance": 1e-9,
                "primal_feasibility_tolerance": 1e-9,
            },
        )

    minimum = solve(objective, "highs-ds")
    maximum = solve(-objective, "highs-ds")
    if not minimum.success or not maximum.success:
        raise RuntimeError(
            f"linear programme failed: min={minimum.message}; max={maximum.message}"
        )
    minimum_ipm = solve(objective, "highs-ipm")
    maximum_ipm = solve(-objective, "highs-ipm")
    agreement_lower = (
        abs(minimum.fun - minimum_ipm.fun) if minimum_ipm.success else float("inf")
    )
    agreement_upper = (
        abs(maximum.fun - maximum_ipm.fun) if maximum_ipm.success else float("inf")
    )

    certified_lower = _best_dual_bound(
        objective, equality, reduced_target, a_ub, b_ub,
        box_lower, box_upper,
        np.asarray(minimum.eqlin.marginals),
        np.asarray(minimum.ineqlin.marginals),
    )
    certified_upper = -_best_dual_bound(
        -objective, equality, reduced_target, a_ub, b_ub,
        box_lower, box_upper,
        np.asarray(maximum.eqlin.marginals),
        np.asarray(maximum.ineqlin.marginals),
    )

    lower_history = minimum.x[:cells]
    upper_history = maximum.x[:cells]
    residual_lower = float(np.max(np.abs(operator @ lower_history - target_moments)))
    residual_upper = float(np.max(np.abs(operator @ upper_history - target_moments)))
    witness_objective = float(objective[:cells] @ reference)
    lower_value = float(minimum.fun)
    upper_value = float(-maximum.fun)
    tol = 1e-7
    if not (lower_value - tol <= witness_objective <= upper_value + tol):
        raise RuntimeError(
            "feasible witness objective lies outside the returned interval: "
            f"{witness_objective} not in [{lower_value}, {upper_value}]"
        )

    return BoundResult(
        lower=lower_value,
        upper=upper_value,
        lower_history=lower_history,
        upper_history=upper_history,
        edges=edges,
        interval=interval,
        tv_budget=tv_budget,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        equality_residual_lower=residual_lower,
        equality_residual_upper=residual_upper,
        retained_rank=retained,
        nominal_rank=nominal_rank,
        rank_tolerance=rank_tolerance,
        singular_values=singular,
        dropped_compatibility=dropped_compatibility,
        reduced_condition=float(singular[0] / singular[retained - 1]),
        witness_residual=witness_residual,
        witness_objective=witness_objective,
        certified_lower=certified_lower,
        certified_upper=certified_upper,
        dual_gap_lower=abs(lower_value - certified_lower),
        dual_gap_upper=abs(certified_upper - upper_value),
        solver_agreement_lower=agreement_lower,
        solver_agreement_upper=agreement_upper,
        solver_status=(
            f"min: {minimum.status}/{minimum.message.splitlines()[0]}; "
            f"max: {maximum.status}/{maximum.message.splitlines()[0]}"
        ),
    )

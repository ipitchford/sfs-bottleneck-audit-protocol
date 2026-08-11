"""Empirical constraint sets and certified functionals for observed spectra.

The observed data are normalised class proportions q with declared confidence
half-widths delta. Writing xi(h) = G h + t for the expected branch spectrum
(G, t from the certified v0.2.0 operators, unit deep-time tail) and
T(h) = 1' xi(h) for its total, the constraint

    (q_i - delta_i) T(h) <= xi_i(h) <= (q_i + delta_i) T(h)

is linear in h, so no absolute mutation-rate scale is ever fixed: h is
measured relative to the deep-time (tail) level, and the spectrum enters only
through proportions. Window averages of h are linear functionals (certified
LPs); the depression ratio of two window averages is a linear-fractional
programme, made linear by the Charnes-Cooper transform, so the same dual
certificates apply.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sfs_identifiability.operators import direct_sfs_grid_operator

from .certify import CertifiedLP, certified_interval


def load_fitcoal_row(path: str | Path) -> tuple[np.ndarray, int]:
    """Single whitespace row of counts for classes 1..n-1 (Hu/Deng format)."""
    values = np.loadtxt(str(path), dtype=float).ravel()
    if values.ndim != 1 or len(values) < 3 or np.any(values < 0):
        raise ValueError("expected one non-negative row of class counts")
    return values, len(values) + 1


def load_column_with_monomorphic(path: str | Path) -> tuple[np.ndarray, int]:
    """One count per line for classes 0..n (Cousins format); strips 0 and n."""
    values = np.loadtxt(str(path), dtype=float).ravel()
    if values.ndim != 1 or len(values) < 5 or np.any(values < 0):
        raise ValueError("expected one non-negative count per line")
    return values[1:-1], len(values) - 1


@dataclass(frozen=True)
class EmpiricalConstraintSet:
    sample_size: int
    edges: np.ndarray
    cells: int
    a_ub: np.ndarray
    b_ub: np.ndarray
    bounds: list
    proportions: np.ndarray
    half_widths: np.ndarray
    segregating_sites: float
    z_score: float
    overdispersion: float
    tv_budget: float
    box_lower: float
    box_upper: float

    @classmethod
    def build(
        cls,
        counts: np.ndarray,
        n: int,
        edges: np.ndarray,
        *,
        z_score: float,
        overdispersion: float,
        tv_budget: float,
        box_lower: float = 0.05,
        box_upper: float = 20.0,
    ) -> "EmpiricalConstraintSet":
        counts = np.asarray(counts, dtype=float)
        if counts.shape != (n - 1,):
            raise ValueError("counts must have n-1 entries")
        total = float(counts.sum())
        if total <= 0:
            raise ValueError("counts must have positive total")
        q = counts / total
        delta = z_score * np.sqrt(overdispersion * q * (1.0 - q) / total)

        edges = np.asarray(edges, dtype=float)
        cells = len(edges) - 1
        operator, tail = direct_sfs_grid_operator(n, edges)
        column_totals = operator.sum(axis=0)
        tail_total = float(tail.sum())

        slacks = cells - 1
        variables = cells + slacks
        rows: list[np.ndarray] = []
        rhs: list[float] = []
        for i in range(n - 1):
            # xi_i(h) >= (q_i - delta_i) T(h)
            row = np.zeros(variables)
            row[:cells] = (q[i] - delta[i]) * column_totals - operator[i]
            rows.append(row)
            rhs.append(tail[i] - (q[i] - delta[i]) * tail_total)
            # xi_i(h) <= (q_i + delta_i) T(h)
            row = np.zeros(variables)
            row[:cells] = operator[i] - (q[i] + delta[i]) * column_totals
            rows.append(row)
            rhs.append((q[i] + delta[i]) * tail_total - tail[i])
        for j in range(slacks):
            row = np.zeros(variables)
            row[j + 1] = 1.0
            row[j] = -1.0
            row[cells + j] = -1.0
            rows.append(row)
            rhs.append(0.0)
            row = np.zeros(variables)
            row[j] = 1.0
            row[j + 1] = -1.0
            row[cells + j] = -1.0
            rows.append(row)
            rhs.append(0.0)
        row = np.zeros(variables)
        row[cells:] = 1.0
        rows.append(row)
        rhs.append(tv_budget)

        bounds = [(box_lower, box_upper)] * cells + [(0.0, tv_budget)] * slacks
        return cls(
            sample_size=n,
            edges=edges,
            cells=cells,
            a_ub=np.asarray(rows),
            b_ub=np.asarray(rhs),
            bounds=bounds,
            proportions=q,
            half_widths=delta,
            segregating_sites=total,
            z_score=z_score,
            overdispersion=overdispersion,
            tv_budget=tv_budget,
            box_lower=box_lower,
            box_upper=box_upper,
        )

    def window_weights(self, window: tuple[float, float]) -> np.ndarray:
        left, right = window
        if not (self.edges[0] <= left < right <= self.edges[-1]):
            raise ValueError("window must lie within the grid")
        overlap = np.maximum(
            0.0,
            np.minimum(self.edges[1:], right) - np.maximum(self.edges[:-1], left),
        )
        return overlap / (right - left)

    def window_average_interval(
        self, window: tuple[float, float]
    ) -> tuple[CertifiedLP, CertifiedLP]:
        weights = self.window_weights(window)
        objective = np.concatenate([weights, np.zeros(self.cells - 1)])
        return certified_interval(objective, self.a_ub, self.b_ub, self.bounds)

    def depression_ratio_interval(
        self,
        claim_window: tuple[float, float],
        reference_window: tuple[float, float],
    ) -> tuple[CertifiedLP, CertifiedLP]:
        """Certified bounds on (claim-window mean h) / (reference-window mean h).

        Charnes-Cooper: with original variables zvec (h then slacks) and
        constraints A zvec <= d, box lo <= zvec <= hi, substitute
        y = zvec * w, w = 1 / (b' h) > 0. All constraints become homogeneous
        rows in (y, w); the denominator becomes the equality b' y_h = 1; and
        the objective becomes a' y_h. The denominator is at least the h box
        floor, so w <= 1 / box_lower keeps every transformed bound finite.
        """
        a_weights = self.window_weights(claim_window)
        b_weights = self.window_weights(reference_window)
        variables = self.cells + (self.cells - 1)
        w_max = 1.0 / self.box_lower

        rows: list[np.ndarray] = []
        rhs: list[float] = []
        # Original inequality rows: A y - d w <= 0.
        for row, d in zip(self.a_ub, self.b_ub):
            new = np.concatenate([row, [-d]])
            rows.append(new)
            rhs.append(0.0)
        # Original box rows become homogeneous in (y, w).
        for j in range(self.cells):
            new = np.zeros(variables + 1)
            new[j] = 1.0
            new[-1] = -self.box_upper
            rows.append(new)
            rhs.append(0.0)
            new = np.zeros(variables + 1)
            new[j] = -1.0
            new[-1] = self.box_lower
            rows.append(new)
            rhs.append(0.0)
        for j in range(self.cells, variables):
            new = np.zeros(variables + 1)
            new[j] = 1.0
            new[-1] = -self.tv_budget
            rows.append(new)
            rhs.append(0.0)

        a_eq = np.zeros((1, variables + 1))
        a_eq[0, : self.cells] = b_weights
        b_eq = np.array([1.0])

        objective = np.zeros(variables + 1)
        objective[: self.cells] = a_weights

        bounds = (
            [(0.0, self.box_upper * w_max)] * self.cells
            + [(0.0, self.tv_budget * w_max)] * (self.cells - 1)
            + [(0.0, w_max)]
        )
        return certified_interval(
            objective, np.asarray(rows), np.asarray(rhs), bounds, a_eq, b_eq
        )

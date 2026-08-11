"""Successor protocol A2: linkage-aware error model (frozen literals).

Additive module; no frozen A1 file is modified. The error-model FORM and
parameter ladders are data-informed: they derive, by the rules stated here,
from the published Release B rejection diagnostics (doi 10.5281/zenodo.21893572).
No certified interval endpoint on real data existed before this module was
frozen; Release B produced only class rejections. The functionals, window
family, grid, box and reference windows remain exactly those frozen in A1.

Error-model components and their derivation rules:

1. Ancestral misidentification: a declared fraction e of variants carries a
   flipped ancestral state, mapping derived class i to n - i. The expected
   observed spectrum is the linear transform (1-e) xi + e flip(xi), so every
   programme remains a certified LP. Ladder E_LADDER, bracketing published
   mispolarisation estimates; e = 0 retains the A1 model.
2. Class masking: the 'interior' variant drops the constraint rows of classes
   1, 2, n-2, n-1 (the two contamination channels Release B identified);
   'all' retains every class.
3. Linkage overdispersion by block rule: phi = S / n_eff, with n_eff the
   declared count of approximately independent autosomal blocks for block
   lengths {3 Mb, 1 Mb, 0.3 Mb}. This is a recombination-scale rule, not a
   fitted parameter.
"""

from __future__ import annotations

import numpy as np

from sfs_identifiability.operators import direct_sfs_grid_operator

from .empirical import EmpiricalConstraintSet

E_LADDER = (0.0, 0.005, 0.02)
CLASS_VARIANTS = ("all", "interior")
BLOCK_NEFF_LADDER = (1000, 3000, 10000)
TV_BUDGETS_A2 = (0.5, 5.0, 50.0)


def mispolarised_operator(
    operator: np.ndarray, tail: np.ndarray, e: float
) -> tuple[np.ndarray, np.ndarray]:
    """Expected observed spectrum under ancestral-flip fraction e (class i -> n-i)."""
    if not 0.0 <= e < 0.5:
        raise ValueError("e must lie in [0, 0.5)")
    return (
        (1.0 - e) * operator + e * operator[::-1, :],
        (1.0 - e) * tail + e * tail[::-1],
    )


def constraint_row_mask(n: int, variant: str) -> np.ndarray:
    """Boolean mask over classes 1..n-1 selecting which constraint rows to keep."""
    keep = np.ones(n - 1, dtype=bool)
    if variant == "interior":
        keep[[0, 1, n - 3, n - 2]] = False
    elif variant != "all":
        raise ValueError(f"unknown class variant: {variant}")
    return keep


def overdispersion_for(segregating_sites: float, n_eff: int) -> float:
    """Block rule: the multinomial total deflates to n_eff independent units."""
    return max(1.0, float(segregating_sites) / float(n_eff))


def build_a2_constraint_set(
    counts: np.ndarray,
    n: int,
    edges: np.ndarray,
    *,
    e: float,
    variant: str,
    n_eff: int,
    tv_budget: float,
    z_score: float,
    box_lower: float,
    box_upper: float,
) -> EmpiricalConstraintSet:
    """A1 constraint construction with the A2 error model, as one LP system."""
    counts = np.asarray(counts, dtype=float)
    if counts.shape != (n - 1,):
        raise ValueError("counts must have n-1 entries")
    total = float(counts.sum())
    q = counts / total
    phi = overdispersion_for(total, n_eff)
    delta = z_score * np.sqrt(phi * q * (1.0 - q) / total)

    edges = np.asarray(edges, dtype=float)
    cells = len(edges) - 1
    base_operator, base_tail = direct_sfs_grid_operator(n, edges)
    operator, tail = mispolarised_operator(base_operator, base_tail, e)
    column_totals = operator.sum(axis=0)
    tail_total = float(tail.sum())
    keep = constraint_row_mask(n, variant)

    slacks = cells - 1
    variables = cells + slacks
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for i in range(n - 1):
        if not keep[i]:
            continue
        row = np.zeros(variables)
        row[:cells] = (q[i] - delta[i]) * column_totals - operator[i]
        rows.append(row)
        rhs.append(tail[i] - (q[i] - delta[i]) * tail_total)
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
    return EmpiricalConstraintSet(
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
        overdispersion=phi,
        tv_budget=tv_budget,
        box_lower=box_lower,
        box_upper=box_upper,
    )

"""Synthetic-only validation and controls for the A2 error model."""

from __future__ import annotations

import numpy as np
import pytest

from sfs_identifiability.operators import direct_sfs_grid_operator
from bottleneck_audit import LPInfeasible
from bottleneck_audit import windows as W
from bottleneck_audit.protocol_a2 import (
    build_a2_constraint_set,
    constraint_row_mask,
    mispolarised_operator,
    overdispersion_for,
)

N = 60
EDGES = np.linspace(0.0, 2.0, 31)
CENTRES = 0.5 * (EDGES[:-1] + EDGES[1:])
SITES = 2_000_000


def _mispolarised_counts(history: np.ndarray, e: float) -> np.ndarray:
    operator, tail = direct_sfs_grid_operator(N, EDGES)
    op_e, tail_e = mispolarised_operator(operator, tail, e)
    spectrum = op_e @ history + tail_e
    return np.round(spectrum / spectrum.sum() * SITES)


def _build(counts, *, e, variant, n_eff, tv_budget=5.0):
    return build_a2_constraint_set(
        counts, N, EDGES, e=e, variant=variant, n_eff=n_eff,
        tv_budget=tv_budget, z_score=W.Z_SCORE,
        box_lower=W.BOX[0], box_upper=W.BOX[1],
    )


def test_flip_operator_preserves_totals_and_reduces_to_identity() -> None:
    operator, tail = direct_sfs_grid_operator(N, EDGES)
    op0, tail0 = mispolarised_operator(operator, tail, 0.0)
    assert np.array_equal(op0, operator) and np.array_equal(tail0, tail)
    op_e, tail_e = mispolarised_operator(operator, tail, 0.02)
    assert np.allclose(op_e.sum(axis=0), operator.sum(axis=0))
    assert np.isclose(tail_e.sum(), tail.sum())


def test_interior_mask_drops_exactly_four_classes() -> None:
    keep = constraint_row_mask(N, "interior")
    assert keep.sum() == N - 5
    keep_all = constraint_row_mask(N, "all")
    assert keep_all.all()


def test_block_rule_is_monotone() -> None:
    assert overdispersion_for(3_000_000, 3000) == pytest.approx(1000.0)
    assert overdispersion_for(100, 3000) == 1.0  # floor at multinomial


def test_matched_e_is_feasible_and_covers_witness() -> None:
    history = np.where((CENTRES >= 0.55) & (CENTRES < 0.65), 0.08, 1.0)
    counts = _mispolarised_counts(history, 0.02)
    constraint_set = _build(counts, e=0.02, variant="all", n_eff=3000)
    low, high = constraint_set.depression_ratio_interval((0.55, 0.65), W.REFERENCE_WINDOW)
    weights_claim = constraint_set.window_weights((0.55, 0.65))
    weights_ref = constraint_set.window_weights(W.REFERENCE_WINDOW)
    truth = float((weights_claim @ history) / (weights_ref @ history))
    assert low.certified_bound - 1e-9 <= truth <= high.certified_bound + 1e-9


def test_control_unmodelled_mispolarisation_detected_at_tight_noise() -> None:
    # With phi floored at 1 (tiny n_eff denominator effect removed by using a
    # huge n_eff) and e = 0 in the model, an e = 0.02 spectrum must be
    # rejected: the flip signature is not absorbable.
    history = np.ones(len(CENTRES))
    counts = _mispolarised_counts(history, 0.02)
    constraint_set = _build(counts, e=0.0, variant="all", n_eff=10**9, tv_budget=0.5)
    with pytest.raises(LPInfeasible):
        constraint_set.window_average_interval((0.8, 1.0))


def test_interior_variant_absorbs_tail_contamination() -> None:
    history = np.ones(len(CENTRES))
    counts = _mispolarised_counts(history, 0.005)
    constraint_set = _build(counts, e=0.0, variant="interior", n_eff=3000, tv_budget=0.5)
    low, high = constraint_set.window_average_interval((0.8, 1.0))
    assert low.certified_bound <= 1.0 <= high.certified_bound

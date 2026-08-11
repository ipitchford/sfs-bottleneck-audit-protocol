"""Machinery tests plus mutation and negative controls (synthetic only)."""

from __future__ import annotations

import numpy as np
import pytest

from sfs_identifiability.independent_forward import sfs_weight_matrix_exact
from sfs_identifiability.operators import direct_sfs_grid_operator
from bottleneck_audit import (
    EmpiricalConstraintSet,
    claim_window_family,
    load_column_with_monomorphic,
    load_fitcoal_row,
)
from bottleneck_audit import windows as W

N = 60  # smaller sample size keeps the test suite fast
EDGES = np.linspace(0.0, 2.0, 31)
CENTRES = 0.5 * (EDGES[:-1] + EDGES[1:])
SITES = 2_000_000


def _counts_for(history: np.ndarray) -> np.ndarray:
    operator, tail = direct_sfs_grid_operator(N, EDGES)
    spectrum = operator @ history + tail
    return np.round(spectrum / spectrum.sum() * SITES)


def _build(counts, *, tv_budget=0.5, overdispersion=1.0):
    return EmpiricalConstraintSet.build(
        counts, N, EDGES,
        z_score=W.Z_SCORE, overdispersion=overdispersion, tv_budget=tv_budget,
        box_lower=W.BOX[0], box_upper=W.BOX[1],
    )


def test_vendored_package_intact() -> None:
    # The exact rational constant-history identity gates the vendored copy.
    sfs_weight_matrix_exact(12)


def test_loaders_round_trip(tmp_path) -> None:
    counts = np.arange(1.0, 60.0)
    row_file = tmp_path / "row.txt"
    row_file.write_text(" ".join(str(int(v)) for v in counts) + "\n")
    loaded, n = load_fitcoal_row(row_file)
    assert n == 60 and np.array_equal(loaded, counts)

    column_file = tmp_path / "col.txt"
    column_file.write_text(
        "\n".join(str(int(v)) for v in [999] + list(counts) + [7]) + "\n"
    )
    loaded, n = load_column_with_monomorphic(column_file)
    assert n == 60 and np.array_equal(loaded, counts)


def test_loader_rejects_malformed(tmp_path) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_text("1 -2 3\n")
    with pytest.raises(ValueError):
        load_fitcoal_row(bad)


def test_window_family_is_declared_and_inside_grid() -> None:
    family = claim_window_family()
    assert len(family) == 12
    for entry in family:
        left, right = entry["window"]
        assert 0.0 < left < right
        assert right < 2.0  # every declared mapping fits the frozen horizon


def test_witness_coverage_constant() -> None:
    history = np.ones(len(CENTRES))
    constraint_set = _build(_counts_for(history))
    low, high = constraint_set.depression_ratio_interval((0.8, 1.0), W.REFERENCE_WINDOW)
    assert low.certified_bound - 1e-9 <= 1.0 <= high.certified_bound + 1e-9
    assert low.dual_gap < 1e-6 and high.dual_gap < 1e-6


def test_witness_coverage_bottleneck() -> None:
    history = np.where((CENTRES >= 0.8) & (CENTRES < 1.0), 0.08, 1.0)
    constraint_set = _build(_counts_for(history), tv_budget=5.0)
    weights_claim = constraint_set.window_weights((0.8, 1.0))
    weights_ref = constraint_set.window_weights(W.REFERENCE_WINDOW)
    truth = float((weights_claim @ history) / (weights_ref @ history))
    low, high = constraint_set.depression_ratio_interval((0.8, 1.0), W.REFERENCE_WINDOW)
    assert low.certified_bound - 1e-9 <= truth <= high.certified_bound + 1e-9


def test_window_average_contains_generator_value() -> None:
    history = np.where(CENTRES < 0.2, 3.0, 1.0)
    constraint_set = _build(_counts_for(history), tv_budget=5.0)
    low, high = constraint_set.window_average_interval((0.0, 0.2))
    assert low.certified_bound - 1e-9 <= 3.0 <= high.certified_bound + 1e-9


def test_control_scrambled_spectrum_detected() -> None:
    from bottleneck_audit import LPInfeasible

    history = np.ones(len(CENTRES))
    counts = _counts_for(history)
    scrambled = counts[::-1].copy()  # reversed spectrum is wildly infeasible
    constraint_set = _build(scrambled)
    with pytest.raises(LPInfeasible):
        constraint_set.depression_ratio_interval((0.8, 1.0), W.REFERENCE_WINDOW)


def test_control_zero_budget_narrows_interval() -> None:
    history = np.ones(len(CENTRES))
    tight = _build(_counts_for(history), tv_budget=0.0)
    low, high = tight.depression_ratio_interval((0.8, 1.0), W.REFERENCE_WINDOW)
    assert high.certified_bound - low.certified_bound < 0.05
    assert low.certified_bound <= 1.0 <= high.certified_bound


def test_control_window_shift_changes_result() -> None:
    history = np.where((CENTRES >= 0.8) & (CENTRES < 1.0), 0.08, 1.0)
    constraint_set = _build(_counts_for(history), tv_budget=5.0)
    _, inside_high = constraint_set.window_average_interval((0.8, 1.0))
    outside_low, _ = constraint_set.window_average_interval((0.0, 0.2))
    assert inside_high.certified_bound < 1.0 or outside_low.certified_bound > 0.0
    inside_low, _ = constraint_set.window_average_interval((0.8, 1.0))
    assert abs(inside_low.certified_bound - outside_low.certified_bound) > 0.02


def test_control_wrong_length_counts_rejected() -> None:
    with pytest.raises(ValueError):
        _build(np.ones(17))

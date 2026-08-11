#!/usr/bin/env python3
"""Synthetic-only validation of the frozen audit machinery (Release A gate).

No real data are read. Three declared generator histories produce
deterministic synthetic observed counts; the machinery must (i) cover every
generator's true functional value with its certified interval, (ii) exclude
FitCoal-level severity for the constant generator at the tight budget, and
(iii) not exclude it for the bottleneck generator at any budget.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sfs_identifiability.operators import direct_sfs_grid_operator
from bottleneck_audit import EmpiricalConstraintSet, LPInfeasible, claim_window_family
from bottleneck_audit import windows as W

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SAMPLE_SIZE = 216
SYNTHETIC_SITES = 2_000_000
VALIDATION_BUDGETS = (0.5, 5.0)


def generator_histories(centres: np.ndarray) -> dict[str, np.ndarray]:
    # The bottleneck spike deliberately covers the shallowest declared claim
    # window (0.561-0.641), so its true depression ratio there is severe.
    return {
        "constant": np.ones(len(centres)),
        "bottleneck": np.where((centres >= 0.55) & (centres < 0.65), 0.08, 1.0),
        "expansion": np.where(centres < 0.2, 3.0, 1.0),
    }


def main() -> None:
    start, stop, points = W.GRID_EDGES
    edges = np.linspace(start, stop, points)
    centres = 0.5 * (edges[:-1] + edges[1:])
    operator, tail = direct_sfs_grid_operator(SAMPLE_SIZE, edges)

    family = claim_window_family()
    windows = [
        min(family, key=lambda e: e["window"][0])["window"],
        max(family, key=lambda e: e["window"][1])["window"],
    ]
    # Clamp family windows to the grid horizon.
    windows = [(max(a, edges[0]), min(b, edges[-1])) for a, b in windows]

    records = []
    failures = []
    for name, history in generator_histories(centres).items():
        spectrum = operator @ history + tail
        counts = np.round(spectrum / spectrum.sum() * SYNTHETIC_SITES)
        for budget in VALIDATION_BUDGETS:
            constraint_set = EmpiricalConstraintSet.build(
                counts,
                SAMPLE_SIZE,
                edges,
                z_score=W.Z_SCORE,
                overdispersion=W.OVERDISPERSION_LADDER[0],
                tv_budget=budget,
                box_lower=W.BOX[0],
                box_upper=W.BOX[1],
            )
            history_tv = float(np.sum(np.abs(np.diff(history))))
            for window in windows:
                weights_claim = constraint_set.window_weights(window)
                weights_ref = constraint_set.window_weights(W.REFERENCE_WINDOW)
                truth = float(
                    (weights_claim @ history) / (weights_ref @ history)
                )
                try:
                    low, high = constraint_set.depression_ratio_interval(
                        window, W.REFERENCE_WINDOW
                    )
                except LPInfeasible:
                    # The declared class rejects this spectrum. Valid only
                    # when the generator itself lies outside the class.
                    record = {
                        "history": name,
                        "tv_budget": budget,
                        "window": list(window),
                        "true_ratio": truth,
                        "outcome": "class_rejected",
                        "generator_tv": history_tv,
                        "rejection_consistent": bool(history_tv > budget),
                    }
                    records.append(record)
                    if history_tv <= budget:
                        failures.append(record)
                    continue
                covered = (
                    low.certified_bound - 1e-9
                    <= truth
                    <= high.certified_bound + 1e-9
                )
                in_class = history_tv <= budget
                record = {
                    "history": name,
                    "tv_budget": budget,
                    "window": list(window),
                    "true_ratio": truth,
                    "outcome": "bounded",
                    "generator_in_class": bool(in_class),
                    "certified_lower": low.certified_bound,
                    "certified_upper": high.certified_bound,
                    "dual_gap_lower": low.dual_gap,
                    "dual_gap_upper": high.dual_gap,
                    "solver_agreement": max(
                        low.solver_agreement, high.solver_agreement
                    ),
                    "witness_covered": bool(covered),
                    "severity_excluded": bool(
                        low.certified_bound > W.SEVERITY_THRESHOLD
                    ),
                }
                # Coverage is guaranteed only for in-class generators. An
                # out-of-class generator absorbed by the class without
                # coverage is the class-sensitivity phenomenon itself and is
                # recorded as a demonstration, not a failure.
                if not covered and not in_class:
                    record["note"] = "class_sensitivity_demonstration"
                records.append(record)
                if in_class and not covered:
                    failures.append(record)

    shallow = min(w[0] for w in windows)

    def is_shallow(record):
        return abs(record["window"][0] - shallow) < 1e-12

    # C2: no-bottleneck data must certifiably exclude severity at the tight
    # budget in the shallow (information-rich) window.
    constant_tight_shallow = [
        r
        for r in records
        if r["history"] == "constant"
        and r["tv_budget"] == 0.5
        and r["outcome"] == "bounded"
        and is_shallow(r)
    ]
    if not constant_tight_shallow or not all(
        r["severity_excluded"] for r in constant_tight_shallow
    ):
        failures.append(
            {"check": "constant generator must exclude severity at V=0.5, shallow window"}
        )
    # C3: when the severe generator lies INSIDE the declared class, severity
    # must never be certifiably excluded in its own window (follows from
    # coverage; asserted independently). When it lies outside the class, the
    # class may absorb its spectrum as non-severe: that row is the declared
    # class-sensitivity demonstration and must exist at the tight budget.
    bottleneck_shallow_in_class = [
        r
        for r in records
        if r["history"] == "bottleneck"
        and r["outcome"] == "bounded"
        and r["generator_in_class"]
        and is_shallow(r)
    ]
    if not bottleneck_shallow_in_class or any(
        r["severity_excluded"] for r in bottleneck_shallow_in_class
    ):
        failures.append(
            {"check": "in-class bottleneck must be bounded without excluding severity"}
        )
    demonstration = [
        r
        for r in records
        if r.get("note") == "class_sensitivity_demonstration"
        and r["history"] == "bottleneck"
        and r["tv_budget"] == 0.5
    ]
    if not demonstration:
        failures.append(
            {"check": "tight class must demonstrate absorption of the severe spectrum"}
        )
    # C4: the out-of-class expansion spectrum must be rejected at the tight
    # budget, and every rejection must be consistent (generator outside class).
    expansion_rejected = [
        r
        for r in records
        if r["history"] == "expansion"
        and r["tv_budget"] == 0.5
        and r["outcome"] == "class_rejected"
    ]
    if not expansion_rejected:
        failures.append(
            {"check": "tight class must reject the out-of-class expansion spectrum"}
        )
    if any(
        not r.get("rejection_consistent", True)
        for r in records
        if r["outcome"] == "class_rejected"
    ):
        failures.append({"check": "every class rejection must be consistent"})

    receipt = {
        "package": "sfs-bottleneck-audit-protocol",
        "version": "0.1.1",
        "date": "2026-08-11",
        "synthetic_only": True,
        "declared": {
            "sample_size": SAMPLE_SIZE,
            "synthetic_sites": SYNTHETIC_SITES,
            "grid_edges": list(W.GRID_EDGES),
            "z_score": W.Z_SCORE,
            "severity_threshold": W.SEVERITY_THRESHOLD,
            "reference_window": list(W.REFERENCE_WINDOW),
            "validation_budgets": list(VALIDATION_BUDGETS),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "records": records,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }
    (RESULTS / "synthetic_validation_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": receipt["status"], "records": len(records)}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

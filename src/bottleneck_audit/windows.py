"""The declared claim-window family and reference window (frozen values).

The Hu et al. claim epoch is 930-813 kya. Mapping calendar years to
transformed coalescent time requires a generation time and a reference
diploid size: tau = years / (generation_time * 2 * N_ref). Both enter through
declared ladders bracketing published values, giving a 12-member window
family; every functional is evaluated across the whole family. The reference
(denominator) window for the depression ratio is fixed near the present.
These values are frozen by Release A and may not be changed in Release B.
"""

from __future__ import annotations

CLAIM_EPOCH_YEARS = (813_000.0, 930_000.0)
GENERATION_TIME_YEARS = (24.0, 26.9, 29.0)
REFERENCE_DIPLOID_SIZES = (12_500.0, 15_000.0, 20_000.0, 25_000.0)
REFERENCE_WINDOW = (0.0, 0.1)
SEVERITY_THRESHOLD = 0.1
GRID_EDGES = (0.0, 2.0, 61)  # linspace start, stop, points -> 60 cells
TV_BUDGETS = (0.5, 2.0, 5.0, 10.0, 20.0, 50.0)
OVERDISPERSION_LADDER = (1.0, 2.0, 5.0, 10.0)
Z_SCORE = 3.0
BOX = (0.05, 20.0)


def claim_window_family() -> list[dict]:
    """All 12 declared mappings of the claim epoch into coalescent time."""
    young, old = CLAIM_EPOCH_YEARS
    family = []
    for generation in GENERATION_TIME_YEARS:
        for n_ref in REFERENCE_DIPLOID_SIZES:
            scale = generation * 2.0 * n_ref
            family.append(
                {
                    "generation_time_years": generation,
                    "reference_diploid_size": n_ref,
                    "window": (young / scale, old / scale),
                }
            )
    return family

# Successor protocol A2: linkage-aware error model (frozen literals)

**Version:** protocol v0.2.0, 11 August 2026. Extends the A1 freeze
(PROTOCOL.md, tag v0.1.1-candidate, doi 10.5281/zenodo.21893436); no A1 file
is modified. Executed audit under A1: doi 10.5281/zenodo.21893572
(universal class rejection).

## Honesty statement (read first)

This protocol is **data-informed, and says so**: the error-model form and
parameter ladders derive, by the rules below, from the published Release B
rejection diagnostics computed on the same nine spectra this protocol will
be applied to. What remains genuinely pre-registered: **no certified interval
endpoint on real data has ever been computed** (Release B produced only
rejections), the functionals, window family, grid, box, reference windows
and severity threshold are unchanged from A1, and the ladders below are
fixed by recombination-scale and literature-scale rules, not by inspecting
any interval. The remaining risk — that the model family itself was chosen
after seeing which classes misfit — is disclosed, not hidden.

## Frozen A2 declarations (see src/bottleneck_audit/protocol_a2.py)

- Mispolarisation ladder: e in {0, 0.005, 0.02} (linear flip transform,
  class i -> n-i; e is not fitted).
- Class variants: {all, interior}; interior drops constraint rows for
  classes 1, 2, n-2, n-1.
- Overdispersion by block rule: phi = S / n_eff, n_eff in {1000, 3000,
  10000} (autosomal blocks of ~3, ~1, ~0.3 Mb).
- TV budgets: {0.5, 5, 50}. All other literals inherited from A1.
- Cell outcomes and reporting as in A1 (bounded with dual certificates /
  class_rejected). Data: the nine A1-declared spectra, same binding hashes.
- Pre-registered interpretation unchanged from A1 section 1; additionally,
  cells feasible only at the interior variant or e > 0 label the excluded
  classes as the contamination carriers.

## Validation (synthetic only)

tests/test_protocol_a2.py: flip-operator identities, mask and block-rule
checks, witness coverage under matched e, and two controls — an unmodelled
e = 0.02 spectrum is rejected at tight noise, and the interior variant
absorbs tail contamination. All pass before this freeze.

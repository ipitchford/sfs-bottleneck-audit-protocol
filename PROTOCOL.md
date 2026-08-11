# Frozen protocol: SFS identifiability audit of the 930 kya bottleneck claim

**Package:** sfs-bottleneck-audit-protocol v0.1.0, 11 August 2026.
**Status:** anonymous, unrefereed, producer-validated preregistration. This
document freezes every analysis choice for the executed audit (Release B).
Release B runs this pipeline byte-for-byte on the declared data; any departure
must be recorded in `DEVIATIONS.md` before results are inspected.

**Development discipline.** All machinery in this package was developed and
validated on synthetic data only (`examples/run_synthetic_validation.py`).
No real spectrum has been read by any analysis code in this repository.

## 1. Question and pre-registered interpretation

For each declared population and processing variant, compute certified
identified sets for the declared functionals under the declared classes, and
label each dispute-relevant feature per the five-label taxonomy of
sfs-identifiability-audit v0.2.0 (vendored in `src/sfs_identifiability/`).

Pre-registered interpretation: wide sets or class-dependence at plausible
budgets mean the contested severity is *class-identified* — that finding
stands on its own. A severity exclusion surviving all declared classes and
sensitivity axes would be the stronger positive finding. Neither outcome is
privileged; both are publishable under this protocol. No adjudication of
whether the bottleneck happened; no refitting of any published method.

## 2. Declared data (from T1 provenance, hashes binding)

Release B may read exactly these nine spectra, identified by sha256:

| id | source | file | sha256 (first 16) | n |
|---|---|---|---|---|
| HU-YRI | Mendeley 10.17632/xmf5r8nzrn.3 (CC BY 4.0) | SFS/1000GP/YRI.total.sfs | see t1 ledger | 216 |
| HU-ESN | same | SFS/1000GP/ESN.total.sfs | see t1 ledger | 198 |
| HU-LWK | same | SFS/1000GP/LWK.total.sfs | see t1 ledger | 198 |
| HU-MSL | same | SFS/1000GP/MSL.total.sfs | see t1 ledger | 170 |
| HU-GWD | same | SFS/1000GP/GWD.total.sfs | see t1 ledger | 226 |
| HU-CEU | same | SFS/1000GP/CEU.total.sfs | see t1 ledger | 198 |
| HU-CHB | same | SFS/1000GP/CHB.total.sfs | see t1 ledger | 206 |
| DENG-YRI | github YunDeng98/bottleneck_demography @84df4a90 (MIT) | input_sfs.txt | 0f9196e0fea88bfb | 216 |
| COUSINS-YRI | github trevorcousins/insufficient_evidence_panmictic_bottleneck @5d48038b (no licence: fetch, do not redistribute) | YRI_GRCh37_noncoding_polarized.txt | 4593acb24d42d924 | 216 |

Full hashes: `DATA_PROVENANCE.md`. The two folded (nonpolarized) Cousins
variants are excluded (the theorem is stated for unfolded spectra); their
existence is reported in the sensitivity discussion.

**Loaders (frozen):** Hu/Deng files via `load_fitcoal_row` (one row, classes
1..n-1). Cousins file via `load_column_with_monomorphic` (one count per line,
classes 0..n; classes 0 and n stripped). A file failing its loader or hash
check is dropped and recorded; it is never repaired by hand.

## 3. Declared model and constraint construction (frozen literals)

- Forward model: neutral Kingman coalescent, infinite sites, single panmictic
  population; expected unfolded branch spectrum xi(h) = G h + t from
  `direct_sfs_grid_operator`, with unit tail after the final edge (h is
  measured relative to the deep-time level).
- Grid: 60 cells, edges = linspace(0.0, 2.0, 61).
- Constraint: for every class i, (q_i - d_i) T(h) <= xi_i(h) <= (q_i + d_i) T(h),
  with q the observed proportions, T(h) the spectrum total, and
  d_i = z sqrt(phi q_i (1 - q_i) / S), S the file's total count.
- z = 3.0 (fixed). Overdispersion ladder phi in {1, 2, 5, 10}.
- Box: 0.05 <= h_j <= 20. TV budgets V in {0.5, 2, 5, 10, 20, 50}.
- Outcome semantics: each (file, phi, V) is either *bounded* (certified
  intervals reported) or *class_rejected* (`LPInfeasible`: no history in the
  declared class reproduces the spectrum within the declared tolerance).
  Class rejection is a reported scientific outcome.

## 4. Declared functionals and windows (frozen literals)

- Claim epoch 813-930 kya. Window family: tau = years / (g * 2 * N_ref) for
  g in {24, 26.9, 29} years and N_ref in {12500, 15000, 20000, 25000} —
  12 windows, all inside the grid horizon. Reference window: tau in [0, 0.1].
- F1 (headline): depression ratio = (claim-window mean h) / (reference-window
  mean h), certified via the Charnes-Cooper linear-fractional programme.
- F2: claim-window mean h (certified LP; interpreted relative to deep-time
  level).
- F3: severity exclusion — "FitCoal-severity excluded" at (file, phi, V,
  window) iff the certified lower bound of F1 exceeds 0.1.
- Every certified endpoint carries the dual certificate, dual gap, and
  dual-simplex/interior-point agreement, all reported.

## 5. Declared sensitivity axes (complete list)

Population (7 Hu files); processing (YRI three ways: HU/DENG/COUSINS);
overdispersion ladder; TV ladder; window family (12); reference-window
variant tau in [0, 0.05] reported alongside the primary [0, 0.1]. No other
axis may be added after this freeze.

## 6. Class-identified comparison

The published FitCoal and mushi YRI curves are evaluated (no refitting)
against the certified sets: their TV values locate them on the declared
budget ladder, and their F1 values are compared with the certified intervals.
Digitisation provenance for those curves must be recorded in Release B.

## 7. Validation and controls (this release)

`results/synthetic_validation_receipt.json` must show PASS with:
C1 witness coverage for every in-class generator; C2 severity exclusion for
the constant generator at V=0.5 in the shallow window; C3 in-class severe
generator bounded without severity exclusion; C4 tight-class rejection of the
out-of-class expansion spectrum, with every rejection consistent; and the
recorded class-sensitivity demonstration (severe-truth spectrum absorbed as
certifiably non-severe by the tight class). Test suite: 11 checks including
loader validation, vendored-package integrity, and mutation controls.

## 8. Kill criteria (unchanged from scope)

A data file failing hash or loader checks is dropped, not repaired. If all
African files are dropped, Release B is abandoned and the failure published
in this repository instead. Solver failure other than `LPInfeasible` on any
declared cell aborts that cell and is recorded; more than 5% aborted cells
aborts Release B.

## 9. Release B execution rule

One command per file: load, build constraint sets over the declared ladders,
evaluate F1-F3 over the window family, write one populated audit template per
file (templates in `templates/`), and one machine-readable results table.
The code may not be edited between Release A and Release B except through
`DEVIATIONS.md` entries recorded before results are inspected.

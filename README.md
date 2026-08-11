# SFS bottleneck audit — frozen protocol (Release A) v0.1.0

**Status: anonymous, unrefereed, producer-validated preregistration.**
Developed and validated on synthetic data only; no real spectrum has been
read by any analysis code in this repository. The entire package is dedicated
to the public domain under CC0 1.0 (see LICENSE); no copyright is claimed.

This package freezes every analysis choice for a pre-registered
identifiability audit of the disputed 930-813 kya human bottleneck claim,
applying the certified machinery of sfs-identifiability-audit v0.2.0
(vendored in `src/sfs_identifiability/`, unchanged) to the dispute papers'
own published spectra. The executed audit is a separate later release
(Release B) that runs this pipeline byte-for-byte; `DEVIATIONS.md` (empty at
freeze) is the only permitted change channel.

- `PROTOCOL.md` — the frozen declarations: data hashes, model, grid, ladders,
  window family, functionals, outcome semantics, kill criteria.
- `DATA_PROVENANCE.md` — binding sha256 digests and fetch rules for the nine
  declared spectra (Hu et al. CC BY 4.0; Deng MIT; Cousins no-licence
  fetch-only).
- `src/bottleneck_audit/` — new machinery: certified inequality LPs with
  50-digit dual certificates (`certify.py`), empirical constraint sets that
  never fix an absolute mutation-rate scale (`empirical.py`), and the frozen
  window family (`windows.py`). The depression ratio is a linear-fractional
  programme made linear by the Charnes-Cooper transform, so every endpoint
  keeps the same dual-certificate guarantees.
- `examples/run_synthetic_validation.py` — the synthetic-only validation and
  control run; its receipt (`results/synthetic_validation_receipt.json`)
  must show PASS for the freeze to be valid.
- `tests/` — 11 deterministic checks including loader validation, witness
  coverage, class-rejection controls and vendored-package integrity.
- `templates/` — one empty audit template per declared dataset, to be
  populated only by Release B.

## Reproduce the validation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock
PYTHONPATH=src python examples/run_synthetic_validation.py
PYTHONPATH=src pytest -q
```

## Key validation results (synthetic, deterministic)

- Certified intervals cover the true functional of every in-class generator;
  dual gaps below 1e-12; both HiGHS algorithms agree.
- A constant-history dataset certifiably **excludes** FitCoal-level severity
  (ratio <= 0.1) at the tight budget in the information-rich window.
- A dataset generated from a genuinely severe bottleneck is **absorbed** by
  the tight smoothness class as certifiably non-severe — the recorded
  class-sensitivity demonstration, which is the phenomenon at the centre of
  the published dispute.
- An out-of-class expansion spectrum is **rejected** by the tight class
  (`class_rejected`), showing spectrum/class incompatibility is detected,
  not smoothed over.

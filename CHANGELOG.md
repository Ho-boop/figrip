# Changelog

## 0.1.0 — 2026-10-09 (alpha)

- Package the initial vector-PDF extraction pipeline and OPV calculations for public use.
- Add a self-generated, ground-truth-checked J–V / outlined J–V / EQE demo.
- Add installation instructions, architecture, explicit limitations, roadmap and contribution guide.
- Add Python package metadata, dependency attribution and GitHub Actions test/build/demo checks.
- Re-run the existing suite locally: 151 passed on Windows, Python 3.14.2.
- Preserve the 2026-10-08 real-paper baseline separately: 87.0% plot recall, 58.5% fully correct among detected panels.

Twin-y automatic calibration/series assignment and second-round extraction fixes are unfinished. This release does not claim those repairs.

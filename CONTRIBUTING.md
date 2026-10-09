# Contributing

Create a virtual environment, then run:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python tools/demo.py
```

For an extraction bug, include the command, the expected number of panels/curves, actual output, and an overlay if you may share it.
Prefer a small self-generated PDF with known source data. Do not upload copyrighted papers, private library files or credentials.

Keep page/data coordinates distinct and retain defaults when extending shared dataclasses.
Add regression tests at the pipeline or module seam that reproduces the failure. Test numerical correctness, not just absence of exceptions.
For real-library comparisons, reuse `selection.json` and retain timing/error records. Synthetic test accuracy and real-paper extraction accuracy must be reported separately.

Private PDF regression tests should skip when the input is absent; tests and demos must not download papers or require accounts.

"""Generate public synthetic plots, extract them, and check against ground truth.

Run from a source checkout after `python -m pip install -e ".[dev]"`.
No private papers, API keys or model services are required.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from figrip import export, opv
from figrip.pipeline import extract

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "tests" / "corpus" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load corpus module: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "out" / "demo")
    args = parser.parse_args()
    dest = args.out.resolve()
    inputs = dest / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    make = _load("figrip_demo_corpus", "make_corpus.py")
    scoring = _load("figrip_demo_score", "score.py")
    report = {"synthetic": True, "curve_tolerance_fraction_of_axis_range": 0.005, "cases": []}
    for name in ("jv_opv", "outlined_all", "eqe"):
        truth = getattr(make, f"case_{name}")(inputs)
        pdf = inputs / truth["pdf"]
        results = extract(str(pdf))
        score = scoring.score(truth, results, tol=0.005)
        if not score["ok"]:
            raise RuntimeError(scoring.format_report(score))
        output = dest / name
        subprocess.run([sys.executable, "-m", "figrip", "extract", str(pdf), "-p", "1",
                        "-o", str(output), "--format", "all", "--preview"], check=True)
        expected = [output / f"{name}.json"]
        for i, result in enumerate(results, 1):
            stem = export.panel_stem(result, i)
            expected.extend(output / f"{stem}{suffix}" for suffix in (".csv", ".png", "_origin.txt"))
        record = {"name": name, "score": score, "series_count": sum(len(r.series) for r in results)}
        if name != "eqe":
            subprocess.run([sys.executable, "-m", "figrip", "jv", str(pdf), "-p", "1",
                            "--panel", "1", "-o", str(output / "jv")], check=True)
            if not list((output / "jv").glob("*.txt")):
                raise RuntimeError(f"no device metric TXT files: {name}")
            found = sorted([opv.jv_metrics(s.data[:, 0], s.data[:, 1])
                            for r in results for s in r.series], key=lambda m: m["pce"])
            known = sorted(truth["panels"][0]["series"], key=lambda s: s["pce"])
            if len(found) != len(known):
                raise RuntimeError(f"J-V series count differs from truth: {name}")
            record["metrics"] = []
            for actual, source in zip(found, known):
                reference = {k: source[k] * (100 if k == "ff" else 1)
                             for k in ("jsc", "voc", "ff", "pce")}
                errors = {k: abs(actual[k] - reference[k]) for k in reference}
                limits = {"jsc": 0.05, "voc": 0.01, "ff": 1.0, "pce": 0.05}
                if any(errors[k] > limits[k] for k in errors):
                    raise RuntimeError(f"J-V metrics differ from model: {name}: {errors}")
                record["metrics"].append({"reference": reference,
                                          "recovered": {k: actual[k] for k in reference},
                                          "absolute_error": errors})
        if any(not p.is_file() or p.stat().st_size == 0 for p in expected):
            raise RuntimeError(f"missing or empty export: {name}")
        report["cases"].append(record)
    (dest / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nPASS: 3 synthetic cases; exports, overlays and J-V metrics verified.\nReport: {dest / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

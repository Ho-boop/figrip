"""End-to-end: synthetic corpus (exact ground truth) -> figrip.pipeline.extract -> score."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

CORPUS = Path(__file__).resolve().parent / "corpus"
sys.path.insert(0, str(CORPUS))

import make_corpus  # noqa: E402
import score as scoring  # noqa: E402

TOL = 0.005  # 0.5 % of the axis range


@pytest.fixture(scope="session")
def corpus():
    cases = make_corpus.build(make_corpus.DEFAULT_OUT)
    return {c["case"]: c for c in cases}


def _extract(pdf):
    try:
        from figrip.pipeline import extract
        return extract(str(pdf))
    except Exception as e:  # missing/incomplete modules must fail cleanly, not crash collection
        pytest.fail(f"figrip.pipeline.extract raised {type(e).__name__}: {e}", pytrace=False)


@pytest.mark.parametrize("name", make_corpus.CASE_NAMES)
def test_case(name, corpus):
    case = corpus[name]
    results = _extract(make_corpus.DEFAULT_OUT / case["pdf"])
    sc = scoring.score(case, results, tol=TOL)
    report = scoring.format_report(sc)

    if not case["panels"]:  # raster_figure: the tool must not hallucinate
        assert len(results) == 0, f"expected zero panels\n{report}"
        return

    assert sc["panel_count_ok"], f"wrong number of panels\n{report}"
    for p in sc["panels"]:
        assert p["found_idx"] is not None, f"panel {p['truth_idx']} not found\n{report}"
        for s in p["series"]:
            assert s["found_idx"] is not None, f"series {s['truth_name']!r} not found\n{report}"
            assert s["ok"], f"series {s['truth_name']!r} off by more than {TOL:.2%}\n{report}"
    assert sc["ok"], report


# ---- self-check of the scorer: an oracle extractor must pass, a biased one must fail -----------------
def _oracle(case, dy=0.0, drop_names=False):
    from types import SimpleNamespace as NS
    import numpy as np
    out = []
    for p in case["panels"]:
        x0, y0, x1, y1 = p["bbox_hint"]
        ser = []
        yl = p["ylim"]
        for s in p["series"]:
            y = np.asarray(s["y"], float)
            y = y + dy * (yl[1] - yl[0]) if p["yscale"] == "linear" else y * 10 ** (dy * np.log10(yl[1] / yl[0]))
            ser.append(NS(name=None if drop_names else s["name"], kind=s["kind"],
                          data=np.column_stack([s["x"], y])))
        out.append(NS(panel=NS(bbox=(x0 + 1, y0 + 1, x1 - 1, y1 - 1)), series=ser))
    return out


@pytest.mark.parametrize("name", [n for n in make_corpus.CASE_NAMES if n != "raster_figure"])
def test_scorer_oracle(name, corpus):
    case = corpus[name]
    assert scoring.score(case, _oracle(case), tol=TOL)["ok"]
    assert scoring.score(case, _oracle(case, drop_names=True), tol=TOL)["ok"]
    assert not scoring.score(case, _oracle(case, dy=0.02), tol=TOL)["ok"]
    assert not scoring.score(case, _oracle(case)[:-1], tol=TOL)["ok"]  # missing panel

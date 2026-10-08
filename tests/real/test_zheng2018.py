"""Real-paper regression: Zheng et al., Adv. Mater. 2018, 1801801, Figure 2d/2f.

Every drawing on that page is a fill: tick labels are outlined glyphs and the
curves are outlined strokes. Table 2 on the same page is the ground truth.
The PDF lives in the user's Zotero library and is never committed.
"""
from pathlib import Path

import pytest

from figrip import opv, pipeline

PDF = Path("D:/Zotero/storage/QIZR5ARD/Zheng_et_al_2018_Adv_Mater_Fine_tuned_HTL_high_FF_OSCs.pdf")

# Table 2: (Voc V, Jsc mA/cm2, FF %, PCE %) in legend order
TABLE2 = {
    "d": [(0.78, 16.78, 66.70, 8.73), (0.87, 20.73, 80.79, 14.57), (0.87, 20.68, 73.87, 13.29)],
    "f": [(0.74, 17.14, 62.76, 7.96), (0.85, 21.47, 70.03, 12.78), (0.84, 21.29, 63.41, 11.34)],
}


@pytest.fixture(scope="module")
def results():
    if not PDF.exists():
        pytest.skip("Zotero benchmark PDF not available")
    return pipeline.extract(str(PDF), pages=[3])


def test_finds_all_five_plots(results):
    assert sorted(r.panel.label for r in results) == ["c", "d", "e", "f", "g"]


@pytest.mark.parametrize("label", ["d", "f"])
def test_jv_matches_table2(results, label):
    r = next(r for r in results if r.panel.label == label)
    got = sorted((opv.jv_metrics(s.data[:, 0], s.data[:, 1]) for s in r.series), key=lambda m: m["pce"])
    want = sorted(TABLE2[label], key=lambda t: t[3])
    assert len(got) == len(want)
    for m, (voc, jsc, ff, pce) in zip(got, want):
        assert m["jsc"] == pytest.approx(jsc, abs=0.05)
        assert m["voc"] == pytest.approx(voc, abs=0.01)   # table rounds Voc to 0.01 V
        assert m["ff"] == pytest.approx(ff, abs=1.0)      # sparse vertices near the knee
        assert m["pce"] == pytest.approx(pce, abs=0.05)

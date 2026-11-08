"""The hand-checkable example: exact fractions vs Stage 10's evaluator and MILP."""
import sys
from fractions import Fraction as F

from conftest import ROOT

sys.path.insert(0, str(ROOT / "src"))
import handcheck as hc  # noqa: E402

# Worked out on paper (alpha = 1/2, so v0 = 1; shelf for 2).
EXPECTED = {(0, 1): F(15, 2), (0, 2): F(160, 21), (0, 3): F(13, 2),
            (1, 2): F(148, 21), (1, 3): F(6), (2, 3): F(22, 5)}


def test_worked_example_matches_paper_and_stage10():
    res = hc.check(hc.worked_example())
    assert {t["shelf"]: t["hand_total"] for t in res["table"]} == EXPECTED
    assert all(t["agree"] for t in res["table"])            # Stage 10 evaluate() gives the same
    assert res["milp_shelf"] == (0, 2)                       # Stage 10 MILP picks {A, C}
    assert res["all_agree"]


def test_real_products_agree_with_stage10():
    inst = hc.from_data(["B001ELL4RM", "B0055ZS98K", "B082P9S82G", "B009NVDN7Y"],
                        demand=[4, 3, 2, 1], alpha=0.6, capacity=2)
    assert hc.check(inst)["all_agree"]

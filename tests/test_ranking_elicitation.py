"""Stage 05 helpers: parsing LLM output and the candidate orders shown to each model."""
import json

import pandas as pd
import pytest

from conftest import OUTPUTS

PARSE_OK = [
    '{"ranking": ["A", "B"], "reasoning": "A is best."}',
    '```json\n{"ranking": ["A"], "reasoning": "ok"}\n```',
    '<think>long reasoning...</think>\n{"ranking": ["A"], "reasoning": "ok"}',
    'Here is my answer:\n{"ranking": ["A"], "reasoning": "ok"}',
    '{"ranking": ["A"], "reasoning": "ok"}\nDone!',
    '{"ranking": ["A", "B"], "reasoning": "A beats B because {score: 1}."}',
]
PARSE_FAIL = [None, "", "<think>cut off mid-thought", '["A", "B"]', '{"ranking": ["A"], "reasoning": }']


@pytest.mark.parametrize("raw", PARSE_OK)
def test_extract_json_accepts_common_llm_formats(stage05, raw):
    assert stage05._extract_json(raw)["ranking"][0] == "A"


@pytest.mark.parametrize("raw", PARSE_FAIL)
def test_extract_json_rejects_unusable_output(stage05, raw):
    with pytest.raises((ValueError, json.JSONDecodeError)):
        stage05._extract_json(raw)


def test_permutations_are_deterministic_and_complete(stage05):
    cands = [f"C{i}" for i in range(7)]
    assert stage05.get_permuted_candidates(cands, "X", 0) == cands
    assert stage05.get_permuted_candidates(cands, "X", 1) == cands[::-1]
    for p in (2, 3):
        a = stage05.get_permuted_candidates(cands, "X", p)
        assert a == stage05.get_permuted_candidates(cands, "X", p)   # same across calls/processes
        assert sorted(a) == sorted(cands)


def test_committed_rankings_used_the_documented_candidate_orders(stage05):
    """Regenerate all 960 candidate orders from the embedding scores and compare with what
    the models were actually shown (stored with each ranking)."""
    from config import ALL_PAIRS_SCORES, EMBEDDING_TOP_K

    cands = stage05.load_top_k_candidates(ALL_PAIRS_SCORES, EMBEDDING_TOP_K)
    raw = pd.read_parquet(OUTPUTS / "rankings" / "raw_rankings.parquet")
    for r in raw.itertuples():
        expected = stage05.get_permuted_candidates(cands[r.oos_asin], r.oos_asin, r.perm_index)
        assert json.loads(r.candidate_order) == expected, (r.model, r.oos_asin, r.perm_index)

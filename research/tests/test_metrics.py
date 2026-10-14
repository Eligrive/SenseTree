import math

import numpy as np
import pytest

from sensetree_research.metrics import mrr_at_k, ndcg_at_k, per_query, ranked, recall_at_k
from sensetree_research.stats import compare


def test_ndcg_perfect_and_empty():
    rels = {"a": 1, "b": 1}
    assert ndcg_at_k(["a", "b", "c"], rels, 10) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], rels, 10) == 0.0


def test_ndcg_matches_hand_computation_with_graded_relevance():
    rels = {"a": 2, "b": 1}
    dcg = 1 / math.log2(2) + 2 / math.log2(3)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(["b", "a"], rels, 10) == pytest.approx(dcg / idcg)


def test_ndcg_cutoff_ignores_documents_below_k():
    assert ndcg_at_k(["x", "a"], {"a": 1}, 1) == 0.0


def test_recall_and_mrr():
    rels = {"a": 1, "b": 1, "c": 0}
    assert recall_at_k(["x", "a", "c"], rels, 3) == pytest.approx(0.5)
    assert mrr_at_k(["x", "c", "b"], rels, 10) == pytest.approx(1 / 3)
    assert mrr_at_k(["x", "c", "b"], rels, 2) == 0.0


def test_ranking_is_deterministic_on_ties():
    assert ranked({"b": 1.0, "a": 1.0, "c": 2.0}) == ["c", "a", "b"]


def test_missing_query_scores_zero_and_unjudged_query_is_skipped():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}, "q3": {"c": 0}}
    run = {"q1": {"a": 1.0}}
    scores = per_query(run, qrels, ["ndcg@10"])["ndcg@10"]
    assert scores == {"q1": 1.0, "q2": 0.0}


def test_comparison_detects_a_consistent_gain():
    rng = np.random.default_rng(0)
    base = {str(i): float(v) for i, v in enumerate(rng.uniform(0, 1, 200))}
    better = {k: min(1.0, v + 0.05) for k, v in base.items()}
    c = compare(base, better)
    assert c.delta > 0 and c.ci_low > 0 and c.p_value < 0.01


def test_comparison_of_identical_systems_is_not_significant():
    base = {str(i): float(i % 7) / 7 for i in range(100)}
    c = compare(base, dict(base))
    assert c.delta == 0 and c.p_value == 1.0

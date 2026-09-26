from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main
from credit_evidence_retention import strong_categories
from strong_rag_baseline.indexer import build_index
from strong_rag_baseline.retriever import BM25Index


PUBLIC = (
    Path(r"D:\OpenCode\Agenthon2026_T4")
    / "reference"
    / "track4-analysis-public"
    / "units"
    / "t4-credit-event-2023"
)


def _task():
    return json.loads((PUBLIC / "task.json").read_text(encoding="utf-8"))


def test_adversarial_facts():
    assert strong_categories(
        "Risk Factors. We may be unable to obtain additional financing and could default on debt."
    ) == []
    assert strong_categories(
        "The facility has aggregate commitments of $500 million, subject to a borrowing base "
        "and lender conditions. No amount currently available is disclosed."
    ) == []
    assert strong_categories(
        "As of November 26, 2022, total liquidity included $1,274.1 million "
        "of available revolver borrowing capacity."
    ) == ["credit_availability"]


def test_credit_production_candidates_are_same_issuer_and_top5():
    task=_task()
    corpus=build_index(PUBLIC/"corpus",task["cutoff_date"])
    index=BM25Index(corpus.chunks,task["cutoff_date"])
    for entity in task["entities"]:
        rows=main._production_candidates(task,entity,index)
        assert len(rows) <= main.TOP_K
        cik=str(entity["cik"])
        assert all(cik in row.doc_id for row in rows)


def test_non_credit_candidate_path_unchanged():
    task=_task()
    task["family"]="post_earnings_reaction"
    corpus=build_index(PUBLIC/"corpus",task["cutoff_date"])
    index=BM25Index(corpus.chunks,task["cutoff_date"])
    entity=task["entities"][0]
    base=main._candidates(task,entity,index)
    prod=main._production_candidates(task,entity,index)
    assert [(x.doc_id,x.span_start,x.span_end) for x in prod] == [
        (x.doc_id,x.span_start,x.span_end) for x in base
    ]

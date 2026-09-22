"""Assemble and self-check the final answer JSON.

The self-check enforces the mistakes that are cheap to catch locally and fatal
at scoring time: claims whose spans do not resolve in the corpus, missing
interval bounds, and a wrong top-level shape (``notes`` must be an object).
It is a last-line assertion — grounding already happened in the agent.
"""
from __future__ import annotations

from .agent import EntityResult
from .indexer import IndexedCorpus


def build_answer(
    task: dict, results: list[EntityResult], corpus: IndexedCorpus
) -> dict:
    total_dropped = sum(r.dropped_claims for r in results)
    total_claims = sum(len(r.prediction["claims"]) for r in results)
    answer = {
        "task_id": task.get("task_id", ""),
        "schema_version": task.get("schema_version", "3"),
        "entity_predictions": [r.prediction for r in results],
        "evidence_trace": (
            f"strong_rag_baseline: BM25 span-chunk retrieval + $MODEL_ENDPOINT. "
            f"{total_claims} grounded claims kept, {total_dropped} ungroundable "
            f"evidence items dropped. All cited spans resolved in the frozen "
            f"corpus; embargo enforced at retrieval time."
        ),
        "notes": {
            "agent": "strong_rag_baseline",
            "retrieval": "bm25-span-chunks",
            "dropped_evidence_items": total_dropped,
        },
    }
    _assert_valid(answer, corpus)
    return answer


def _assert_valid(answer: dict, corpus: IndexedCorpus) -> None:
    assert isinstance(answer["notes"], dict), "top-level notes must be an object"
    for entity in answer["entity_predictions"]:
        interval = entity.get("interval") or {}
        assert "lo" in interval and "hi" in interval, (
            f"{entity.get('entity_id')}: interval must contain lo and hi"
        )
        for claim in entity.get("claims", []):
            doc_text = corpus.doc_texts.get(claim["doc_id"], "")
            assert 0 <= claim["span_start"] < claim["span_end"] <= len(doc_text), (
                f"{entity.get('entity_id')}: span "
                f"[{claim['span_start']}, {claim['span_end']}) does not resolve "
                f"in {claim['doc_id']!r}"
            )

from __future__ import annotations

from typing import Any

from strong_rag_baseline.indexer import Chunk, IndexedCorpus
from strong_rag_baseline.retriever import BM25Index


_POSITIVE = (
    "increased",
    "increase",
    "grew",
    "growth",
    "higher",
    "improved",
    "improvement",
    "strong",
    "record",
)
_NEGATIVE = (
    "decreased",
    "decrease",
    "declined",
    "decline",
    "lower",
    "weaker",
    "weakness",
    "down",
)
_BUSINESS = (
    "revenue",
    "net sales",
    "operating income",
    "operating margin",
    "gross margin",
    "earnings per share",
    "diluted earnings",
    "guidance",
    "outlook",
)


def _entity_chunks(
    index: BM25Index,
    entity: dict[str, Any],
    limit: int = 15,
) -> list[Chunk]:
    cik = str(entity.get("cik") or "")
    query = (
        f"{entity.get('name','')} revenue net sales operating income operating "
        "margin earnings guidance outlook increase decrease growth"
    )
    rows = [
        x.chunk
        for x in index.search(query, 100)
        if not cik or cik in x.chunk.doc_id
    ]
    return rows[:limit]


def _chunk_signal(chunk: Chunk) -> float:
    text = chunk.text.lower()
    pos = sum(text.count(x) for x in _POSITIVE)
    neg = sum(text.count(x) for x in _NEGATIVE)
    business_hits = sum(term in text for term in _BUSINESS)
    multiplier = 1.0 + 0.25 * min(4, business_hits)
    return (pos - neg) * multiplier


def _entity_signal(chunks: list[Chunk]) -> float:
    # Concentrate on the strongest retrieved business passages rather than
    # letting long boilerplate dominate.
    return sum(_chunk_signal(c) for c in chunks[:8])


def _label_point(signal: float) -> tuple[str, float]:
    if signal >= 2.0:
        return "positive_reaction", 3.0
    if signal <= -2.0:
        return "negative_reaction", -3.0
    return "flat", 0.0


def run_postearn(
    task: dict[str, Any],
    index: BM25Index,
    corpus: IndexedCorpus,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    del corpus  # index already contains the admitted cutoff-safe corpus.
    level = float(task.get("interval_level", 0.90))
    half_width = 20.0
    predictions: list[dict[str, Any]] = []
    signal_notes: dict[str, Any] = {}

    for entity in task.get("entities", []):
        eid = str(entity["entity_id"])
        chunks = _entity_chunks(index, entity)
        signal = _entity_signal(chunks)
        label, point = _label_point(signal)

        direction = 1.0 if point > 0 else (-1.0 if point < 0 else 0.0)
        if direction:
            evidence_chunks = sorted(
                chunks,
                key=lambda c: (-(direction * _chunk_signal(c)), c.doc_id, c.span_start),
            )
        else:
            evidence_chunks = sorted(
                chunks,
                key=lambda c: (-abs(_chunk_signal(c)), c.doc_id, c.span_start),
            )

        claims = [
            {
                "doc_id": c.doc_id,
                "span_start": c.span_start,
                "span_end": c.span_end,
                "claim": f"Pre-cutoff operating and earnings evidence for {eid}.",
            }
            for c in evidence_chunks[:3]
        ]

        predictions.append(
            {
                "entity_id": eid,
                "label": label,
                "point_forecast": point,
                "interval": {
                    "level": level,
                    "lo": point - half_width,
                    "hi": point + half_width,
                },
                "claims": claims,
            }
        )
        signal_notes[eid] = {
            "operating_signal_score": signal,
            "label": label,
            "point_forecast_pct": point,
        }

    notes = {
        "adapter": "postearn_v1",
        "signal_threshold": 2.0,
        "interval_half_width_pct_points": half_width,
        "signals": signal_notes,
        "house_calls_attempted": 0,
    }
    return predictions, notes

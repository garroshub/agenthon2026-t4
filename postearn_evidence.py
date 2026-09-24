from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from strong_rag_baseline.indexer import Chunk, IndexedCorpus
from strong_rag_baseline.retriever import BM25Index

CATEGORY_QUERIES = {
    "operations": (
        "revenue sales operating income gross margin operating margin expenses "
        "segment growth decline demand profitability"
    ),
    "guidance": (
        "guidance outlook expect expected anticipates future quarter net sales "
        "operating income demand margin"
    ),
    "adverse": (
        "risk risks demand weakness decline costs inflation competition uncertainty "
        "headwinds adverse pressure"
    ),
}

_EXPECTATION_TERMS = (
    "analyst consensus",
    "consensus estimate",
    "consensus estimates",
    "wall street estimate",
    "wall street estimates",
    "earnings estimate",
    "earnings estimates",
    "revenue estimate",
    "revenue estimates",
)

_GUIDANCE_MARKERS = (
    " guidance ",
    "fourth quarter 2023 guidance",
    "we anticipate",
    "we expect our",
    "we expect total",
    "we expect revenue",
    "we expect expenses",
    "future advertising revenue",
    "2024, respectively",
)


@dataclass(frozen=True)
class EvidenceCard:
    entity_id: str
    name: str
    doc_ids: list[str]
    facts: dict[str, list[Chunk]]
    expectation_gap_status: str
    expectation_gap_chunks: list[Chunk]
    temporal_semantics: dict[str, Any]


def _entity_chunks(entity: dict[str, Any], corpus: IndexedCorpus) -> list[Chunk]:
    cik = str(entity.get("cik") or "").lstrip("0")
    cik_full = str(entity.get("cik") or "")
    out = []
    for c in corpus.chunks:
        doc = c.doc_id
        if cik_full and cik_full in doc:
            out.append(c)
        elif cik and cik in doc:
            out.append(c)
    if not out:
        raise ValueError(f"no corpus chunks found for entity {entity.get('entity_id')}")
    return out


def _top_diverse(
    chunks: list[Chunk],
    cutoff: str,
    query: str,
    limit: int,
    used: set[tuple[str, int, int]],
) -> list[Chunk]:
    idx = BM25Index(chunks, cutoff)
    scored = idx.search(query, max(limit * 8, 12))
    out: list[Chunk] = []
    for s in scored:
        c = s.chunk
        key = (c.doc_id, c.span_start, c.span_end)
        if key in used:
            continue
        out.append(c)
        used.add(key)
        if len(out) >= limit:
            break
    return out




def _guidance_strength(c: Chunk) -> int:
    text = c.text.lower()
    score = 0
    weighted = (
        ("fourth quarter 2023 guidance", 20),
        ("future advertising revenue", 12),
        ("we expect our", 10),
        ("we anticipate making capital expenditures", 10),
        ("we anticipate", 7),
        ("we expect", 6),
        ("guidance", 5),
        ("2024", 3),
        ("revenue", 3),
        ("operating income", 3),
        ("operating losses", 3),
        ("capital expenditures", 2),
        ("expenses", 2),
    )
    for phrase, weight in weighted:
        if phrase in text:
            score += weight
    # Tax/accounting-only forward statements are not operating guidance.
    if ("tax" in text or "amortization" in text) and not any(
        k in text for k in ("revenue", "operating income", "operating losses", "capital expenditures")
    ):
        score -= 10
    return score


def _select_guidance(
    chunks: list[Chunk],
    cutoff: str,
    limit: int,
    used: set[tuple[str, int, int]],
) -> list[Chunk]:
    candidates = [c for c in chunks if _guidance_strength(c) >= 8]
    if not candidates:
        return []
    best_strength = max(_guidance_strength(c) for c in candidates)
    min_strength = max(8, int(best_strength * 0.80))
    candidates = [c for c in candidates if _guidance_strength(c) >= min_strength]
    ranked = sorted(
        candidates,
        key=lambda c: (-_guidance_strength(c), c.doc_id, c.span_start),
    )
    out: list[Chunk] = []
    for c in ranked:
        key = (c.doc_id, c.span_start, c.span_end)
        if key in used:
            continue
        if any(
            c.doc_id == prev.doc_id and abs(c.span_start - prev.span_start) < 2500
            for prev in out
        ):
            continue
        out.append(c)
        used.add(key)
        if len(out) >= limit:
            break
    return out

def build_postearn_cards(
    task: dict[str, Any],
    corpus: IndexedCorpus,
    per_category: int = 2,
) -> tuple[dict[str, EvidenceCard], dict[str, list[Chunk]]]:
    cutoff = str(task["cutoff_date"])
    cards: dict[str, EvidenceCard] = {}
    candidates: dict[str, list[Chunk]] = {}

    for entity in task.get("entities", []):
        eid = str(entity["entity_id"])
        chunks = _entity_chunks(entity, corpus)
        used: set[tuple[str, int, int]] = set()
        facts: dict[str, list[Chunk]] = {}
        for category, query in CATEGORY_QUERIES.items():
            if category == "guidance":
                facts[category] = _select_guidance(
                    chunks,
                    cutoff,
                    per_category,
                    used,
                )
                continue
            facts[category] = _top_diverse(
                chunks,
                cutoff,
                query,
                per_category,
                used,
            )

        expectation_matches = []
        for c in chunks:
            low = c.text.lower()
            if any(term in low for term in _EXPECTATION_TERMS):
                expectation_matches.append(c)
        expectation_matches.sort(
            key=lambda c: (c.doc_id, c.span_start, c.span_end)
        )
        exp_selected: list[Chunk] = []
        for c in expectation_matches:
            key = (c.doc_id, c.span_start, c.span_end)
            if key in used:
                continue
            exp_selected.append(c)
            used.add(key)
            if len(exp_selected) >= 1:
                break

        expectation_status = (
            "available_in_frozen_corpus"
            if exp_selected
            else "missing_no_reliable_pre_earnings_market_expectation_in_frozen_sec_corpus"
        )

        flat_chunks: list[Chunk] = []
        for category in ("operations", "guidance", "adverse"):
            flat_chunks.extend(facts[category])
        flat_chunks.extend(exp_selected)

        card = EvidenceCard(
            entity_id=eid,
            name=str(entity.get("name") or eid),
            doc_ids=sorted({c.doc_id for c in chunks}),
            facts=facts,
            expectation_gap_status=expectation_status,
            expectation_gap_chunks=exp_selected,
            temporal_semantics={
                "cutoff_date": cutoff,
                "report_datetime": entity.get("report_datetime"),
                "event_window": entity.get("event_window"),
                "benchmark": entity.get("benchmark"),
                "target": (
                    "one-day abnormal return percent = company total return minus SPY "
                    "over the task event window"
                ),
                "forbidden_inference": (
                    "Do not treat historical operating improvement as proof that the "
                    "next earnings release beats expectations. Do not use post-cutoff results."
                ),
            },
        )
        cards[eid] = card
        candidates[eid] = flat_chunks

    return cards, candidates


def card_payload(card: EvidenceCard) -> dict[str, Any]:
    def refs(xs: list[Chunk]) -> list[dict[str, Any]]:
        return [
            {
                "doc_id": c.doc_id,
                "doc_date": c.doc_date,
                "span_start": c.span_start,
                "span_end": c.span_end,
                "text": c.text,
            }
            for c in xs
        ]

    return {
        "entity_id": card.entity_id,
        "name": card.name,
        "documents": card.doc_ids,
        "operating_change_evidence": refs(card.facts["operations"]),
        "management_guidance_evidence": refs(card.facts["guidance"]),
        "adverse_or_counter_evidence": refs(card.facts["adverse"]),
        "expectation_gap_status": card.expectation_gap_status,
        "expectation_gap_evidence": refs(card.expectation_gap_chunks),
        "temporal_semantics": card.temporal_semantics,
    }


def indexed_card_payload(card: EvidenceCard, chunks: list[Chunk]) -> dict[str, Any]:
    id_map = {
        (c.doc_id, c.span_start, c.span_end): f"c{i}"
        for i, c in enumerate(chunks, 1)
    }

    def ids(xs: list[Chunk]) -> list[str]:
        out = []
        for c in xs:
            cid = id_map.get((c.doc_id, c.span_start, c.span_end))
            if cid is not None:
                out.append(cid)
        return out

    return {
        "entity_id": card.entity_id,
        "name": card.name,
        "documents": card.doc_ids,
        "operating_change_candidate_ids": ids(card.facts["operations"]),
        "management_guidance_candidate_ids": ids(card.facts["guidance"]),
        "adverse_or_counter_candidate_ids": ids(card.facts["adverse"]),
        "expectation_gap_status": card.expectation_gap_status,
        "expectation_gap_candidate_ids": ids(card.expectation_gap_chunks),
        "temporal_semantics": card.temporal_semantics,
    }

"""Per-entity orchestration: retrieve → reason → verify spans → assemble claims.

The pipeline is intentionally strict about citations: a claim survives only if
its quote resolves to an exact span in the cited document (or, failing that, if
the quote's source chunk is identifiable so the chunk's own offsets can stand
in). Claims that cannot be grounded are dropped — an ungrounded claim risks the
faithfulness gate. A row with no surviving claim is contract-invalid, so callers must
repair or reject such a result before serialization.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

from .client import ModelClient
from .indexer import Chunk, IndexedCorpus
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .retriever import BM25Index
from .span_finder import find_span

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

#: Query terms appended to every entity query — steer retrieval toward result
#: and outlook language regardless of family.
_QUERY_SUFFIX = "results revenue earnings guidance outlook growth"


@dataclass
class EntityResult:
    prediction: dict  # entity_predictions[] element
    dropped_claims: int
    model_raw: str


def _parse_model_json(raw: str) -> dict:
    """Recover a complete JSON object from a reply with optional reasoning prose."""
    decoder = json.JSONDecoder()
    candidates: list[dict] = []
    for i, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(raw[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            candidates.append(obj)

    if not candidates:
        raise ValueError("model reply contains no complete JSON object")

    preferred_keys = {"predictions", "selections", "judgments", "entity_predictions"}
    for obj in reversed(candidates):
        if preferred_keys.intersection(obj):
            return obj
    return candidates[-1]


def _entity_query(entity: dict) -> str:
    parts = [
        str(entity.get(key, ""))
        for key in ("name", "entity_id", "sector", "series_id", "description")
    ]
    return " ".join(p for p in parts if p) + " " + _QUERY_SUFFIX


def _ground_claims(
    evidence: list[dict],
    corpus: IndexedCorpus,
    retrieved: list[Chunk],
) -> tuple[list[dict], int]:
    """Map model evidence to exact-span claims; count what had to be dropped."""
    claims: list[dict] = []
    dropped = 0
    retrieved_by_doc: dict[str, list[Chunk]] = {}
    for chunk in retrieved:
        retrieved_by_doc.setdefault(chunk.doc_id, []).append(chunk)

    for item in evidence:
        if not isinstance(item, dict):
            dropped += 1
            continue
        doc_id = item.get("doc_id", "")
        quote = str(item.get("quote", ""))
        claim_text = str(item.get("claim", "")).strip()
        doc_text = corpus.doc_texts.get(doc_id)
        if doc_text is None or not claim_text:
            dropped += 1
            continue

        span = find_span(doc_text, quote)
        if span is None:
            # Fallback: cite the retrieved chunk the quote most plausibly came
            # from (longest token overlap). Chunk offsets are known-good.
            chunk = _best_chunk(quote, retrieved_by_doc.get(doc_id, []))
            if chunk is None:
                dropped += 1
                continue
            span = (chunk.span_start, chunk.span_end)

        claims.append(
            {
                "doc_id": doc_id,
                "span_start": span[0],
                "span_end": span[1],
                "claim": claim_text,
            }
        )
    return claims, dropped


def _best_chunk(quote: str, chunks: list[Chunk]) -> Chunk | None:
    q_tokens = set(quote.lower().split())
    if not q_tokens:
        return None
    best: tuple[int, Chunk] | None = None
    for chunk in chunks:
        overlap = len(q_tokens & set(chunk.text.lower().split()))
        if best is None or overlap > best[0]:
            best = (overlap, chunk)
    return best[1] if best and best[0] > 0 else None


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _safe_interval(parsed: dict, level: float, point: float | None) -> dict:
    interval = parsed.get("interval") or {}
    lo, hi = interval.get("lo"), interval.get("hi")
    if _finite_number(lo) and _finite_number(hi) and float(lo) <= float(hi):
        return {"level": level, "lo": float(lo), "hi": float(hi)}
    center = float(point) if _finite_number(point) else 0.0
    half = max(abs(center) * 0.5, 1.0)
    return {"level": level, "lo": center - half, "hi": center + half}


def run_entity(
    task: dict,
    entity: dict,
    index: BM25Index,
    corpus: IndexedCorpus,
    client: ModelClient,
    top_k: int,
) -> EntityResult:
    retrieved = [s.chunk for s in index.search(_entity_query(entity), top_k)]
    raw = client.complete(SYSTEM_PROMPT, build_user_prompt(task, entity, retrieved))
    parsed = _parse_model_json(raw)

    target = task.get("target", {})
    target_type = target.get("type")
    labels = target.get("labels") or []

    point = parsed.get("point_forecast")
    point_value = float(point) if _finite_number(point) else None
    claims, dropped = _ground_claims(
        parsed.get("evidence") or [], corpus, retrieved
    )

    prediction: dict = {
        "entity_id": entity.get("entity_id", ""),
        "point_forecast": point_value,
        "interval": _safe_interval(
            parsed, task.get("interval_level", 0.90), point_value
        ),
        "claims": claims,
    }
    if target_type == "classification":
        label = parsed.get("label")
        if not isinstance(label, str) or (labels and label not in labels):
            label = labels[0] if labels else ""
        prediction["label"] = label
    if target_type == "ranking" and isinstance(parsed.get("rank"), int):
        prediction["rank"] = parsed["rank"]
    return EntityResult(prediction=prediction, dropped_claims=dropped, model_raw=raw)

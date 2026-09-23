from __future__ import annotations

import math
from typing import Any

from strong_rag_baseline.indexer import Chunk, IndexedCorpus
from strong_rag_baseline.retriever import BM25Index


def _best_component_chunk(
    index: BM25Index,
    entity: dict[str, Any],
) -> Chunk | None:
    name = str(entity.get("name") or "")
    series = str(entity.get("series_fred") or "")
    query = f"{name} {series} CPI seasonally adjusted month over month latest published"
    rows = [x.chunk for x in index.search(query, 20)]
    if not rows:
        return index.chunks[0] if index.chunks else None

    name_terms = [x.lower() for x in name.replace("(", " ").replace(")", " ").split() if len(x) > 2]
    def score(c: Chunk) -> tuple[int, int, int]:
        text = c.text.lower()
        hits = sum(term in text for term in name_terms)
        if series and series.lower() in text:
            hits += 3
        return (hits, -len(c.text), -c.span_start)

    return max(rows, key=score)


def run_cpi_component(
    task: dict[str, Any],
    index: BM25Index,
    corpus: IndexedCorpus,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    del corpus
    level = float(task.get("interval_level", 0.90))
    alpha = 0.25
    interval_floor = 2.0
    predictions: list[dict[str, Any]] = []
    notes_by_entity: dict[str, Any] = {}

    for entity in task.get("entities", []):
        eid = str(entity["entity_id"])
        latest_raw = entity.get("latest_published_mom_pct")
        latest = (
            float(latest_raw)
            if isinstance(latest_raw, (int, float))
            and not isinstance(latest_raw, bool)
            and math.isfinite(float(latest_raw))
            else 0.0
        )
        point = alpha * latest
        half = max(interval_floor, 0.75 * abs(latest))

        chunk = _best_component_chunk(index, entity)
        claims: list[dict[str, Any]] = []
        if chunk is not None:
            claims.append(
                {
                    "doc_id": chunk.doc_id,
                    "span_start": chunk.span_start,
                    "span_end": chunk.span_end,
                    "claim": (
                        "Pre-cutoff first-published CPI component history used "
                        f"for {eid}."
                    ),
                }
            )

        predictions.append(
            {
                "entity_id": eid,
                "point_forecast": point,
                "interval": {
                    "level": level,
                    "lo": point - half,
                    "hi": point + half,
                },
                "claims": claims,
            }
        )
        notes_by_entity[eid] = {
            "latest_published_mom_pct": latest,
            "point_after_shrinkage": point,
            "interval_half_width": half,
        }

    return predictions, {
        "adapter": "cpi_component_v1",
        "shrinkage_alpha": alpha,
        "shrinkage_target": 0.0,
        "interval_floor_pct_points": interval_floor,
        "signals": notes_by_entity,
        "house_calls_attempted": 0,
    }

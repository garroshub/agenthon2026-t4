from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping

from strong_rag_baseline.agent import EntityResult, _parse_model_json
from strong_rag_baseline.client import HTTPModelClient, MockModelClient
from strong_rag_baseline.config import Config
from strong_rag_baseline.formatter import build_answer
from strong_rag_baseline.indexer import Chunk, IndexedCorpus, build_index

MARKET_DOC_TEMPLATE = "COT_{entity_id}_20241025"
METHODOLOGY_DOC_ID = "COT_METHODOLOGY_20241025"
MARKET_SNAPSHOT_DOC_ID = "MKT_SNAPSHOT_20241031"

ROW_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})\s*\|\s*"
    r"(?P<oi>[\d,]+)\s*\|\s*"
    r"(?P<long>[\d,]+)\s*\|\s*"
    r"(?P<short>[\d,]+)\s*\|\s*"
    r"(?P<net>[+-]?[\d,]+)\s*\|\s*"
    r"(?P<pct>[+-]?\d+(?:\.\d+)?)"
)


def _doc_text(corpus_dir: Path, doc_id: str) -> str:
    data = json.loads((corpus_dir / f"{doc_id}.json").read_text(encoding="utf-8"))
    text = data.get("text")
    if not isinstance(text, str):
        raise ValueError(f"missing text for {doc_id}")
    return text


def _recent_rows(text: str, n: int = 5) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for m in ROW_RE.finditer(text):
        rows.append(
            {
                "date": m.group("date"),
                "open_interest": int(m.group("oi").replace(",", "")),
                "noncommercial_long": int(m.group("long").replace(",", "")),
                "noncommercial_short": int(m.group("short").replace(",", "")),
                "noncommercial_net": int(m.group("net").replace(",", "")),
                "net_pct_open_interest": float(m.group("pct")),
            }
        )
    if len(rows) < n:
        raise ValueError(f"expected at least {n} COT rows")
    return rows[-n:]


def _chunks_for_doc(corpus: IndexedCorpus, doc_id: str) -> list[Chunk]:
    chunks = [c for c in corpus.chunks if c.doc_id == doc_id]
    chunks.sort(key=lambda c: (c.span_start, c.span_end))
    if not chunks:
        raise ValueError(f"missing indexed corpus doc {doc_id}")
    return chunks


def _first_chunk(corpus: IndexedCorpus, doc_id: str) -> Chunk:
    return _chunks_for_doc(corpus, doc_id)[0]


def _latest_chunk(corpus: IndexedCorpus, doc_id: str) -> Chunk:
    return max(_chunks_for_doc(corpus, doc_id), key=lambda c: (c.span_end, c.span_start))


def _candidate_meta(chunks: list[Chunk]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": f"c{i}",
            "doc_id": c.doc_id,
            "doc_date": c.doc_date,
            "span_start": c.span_start,
            "span_end": c.span_end,
        }
        for i, c in enumerate(chunks, 1)
    ]


def _citation_chunks(
    task: dict[str, Any],
    corpus: IndexedCorpus,
) -> dict[str, list[Chunk]]:
    methodology = _first_chunk(corpus, METHODOLOGY_DOC_ID)
    snapshot = _first_chunk(corpus, MARKET_SNAPSHOT_DOC_ID)
    out: dict[str, list[Chunk]] = {}
    for entity in task.get("entities", []):
        eid = str(entity["entity_id"])
        out[eid] = [
            _latest_chunk(corpus, MARKET_DOC_TEMPLATE.format(entity_id=eid)),
            methodology,
            snapshot,
        ]
    return out


def _prompt(
    task: dict[str, Any],
    corpus_dir: Path,
    corpus: IndexedCorpus,
) -> tuple[str, str, dict[str, list[Chunk]], int]:
    entities = [e for e in task.get("entities", []) if isinstance(e, dict)]
    if len(entities) != 10:
        raise ValueError(f"expected exactly 10 COT entities, got {len(entities)}")

    by_entity = _citation_chunks(task, corpus)
    roster: list[dict[str, Any]] = []
    for entity in entities:
        eid = str(entity["entity_id"])
        market_doc_id = MARKET_DOC_TEMPLATE.format(entity_id=eid)
        source_text = _doc_text(corpus_dir, market_doc_id)
        roster.append(
            {
                "entity_id": eid,
                "name": entity.get("name"),
                "asset_class": entity.get("asset_class"),
                "current_net_pct_open_interest": entity.get("net_pct_oi_20241022"),
                "trailing_4wk_net_change_pct_open_interest": entity.get(
                    "trailing_4wk_net_change_pct_oi"
                ),
                "recent_cot_rows": _recent_rows(source_text, 5),
                "market_source_excerpt": by_entity[eid][0].text,
                "citation_candidates": _candidate_meta(by_entity[eid]),
            }
        )

    system = (
        "You are a financial forecasting agent in a sealed benchmark. "
        "Use ONLY the supplied pre-cutoff evidence. This is a cross-sectional ranking task. "
        "Reason about ALL TEN markets together before emitting forecasts. "
        "Every point_forecast must be the expected five-week change in non-commercial net "
        "position measured in PERCENTAGE POINTS OF OPEN INTEREST. Use one common scale across "
        "all ten markets. Do not normalize, center, z-score, or rescale separate subsets. "
        "Return one JSON object only. For every market provide point_forecast, a 90% interval, "
        "and one or two candidate_id values from that market's supplied citation candidates. "
        "Do not use outside knowledge or any post-cutoff realized outcome."
    )

    user_obj = {
        "task_prompt": task.get("prompt", ""),
        "family": task.get("family", ""),
        "cutoff_date": task.get("cutoff_date", ""),
        "target": task.get("target", {}),
        "interval_level": task.get("interval_level", 0.90),
        "scale_contract": {
            "point_forecast_unit": "percentage points of open interest",
            "comparison_scope": "all 10 markets jointly",
            "rank_direction": "larger point_forecast means stronger expected increase and smaller rank number",
            "rank_note": "runtime rebuilds the final global rank from point_forecast; model rank is ignored",
        },
        "shared_evidence": {
            "methodology": {
                "doc_id": METHODOLOGY_DOC_ID,
                "text": _doc_text(corpus_dir, METHODOLOGY_DOC_ID),
            },
            "market_snapshot": {
                "doc_id": MARKET_SNAPSHOT_DOC_ID,
                "text": _doc_text(corpus_dir, MARKET_SNAPSHOT_DOC_ID),
            },
        },
        "output_schema": {
            "predictions": [
                {
                    "entity_id": "string",
                    "point_forecast": 0.0,
                    "interval": {"lo": 0.0, "hi": 0.0},
                    "candidate_ids": ["c1"],
                }
            ]
        },
        "full_roster": roster,
    }
    user = json.dumps(user_obj, ensure_ascii=False)
    return system, user, by_entity, len(system) + len(user)


def _mock_reply(task: dict[str, Any]) -> str:
    rows = []
    for entity in task.get("entities", []):
        current = float(entity.get("net_pct_oi_20241022") or 0.0)
        trail = float(entity.get("trailing_4wk_net_change_pct_oi") or 0.0)
        point = 0.15 * trail - 0.02 * current
        half = max(1.0, abs(point) * 0.75 + 0.5)
        rows.append(
            {
                "entity_id": str(entity["entity_id"]),
                "point_forecast": point,
                "interval": {"lo": point - half, "hi": point + half},
                "candidate_ids": ["c1", "c3"],
            }
        )
    return json.dumps({"predictions": rows})


def _parsed_rows(raw_text: str, allowed_ids: set[str]) -> dict[str, dict[str, Any]]:
    try:
        parsed = _parse_model_json(raw_text)
    except Exception:
        return {}
    items = parsed.get("predictions")
    if not isinstance(items, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        eid = str(item.get("entity_id") or "")
        if eid in allowed_ids and eid not in out:
            out[eid] = item
    return out


def _rebuild_ranks(predictions: list[dict[str, Any]]) -> None:
    ordered = sorted(
        predictions,
        key=lambda row: (-float(row["point_forecast"]), str(row["entity_id"])),
    )
    rank = {str(row["entity_id"]): i + 1 for i, row in enumerate(ordered)}
    for row in predictions:
        row["rank"] = rank[str(row["entity_id"])]
        row.pop("label", None)


def run_cot_joint(
    task: dict[str, Any],
    corpus_dir: Path,
    config: Config,
    use_mock: bool,
    prediction_row: Callable[[dict, dict, Mapping[str, Any], list[Chunk]], dict[str, Any]],
) -> dict[str, Any]:
    if str(task.get("family") or "") != "positioning_shift":
        raise ValueError("joint COT runner only supports family=positioning_shift")

    corpus = build_index(corpus_dir, task["cutoff_date"])
    system, user, by_entity, prompt_chars = _prompt(task, corpus_dir, corpus)
    entities = [e for e in task.get("entities", []) if isinstance(e, dict)]
    allowed = {str(e["entity_id"]) for e in entities}

    if use_mock:
        client = MockModelClient(lambda _system, _user: _mock_reply(task))
    else:
        if not config.model_endpoint:
            raise RuntimeError("MODEL_ENDPOINT is not set")
        client = HTTPModelClient(
            replace(
                config,
                max_retries=1,
                timeout_s=min(config.timeout_s, max(10.0, config.unit_timeout_s - 60.0)),
            )
        )

    raw_text = client.complete(system, user)
    parsed = _parsed_rows(raw_text, allowed)
    predictions = [
        prediction_row(
            task,
            entity,
            parsed.get(str(entity["entity_id"]), {}),
            by_entity[str(entity["entity_id"])],
        )
        for entity in entities
    ]
    _rebuild_ranks(predictions)

    results = [
        EntityResult(prediction=row, dropped_claims=0, model_raw="cot_joint_roster_v1")
        for row in predictions
    ]
    answer = build_answer(task, results, corpus)
    answer["target_type"] = "ranking"
    answer.setdefault("notes", {}).update(
        {
            "cot_joint_roster": True,
            "house_calls": 1,
            "prompt_chars": prompt_chars,
            "parsed_house_entities": len(parsed),
            "fallback_entities": len(entities) - len(parsed),
            "global_rank_rebuilt_from_points": True,
        }
    )
    return answer

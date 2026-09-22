from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from auction_family import run_auction
from strong_rag_baseline.agent import EntityResult, _parse_model_json
from strong_rag_baseline.client import HTTPModelClient, MockModelClient
from strong_rag_baseline.cli import _mock_reply
from strong_rag_baseline.config import Config
from strong_rag_baseline.formatter import build_answer
from strong_rag_baseline.indexer import Chunk, IndexedCorpus, build_index
from strong_rag_baseline.retriever import BM25Index

BATCH_SIZE = 6
TOP_K = 5
MAX_HOUSE_CALLS_GENERIC = 15


def _query(task: dict, entity: dict) -> str:
    target = task.get("target", {})
    parts = [
        str(entity.get(k, ""))
        for k in (
            "entity_id", "name", "ticker", "symbol", "series_id",
            "description", "sector", "tenor"
        )
    ]
    parts += [
        str(target.get("name", "")),
        str(task.get("family", "")),
        str(task.get("prompt", "")),
        "forecast guidance outlook results historical target",
    ]
    return " ".join(x for x in parts if x)


def _candidates(task: dict, entity: dict, index: BM25Index) -> list[Chunk]:
    chunks = [s.chunk for s in index.search(_query(task, entity), TOP_K)]
    if not chunks and index.chunks:
        chunks = [index.chunks[0]]
    return chunks


def _payload_candidates(chunks: list[Chunk]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": f"c{i}",
            "doc_id": c.doc_id,
            "doc_date": c.doc_date,
            "span_start": c.span_start,
            "span_end": c.span_end,
            "text": c.text,
        }
        for i, c in enumerate(chunks, 1)
    ]


def _safe_interval(task: dict, point: float, raw: Any) -> dict[str, float]:
    level = float(task.get("interval_level", 0.90))
    if isinstance(raw, Mapping):
        lo, hi = raw.get("lo"), raw.get("hi")
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            lo, hi = float(lo), float(hi)
            if math.isfinite(lo) and math.isfinite(hi) and lo <= hi:
                return {"level": level, "lo": lo, "hi": hi}
    if task.get("family") == "credit_event":
        return {"level": level, "lo": 0.0, "hi": 1.0}
    half = max(1.0, abs(point) * 0.5)
    return {"level": level, "lo": point - half, "hi": point + half}


def _fallback_point(task: dict, entity: dict) -> float:
    if task.get("family") == "credit_event":
        return 0.5
    for key in (
        "latest_precutoff_estimate", "prior_year_q_eps", "current_eps",
        "value", "score"
    ):
        x = entity.get(key)
        if isinstance(x, (int, float)) and math.isfinite(float(x)):
            return float(x)
    return 0.0


def _fallback_label(task: dict, entity: dict, point: float) -> str | None:
    labels = [str(x) for x in task.get("target", {}).get("labels", [])]
    if not labels:
        return None
    family = task.get("family")
    if family == "eps_yoy_direction":
        prior = entity.get("prior_year_q_eps")
        if isinstance(prior, (int, float)):
            candidate = "up" if point > float(prior) else "down"
            if candidate in labels:
                return candidate
    if family == "macro_revision_direction":
        prior = entity.get("latest_precutoff_estimate")
        if isinstance(prior, (int, float)):
            candidate = "up" if point > float(prior) else "down"
            if candidate in labels:
                return candidate
    if family == "post_earnings_reaction":
        candidate = "positive_reaction" if point > 1 else ("negative_reaction" if point < -1 else "flat")
        if candidate in labels:
            return candidate
    return labels[0]


def _batch_prompt(task: dict, entities: list[dict], by_entity: dict[str, list[Chunk]]) -> tuple[str, str]:
    target = task.get("target", {})
    items = []
    for entity in entities:
        eid = str(entity.get("entity_id", ""))
        items.append(
            {
                "entity_id": eid,
                "entity": {k: v for k, v in entity.items() if k != "corpus_ref"},
                "evidence_candidates": _payload_candidates(by_entity[eid]),
            }
        )
    system = (
        "You are a financial forecasting agent in a sealed benchmark. "
        "Use ONLY supplied pre-cutoff evidence. For each entity forecast the task target, "
        "give a 90% interval, and select one or two candidate_id values that most strongly "
        "support the forecast. Evidence must support the prediction rather than merely describe "
        "the entity. Return one JSON object only. Do not use outside knowledge."
    )
    user = json.dumps(
        {
            "task_prompt": task.get("prompt", ""),
            "family": task.get("family", ""),
            "cutoff_date": task.get("cutoff_date", ""),
            "target": target,
            "interval_level": task.get("interval_level", 0.90),
            "output_schema": {
                "predictions": [
                    {
                        "entity_id": "string",
                        "label": "allowed label or null",
                        "point_forecast": 0.0,
                        "rank": "integer or null",
                        "interval": {"lo": 0.0, "hi": 0.0},
                        "candidate_ids": ["c1"],
                    }
                ]
            },
            "entities": items,
        },
        ensure_ascii=False,
    )
    return system, user


def _claim(eid: str, chunk: Chunk) -> dict[str, Any]:
    return {
        "doc_id": chunk.doc_id,
        "span_start": chunk.span_start,
        "span_end": chunk.span_end,
        "claim": f"Pre-cutoff evidence selected for {eid}.",
    }


def generic_run(task: dict, corpus_dir: Path, config: Config, use_mock: bool) -> dict:
    corpus = build_index(corpus_dir)
    index = BM25Index(corpus.chunks, task["cutoff_date"])
    config = replace(config, max_retries=1)
    http = HTTPModelClient(config)
    entities = [x for x in task.get("entities", []) if isinstance(x, dict)]
    by_entity = {
        str(e.get("entity_id", "")): _candidates(task, e, index)
        for e in entities
    }

    parsed_by_entity: dict[str, dict] = {}
    if not use_mock and config.model_endpoint:
        batches = [entities[i:i+BATCH_SIZE] for i in range(0, len(entities), BATCH_SIZE)]
        for batch in batches[:MAX_HOUSE_CALLS_GENERIC]:
            try:
                system, user = _batch_prompt(task, batch, by_entity)
                parsed = _parse_model_json(http.complete(system, user))
                for item in parsed.get("predictions", []):
                    if isinstance(item, dict) and item.get("entity_id"):
                        parsed_by_entity[str(item["entity_id"])] = item
            except Exception:
                continue

    target = task.get("target", {})
    target_type = target.get("type", "classification")
    allowed_labels = [str(x) for x in target.get("labels", [])]
    predictions: list[dict] = []

    for entity in entities:
        eid = str(entity.get("entity_id", ""))
        raw = parsed_by_entity.get(eid, {})
        p = raw.get("point_forecast")
        point = float(p) if isinstance(p, (int, float)) and math.isfinite(float(p)) else _fallback_point(task, entity)
        label = raw.get("label") if isinstance(raw.get("label"), str) else _fallback_label(task, entity, point)
        if allowed_labels and label not in allowed_labels:
            label = _fallback_label(task, entity, point)

        chunks = by_entity[eid]
        selected: list[Chunk] = []
        ids = raw.get("candidate_ids")
        if isinstance(ids, list):
            for cid in ids[:2]:
                if isinstance(cid, str) and cid.startswith("c"):
                    try:
                        idx = int(cid[1:]) - 1
                    except ValueError:
                        continue
                    if 0 <= idx < len(chunks):
                        selected.append(chunks[idx])
        if not selected and chunks:
            selected = [chunks[0]]

        predictions.append(
            {
                "entity_id": eid,
                "label": label,
                "point_forecast": point,
                "interval": _safe_interval(task, point, raw.get("interval")),
                "claims": [_claim(eid, c) for c in selected],
            }
        )

    if target_type == "ranking":
        order = sorted(predictions, key=lambda r: (-float(r["point_forecast"]), r["entity_id"]))
        ranks = {r["entity_id"]: i + 1 for i, r in enumerate(order)}
        for r in predictions:
            r["rank"] = ranks[r["entity_id"]]

    # Formatter self-check via EntityResult wrapper.
    results = [EntityResult(prediction=p, dropped_claims=0, model_raw="batched") for p in predictions]
    answer = build_answer(task, results, corpus)
    answer["target_type"] = target_type
    answer["notes"]["house_call_cap"] = MAX_HOUSE_CALLS_GENERIC
    answer["notes"]["house_batch_size"] = BATCH_SIZE
    answer["notes"]["retrieval_top_k"] = TOP_K
    answer["notes"]["fallback"] = "deterministic_task_fallback"
    return answer


def auction_run(task: dict, corpus_dir: Path, config: Config, use_mock: bool) -> dict:
    corpus = build_index(corpus_dir)
    index = BM25Index(corpus.chunks, task["cutoff_date"])
    config = replace(config, max_retries=1)
    client = None if use_mock else HTTPModelClient(config)
    return run_auction(task, index, corpus, client)


def emergency_answer(task: dict, corpus_dir: Path) -> dict:
    config = Config.from_env()
    corpus = build_index(corpus_dir)
    index = BM25Index(corpus.chunks, task.get("cutoff_date", "9999-12-31"))
    mock = MockModelClient(reply=_mock_reply)
    from strong_rag_baseline.agent import run_entity
    results = [
        run_entity(task, entity, index, corpus, mock, max(1, min(TOP_K, config.top_k)))
        for entity in task.get("entities", [])
    ]
    answer = build_answer(task, results, corpus)
    answer["target_type"] = task.get("target", {}).get("type")
    answer["notes"]["emergency_fallback"] = True
    return answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("verb", nargs="?", default="analyze", choices=["analyze"])
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args(argv)

    task = json.loads(args.task.read_text(encoding="utf-8"))
    config = Config.from_env()
    try:
        if task.get("family") == "auction_demand":
            answer = auction_run(task, args.corpus, config, args.mock)
        else:
            answer = generic_run(task, args.corpus, config, args.mock)
    except Exception as exc:
        answer = emergency_answer(task, args.corpus)
        answer.setdefault("notes", {})["caught_exception_type"] = type(exc).__name__

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(answer, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} with {len(answer.get('entity_predictions', []))} entities; family={task.get('family','')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from auction_family import run_auction
from eps_growth_family import run_eps_growth
from output_contract import OutputContractError, atomic_write_json, finalize_answer
from safe_calibration import apply_safe_calibration
from v6_e1_eps_adapter import apply_v6_e1_eps_adapter
from strong_rag_baseline.agent import EntityResult, _parse_model_json
from strong_rag_baseline.client import HTTPModelClient
from strong_rag_baseline.config import Config
from strong_rag_baseline.formatter import build_answer
from strong_rag_baseline.indexer import Chunk, IndexedCorpus, build_index
from strong_rag_baseline.retriever import BM25Index

BATCH_SIZE = 6
TOP_K = 5
MAX_HOUSE_CALLS_GENERIC = 4
FINALIZATION_RESERVE_S = 30.0

_STABILITY_FAMILIES = {"positioning_shift", "credit_event"}
_STABILITY_ALT_SEED_OFFSET = 104729

_POSTEARN_TERMS = (
    "revenue",
    "sales",
    "operating income",
    "operating margin",
    "gross margin",
    "earnings",
    "guidance",
    "outlook",
    "forecast",
    "expect",
)


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _target_type(task: Mapping[str, Any]) -> str:
    target = task.get("target")
    from_target = target.get("type") if isinstance(target, Mapping) else None
    from_top = task.get("target_type")
    if from_target is not None and from_top is not None and from_target != from_top:
        raise ValueError(
            f"task target type conflict: {from_target!r} vs {from_top!r}"
        )
    value = from_target or from_top
    if value not in {"classification", "regression", "ranking"}:
        raise ValueError(f"missing/unsupported target type: {value!r}")
    return str(value)


def _query(task: dict, entity: dict) -> str:
    target = task.get("target", {})
    parts = [
        str(entity.get(k, ""))
        for k in (
            "entity_id",
            "name",
            "ticker",
            "symbol",
            "series_id",
            "cik",
            "description",
            "sector",
            "tenor",
        )
    ]
    parts += [
        str(target.get("name", "")),
        str(task.get("family", "")),
        str(task.get("prompt", "")),
        "forecast guidance outlook results historical target",
    ]
    if task.get("family") == "post_earnings_reaction":
        parts.append(
            "revenue sales operating income margin earnings guidance outlook"
        )
    return " ".join(x for x in parts if x)


def _doc_matches_entity(entity: dict, chunk: Chunk) -> bool:
    cik = str(entity.get("cik") or "")
    if cik:
        return cik in chunk.doc_id
    series_id = str(entity.get("series_id") or "")
    if series_id and series_id.upper() in chunk.doc_id.upper():
        return True
    return True


def _chunk_relevance(task: dict, chunk: Chunk, base_score: float) -> float:
    score = float(base_score)
    if task.get("family") == "post_earnings_reaction":
        text = chunk.text.lower()
        strong_terms = (
            "revenue",
            "net sales",
            "operating income",
            "operating margin",
            "gross margin",
            "earnings per share",
            "diluted earnings",
            "year-over-year",
            "year over year",
            "guidance",
            "outlook",
        )
        business_hits = sum(term in text for term in strong_terms)
        score = 12.0 * business_hits + 0.15 * float(base_score)
        if any(
            noise in text
            for noise in (
                "table of contents",
                "signature",
                "securities registered pursuant",
                "cover page",
                "investor relations website",
                "tax court",
                "rivian",
            )
        ):
            score -= 8.0
    return score


def _candidates(task: dict, entity: dict, index: BM25Index) -> list[Chunk]:
    query = _query(task, entity)
    scored = index.search(query, max(80, TOP_K))
    scoped = [
        (s.chunk, _chunk_relevance(task, s.chunk, s.score))
        for s in scored
        if _doc_matches_entity(entity, s.chunk)
    ]

    cik = str(entity.get("cik") or "")
    if cik and len(scoped) < TOP_K:
        existing = {(c.doc_id, c.span_start, c.span_end) for c, _ in scoped}
        q_terms = {x for x in query.lower().split() if len(x) > 2}
        for chunk in index.chunks:
            if cik not in chunk.doc_id:
                continue
            key = (chunk.doc_id, chunk.span_start, chunk.span_end)
            if key in existing:
                continue
            overlap = len(q_terms & set(chunk.text.lower().split()))
            scoped.append((chunk, _chunk_relevance(task, chunk, float(overlap))))
            existing.add(key)

    scoped.sort(
        key=lambda x: (-x[1], x[0].doc_id, x[0].span_start)
    )
    chunks = [c for c, _ in scoped[:TOP_K]]

    if chunks:
        return chunks

    if cik:
        return []

    broad = [s.chunk for s in index.search(query, TOP_K)]
    if not broad and index.chunks:
        broad = [index.chunks[0]]
    return broad


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
        if _finite_number(lo) and _finite_number(hi):
            lo_f, hi_f = float(lo), float(hi)
            if lo_f <= hi_f:
                return {"level": level, "lo": lo_f, "hi": hi_f}
    if task.get("family") == "credit_event":
        return {"level": level, "lo": 0.0, "hi": 1.0}
    half = max(1.0, abs(float(point)) * 0.5)
    return {"level": level, "lo": float(point) - half, "hi": float(point) + half}


def _fallback_point(task: dict, entity: dict) -> float:
    if task.get("family") == "credit_event":
        return 0.5
    for key in (
        "latest_precutoff_estimate",
        "prior_year_q_eps",
        "current_eps",
        "value",
        "score",
    ):
        x = entity.get(key)
        if _finite_number(x):
            return float(x)
    return 0.0


def _fallback_label(task: dict, entity: dict, point: float) -> str | None:
    labels = [str(x) for x in task.get("target", {}).get("labels", [])]
    if not labels:
        return None
    family = task.get("family")
    if family == "eps_yoy_direction":
        prior = entity.get("prior_year_q_eps")
        if _finite_number(prior):
            candidate = "up" if point > float(prior) else "down"
            if candidate in labels:
                return candidate
    if family == "macro_revision_direction":
        prior = entity.get("latest_precutoff_estimate")
        if _finite_number(prior):
            candidate = "up" if point > float(prior) else "down"
            if candidate in labels:
                return candidate
    if family == "post_earnings_reaction":
        candidate = (
            "positive_reaction"
            if point > 1
            else ("negative_reaction" if point < -1 else "flat")
        )
        if candidate in labels:
            return candidate
    return labels[0]


def _batch_prompt(
    task: dict,
    entities: list[dict],
    by_entity: dict[str, list[Chunk]],
) -> tuple[str, str]:
    target = task.get("target", {})
    target_type = _target_type(task)
    items = []
    for entity in entities:
        eid = str(entity.get("entity_id", ""))
        items.append(
            {
                "entity_id": eid,
                "entity": {
                    k: v for k, v in entity.items() if k != "corpus_ref"
                },
                "evidence_candidates": _payload_candidates(by_entity[eid]),
            }
        )

    row_schema: dict[str, Any] = {
        "entity_id": "string",
        "point_forecast": 0.0,
        "interval": {"lo": 0.0, "hi": 0.0},
        "candidate_ids": ["c1"],
    }
    if target_type == "classification":
        row_schema["label"] = "one allowed task label"
    if target_type == "ranking":
        row_schema["rank"] = "integer 1..n"

    unit_hint = ""
    family = str(task.get("family") or "")
    if family == "post_earnings_reaction":
        unit_hint = (
            " The target is the market-adjusted one-day abnormal return after the earnings "
            "release, not whether company fundamentals merely improved year over year. "
            "Classify the likely MARKET REACTION as positive_reaction, negative_reaction, or "
            "flat. Treat flat as a narrow +/-1 percentage-point reaction band, not as a safe "
            "default. Use revenue, margins, EPS, guidance/outlook and explicit surprise cues "
            "to infer whether the release is likely to exceed or disappoint market expectations. "
            "The numeric point and interval are abnormal return versus SPY in PERCENTAGE POINTS; "
            "1 means +1%, not 0.01. Do not use +/-1% as the 90% interval."
        )
    elif family == "rate_curve_cross_section":
        unit_hint = (
            " The target is the CHANGE in each Treasury constant-maturity yield between the "
            "cutoff close and resolution close, measured in BASIS POINTS, not the yield level. "
            "Use start_yield_pct as the anchor and reason about both the common level move and "
            "maturity-specific curve shape. A 0.50 percentage-point yield move equals 50 bps. "
            "The horizon spans an inter-meeting window, so do not default to an unrealistically "
            "tight +/-1 bp interval."
        )

    system = (
        "You are a financial forecasting agent in a sealed benchmark. "
        "Use ONLY supplied pre-cutoff evidence. For each entity forecast the task target, "
        "give a 90% interval, and select one or two candidate_id values that most strongly "
        "support the forecast. Evidence must belong to the correct entity when entity-specific "
        "filings are provided. Return one JSON object only. Do not use outside knowledge."
        + unit_hint
    )
    user = json.dumps(
        {
            "task_prompt": task.get("prompt", ""),
            "family": task.get("family", ""),
            "cutoff_date": task.get("cutoff_date", ""),
            "target": target,
            "target_type": target_type,
            "interval_level": task.get("interval_level", 0.90),
            "output_schema": {"predictions": [row_schema]},
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


def _prediction_row(
    task: dict,
    entity: dict,
    raw: Mapping[str, Any],
    chunks: list[Chunk],
) -> dict[str, Any]:
    target = task.get("target", {})
    target_type = _target_type(task)
    allowed_labels = [str(x) for x in target.get("labels", [])]
    eid = str(entity.get("entity_id", ""))

    p = raw.get("point_forecast")
    point = float(p) if _finite_number(p) else _fallback_point(task, entity)

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

    row: dict[str, Any] = {
        "entity_id": eid,
        "point_forecast": point,
        "interval": _safe_interval(task, point, raw.get("interval")),
        "claims": [_claim(eid, c) for c in selected],
    }

    if target_type == "classification":
        label = (
            raw.get("label")
            if isinstance(raw.get("label"), str)
            else _fallback_label(task, entity, point)
        )
        if allowed_labels and label not in allowed_labels:
            label = _fallback_label(task, entity, point)
        row["label"] = label

    return row



SPECIALIZED_RUNNERS = {
    "eps_growth_regression": run_eps_growth,
}


def specialized_run(
    task: dict,
    corpus_dir: Path,
    family: str,
) -> dict:
    corpus = build_index(corpus_dir, task["cutoff_date"])
    index = BM25Index(corpus.chunks, task["cutoff_date"])
    runner = SPECIALIZED_RUNNERS[family]
    predictions, adapter_notes = runner(task, index, corpus)
    results = [
        EntityResult(prediction=p, dropped_claims=0, model_raw=f"{family}:deterministic")
        for p in predictions
    ]
    answer = build_answer(task, results, corpus)
    answer["target_type"] = _target_type(task)
    answer["notes"].update(adapter_notes)
    return answer

def _collect_house_seed(
    task: dict,
    entities: list[dict],
    by_entity: dict[str, list[Chunk]],
    config: Config,
    started: float,
    seed: int,
    call_budget: int,
) -> tuple[dict[str, dict], int]:
    parsed_by_entity: dict[str, dict] = {}
    calls = 0
    batches = [
        entities[i : i + BATCH_SIZE]
        for i in range(0, len(entities), BATCH_SIZE)
    ]
    for batch in batches:
        if calls >= call_budget:
            break
        remaining = config.unit_timeout_s - (time.monotonic() - started)
        if remaining <= FINALIZATION_RESERVE_S + 10:
            break
        allowed_ids = {str(e.get("entity_id", "")) for e in batch}
        call_config = replace(
            config,
            seed=seed,
            timeout_s=min(
                config.timeout_s,
                max(10.0, remaining - FINALIZATION_RESERVE_S),
            ),
        )
        http = HTTPModelClient(call_config)
        calls += 1
        try:
            system, user = _batch_prompt(task, batch, by_entity)
            parsed = _parse_model_json(http.complete(system, user))
            items = parsed.get("predictions")
            if not isinstance(items, list):
                continue
            seen: set[str] = set()
            for item in items:
                if not isinstance(item, dict):
                    continue
                eid = str(item.get("entity_id") or "")
                if eid not in allowed_ids or eid in seen:
                    continue
                parsed_by_entity[eid] = item
                seen.add(eid)
        except Exception:
            continue
    return parsed_by_entity, calls


def _mean_interval_rows(
    base: Mapping[str, Any],
    alt: Mapping[str, Any],
) -> dict[str, float] | None:
    a = base.get("interval")
    b = alt.get("interval")
    if not (isinstance(a, Mapping) and isinstance(b, Mapping)):
        return None
    alo, ahi = a.get("lo"), a.get("hi")
    blo, bhi = b.get("lo"), b.get("hi")
    if not all(_finite_number(x) for x in (alo, ahi, blo, bhi)):
        return None
    if float(alo) > float(ahi) or float(blo) > float(bhi):
        return None
    return {
        "level": float(task_level)
        if (task_level := a.get("level", b.get("level"))) is not None
        and _finite_number(task_level)
        else 0.90,
        "lo": (float(alo) + float(blo)) / 2.0,
        "hi": (float(ahi) + float(bhi)) / 2.0,
    }


def _aggregate_stability_predictions(
    task: dict,
    entities: list[dict],
    by_entity: dict[str, list[Chunk]],
    base_raw: dict[str, dict],
    alt_raw: dict[str, dict],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    family = str(task.get("family") or "")
    base_rows = {
        str(entity.get("entity_id", "")): _prediction_row(
            task,
            entity,
            base_raw.get(str(entity.get("entity_id", "")), {}),
            by_entity[str(entity.get("entity_id", ""))],
        )
        for entity in entities
    }
    alt_rows = {
        str(entity.get("entity_id", "")): _prediction_row(
            task,
            entity,
            alt_raw.get(str(entity.get("entity_id", "")), {}),
            by_entity[str(entity.get("entity_id", ""))],
        )
        for entity in entities
    }

    actions: dict[str, str] = {}
    predictions: list[dict[str, Any]] = []

    for entity in entities:
        eid = str(entity.get("entity_id", ""))
        base = dict(base_rows[eid])
        alt = alt_rows[eid]
        base_has_house = eid in base_raw
        alt_has_house = eid in alt_raw

        if family == "positioning_shift":
            if (
                base_has_house
                and alt_has_house
                and _finite_number(base.get("point_forecast"))
                and _finite_number(alt.get("point_forecast"))
            ):
                base["point_forecast"] = (
                    float(base["point_forecast"]) + float(alt["point_forecast"])
                ) / 2.0
                interval = _mean_interval_rows(base_rows[eid], alt_rows[eid])
                if interval is not None:
                    base["interval"] = interval
                actions[eid] = "average_two_seed_points"
            else:
                actions[eid] = "base_seed_fallback"
            predictions.append(base)
            continue

        if family == "credit_event":
            if not (base_has_house and alt_has_house):
                actions[eid] = "base_seed_fallback"
                predictions.append(base)
                continue
            if base.get("label") != alt.get("label"):
                actions[eid] = "base_on_label_disagreement"
                predictions.append(base)
                continue
            if not (
                _finite_number(base.get("point_forecast"))
                and _finite_number(alt.get("point_forecast"))
            ):
                actions[eid] = "base_on_invalid_point"
                predictions.append(base)
                continue
            base["point_forecast"] = (
                float(base["point_forecast"]) + float(alt["point_forecast"])
            ) / 2.0
            interval = _mean_interval_rows(base_rows[eid], alt_rows[eid])
            if interval is not None:
                base["interval"] = interval
            actions[eid] = "agreed_label_average_point"
            predictions.append(base)
            continue

        raise ValueError(f"unsupported stability family: {family}")

    if family == "positioning_shift":
        order = sorted(
            predictions,
            key=lambda r: (-float(r["point_forecast"]), r["entity_id"]),
        )
        ranks = {r["entity_id"]: i + 1 for i, r in enumerate(order)}
        for row in predictions:
            row["rank"] = ranks[row["entity_id"]]

    notes = {
        "house_stability_ensemble": True,
        "house_stability_family": family,
        "house_stability_actions": actions,
        "house_stability_base_seed": int(task.get("_base_seed", 0)),
        "house_stability_alt_seed_offset": _STABILITY_ALT_SEED_OFFSET,
        "house_stability_base_house_rows": len(base_raw),
        "house_stability_alt_house_rows": len(alt_raw),
    }
    return predictions, notes


def generic_run(
    task: dict,
    corpus_dir: Path,
    config: Config,
    use_mock: bool,
) -> dict:
    corpus = build_index(corpus_dir, task["cutoff_date"])
    index = BM25Index(corpus.chunks, task["cutoff_date"])
    config = replace(config, max_retries=1)
    entities = [x for x in task.get("entities", []) if isinstance(x, dict)]
    by_entity = {
        str(e.get("entity_id", "")): _candidates(task, e, index)
        for e in entities
    }

    parsed_by_entity: dict[str, dict] = {}
    house_calls = 0
    started = time.monotonic()
    stability_notes: dict[str, Any] = {}

    family = str(task.get("family") or "")
    if (
        family in _STABILITY_FAMILIES
        and not use_mock
        and config.model_endpoint
    ):
        base_seed = int(config.seed)
        alt_seed = base_seed + _STABILITY_ALT_SEED_OFFSET
        base_raw, base_calls = _collect_house_seed(
            task,
            entities,
            by_entity,
            config,
            started,
            base_seed,
            MAX_HOUSE_CALLS_GENERIC // 2,
        )
        house_calls += base_calls
        alt_budget = MAX_HOUSE_CALLS_GENERIC - house_calls
        alt_raw, alt_calls = _collect_house_seed(
            task,
            entities,
            by_entity,
            config,
            started,
            alt_seed,
            alt_budget,
        )
        house_calls += alt_calls

        task_for_notes = dict(task)
        task_for_notes["_base_seed"] = base_seed
        predictions, stability_notes = _aggregate_stability_predictions(
            task_for_notes,
            entities,
            by_entity,
            base_raw,
            alt_raw,
        )
        parsed_by_entity = base_raw
    else:
        if not use_mock and config.model_endpoint:
            parsed_by_entity, house_calls = _collect_house_seed(
                task,
                entities,
                by_entity,
                config,
                started,
                int(config.seed),
                MAX_HOUSE_CALLS_GENERIC,
            )

        predictions = [
            _prediction_row(
                task,
                entity,
                parsed_by_entity.get(str(entity.get("entity_id", "")), {}),
                by_entity[str(entity.get("entity_id", ""))],
            )
            for entity in entities
        ]

    target_type = _target_type(task)
    if target_type == "ranking" and family not in _STABILITY_FAMILIES:
        order = sorted(
            predictions,
            key=lambda r: (-float(r["point_forecast"]), r["entity_id"]),
        )
        ranks = {r["entity_id"]: i + 1 for i, r in enumerate(order)}
        for row in predictions:
            row["rank"] = ranks[row["entity_id"]]

    results = [
        EntityResult(prediction=p, dropped_claims=0, model_raw="batched")
        for p in predictions
    ]
    answer = build_answer(task, results, corpus)
    answer["target_type"] = target_type
    answer["notes"]["house_calls_attempted"] = house_calls
    answer["notes"]["house_call_cap"] = MAX_HOUSE_CALLS_GENERIC
    answer["notes"]["house_batch_size"] = BATCH_SIZE
    answer["notes"]["retrieval_top_k"] = TOP_K
    answer["notes"]["fallback"] = "deterministic_task_fallback"
    answer["notes"].update(stability_notes)
    return answer


def auction_run(
    task: dict,
    corpus_dir: Path,
    config: Config,
    use_mock: bool,
) -> dict:
    corpus = build_index(corpus_dir, task["cutoff_date"])
    index = BM25Index(corpus.chunks, task["cutoff_date"])
    call_config = replace(
        config,
        max_retries=1,
        timeout_s=min(config.timeout_s, max(10.0, config.unit_timeout_s - 60.0)),
    )
    client = (
        None
        if use_mock or not config.model_endpoint
        else HTTPModelClient(call_config)
    )
    return run_auction(task, index, corpus, client)


def emergency_answer(task: dict, corpus_dir: Path) -> dict:
    family = str(task.get("family") or "")
    if family in SPECIALIZED_RUNNERS:
        answer = specialized_run(task, corpus_dir, family)
        answer.setdefault("notes", {})["emergency_fallback"] = True
        return answer

    corpus = build_index(corpus_dir, task["cutoff_date"])
    index = BM25Index(corpus.chunks, task["cutoff_date"])
    entities = [x for x in task.get("entities", []) if isinstance(x, dict)]
    predictions: list[dict[str, Any]] = []

    for entity in entities:
        chunks = _candidates(task, entity, index)
        predictions.append(_prediction_row(task, entity, {}, chunks))

    target_type = _target_type(task)
    if target_type == "ranking":
        order = sorted(
            predictions,
            key=lambda r: (-float(r["point_forecast"]), r["entity_id"]),
        )
        ranks = {r["entity_id"]: i + 1 for i, r in enumerate(order)}
        for row in predictions:
            row["rank"] = ranks[row["entity_id"]]

    results = [
        EntityResult(prediction=p, dropped_claims=0, model_raw="fallback")
        for p in predictions
    ]
    answer = build_answer(task, results, corpus)
    answer["target_type"] = target_type
    answer["notes"]["emergency_fallback"] = True
    return answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "verb", nargs="?", default="analyze", choices=["analyze"]
    )
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args(argv)

    task = json.loads(args.task.read_text(encoding="utf-8"))
    config = Config.from_env()
    primary_error: Exception | None = None

    try:
        family = str(task.get("family") or "")
        if family == "auction_demand":
            answer = auction_run(task, args.corpus, config, args.mock)
        elif family in SPECIALIZED_RUNNERS:
            answer = specialized_run(task, args.corpus, family)
        else:
            answer = generic_run(task, args.corpus, config, args.mock)
    except Exception as exc:
        primary_error = exc
        answer = emergency_answer(task, args.corpus)
        answer.setdefault("notes", {})["caught_exception_type"] = type(exc).__name__

    answer = apply_v6_e1_eps_adapter(task, answer, args.corpus)
    answer = apply_safe_calibration(task, answer)

    try:
        answer = finalize_answer(answer, task, args.corpus)
    except OutputContractError:
        if primary_error is not None or answer.get("notes", {}).get("emergency_fallback"):
            raise
        answer = emergency_answer(task, args.corpus)
        answer.setdefault("notes", {})["contract_repair_fallback"] = True
        answer = apply_v6_e1_eps_adapter(task, answer, args.corpus)
        answer = apply_safe_calibration(task, answer)
        answer = finalize_answer(answer, task, args.corpus)

    atomic_write_json(args.out, answer)
    print(
        f"wrote {args.out} with {len(answer.get('entity_predictions', []))} "
        f"entities; family={task.get('family','')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

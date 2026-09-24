from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping

from postearn_evidence import build_postearn_cards, indexed_card_payload
from strong_rag_baseline.agent import EntityResult, _parse_model_json
from strong_rag_baseline.client import HTTPModelClient, MockModelClient
from strong_rag_baseline.config import Config
from strong_rag_baseline.formatter import build_answer
from strong_rag_baseline.indexer import Chunk, build_index


def _candidate_payload(chunks: list[Chunk]) -> list[dict[str, Any]]:
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


def _prompt(
    task: dict[str, Any],
    entities: list[dict[str, Any]],
    cards: dict[str, Any],
    by_entity: dict[str, list[Chunk]],
    repair: bool = False,
) -> tuple[str, str]:
    system = (
        "You are an earnings-reaction forecasting agent in a sealed benchmark. "
        "Use ONLY the supplied pre-cutoff SEC evidence. The target is the one-day ABNORMAL return after the "
        "2024-02-01 after-close earnings release: company total return minus SPY over the task event window. "
        "positive_reaction means abnormal return > +1%, negative_reaction means < -1%, otherwise flat. "
        "Historical operating improvement is NOT itself evidence that the next release beats market expectations. "
        "If reliable pre-earnings consensus/expectation evidence is missing from the frozen SEC corpus, keep it missing; "
        "do not invent analyst consensus or use remembered 2024 outcomes. "
        "Balance operating momentum, explicit management guidance when present, and adverse/counter evidence. "
        "Return JSON only. For every requested entity provide a finite point_forecast in abnormal-return percentage points, "
        "a 90% interval, one label, probabilities for positive_reaction/flat/negative_reaction, and 1-2 candidate_ids."
    )
    if repair:
        system += (
            " This is a STRICT REPAIR CALL for entities omitted or malformed in an earlier response. "
            "Return each requested entity exactly once with a finite point_forecast."
        )

    payload = []
    for entity in entities:
        eid = str(entity["entity_id"])
        payload.append(
            {
                "entity_id": eid,
                "fact_card": indexed_card_payload(cards[eid], by_entity[eid]),
                "evidence_candidates": _candidate_payload(by_entity[eid]),
            }
        )

    user = json.dumps(
        {
            "task_prompt": task.get("prompt", ""),
            "cutoff_date": task.get("cutoff_date", ""),
            "target": task.get("target", {}),
            "interval_level": task.get("interval_level", 0.90),
            "output_schema": {
                "predictions": [
                    {
                        "entity_id": "string",
                        "point_forecast": 0.0,
                        "interval": {"lo": -10.0, "hi": 10.0},
                        "label": "flat",
                        "probabilities": {
                            "positive_reaction": 0.33,
                            "flat": 0.34,
                            "negative_reaction": 0.33,
                        },
                        "candidate_ids": ["c1"],
                    }
                ]
            },
            "entities": payload,
        },
        ensure_ascii=False,
    )
    return system, user


def _mock_reply(task: dict[str, Any]) -> str:
    # Mock values exercise schema only and are not selected from realized outcomes.
    points = {"AAPL": 0.2, "AMZN": 1.8, "META": 1.6}
    labels = {"AAPL": "flat", "AMZN": "positive_reaction", "META": "positive_reaction"}
    rows = []
    for entity in task["entities"]:
        eid = str(entity["entity_id"])
        p = points[eid]
        label = labels[eid]
        rows.append(
            {
                "entity_id": eid,
                "point_forecast": p,
                "interval": {"lo": p - 3.0, "hi": p + 3.0},
                "label": label,
                "probabilities": {
                    "positive_reaction": 0.70 if label == "positive_reaction" else 0.15,
                    "flat": 0.70 if label == "flat" else 0.15,
                    "negative_reaction": 0.70 if label == "negative_reaction" else 0.15,
                },
                "candidate_ids": ["c1", "c2"],
            }
        )
    return json.dumps({"predictions": rows})


def run_postearn_card(
    task: dict[str, Any],
    corpus_dir: Path,
    config: Config,
    use_mock: bool,
    max_house_calls: int,
    finalization_reserve_s: float,
    prediction_row: Callable[[dict, dict, Mapping[str, Any], list[Chunk]], dict[str, Any]],
    merge_valid: Callable[[dict[str, dict], Mapping[str, Any], set[str]], set[str]],
) -> dict[str, Any]:
    if str(task.get("family") or "") != "post_earnings_reaction":
        raise ValueError("postearn card runner only supports post_earnings_reaction")

    corpus = build_index(corpus_dir, task["cutoff_date"])
    cards, by_entity = build_postearn_cards(task, corpus)
    entities = [e for e in task.get("entities", []) if isinstance(e, dict)]
    parsed_by_entity: dict[str, dict] = {}
    repaired: set[str] = set()
    house_calls = 0
    repair_calls = 0
    prompt_chars = 0
    started = time.monotonic()

    if use_mock:
        parsed = _parse_model_json(_mock_reply(task))
        merge_valid(
            parsed_by_entity,
            parsed,
            {str(e["entity_id"]) for e in entities},
        )
    elif config.model_endpoint:
        remaining = config.unit_timeout_s - (time.monotonic() - started)
        if remaining > finalization_reserve_s + 10:
            call_cfg = replace(
                config,
                max_retries=1,
                timeout_s=min(config.timeout_s, max(10.0, remaining - finalization_reserve_s)),
            )
            system, user = _prompt(task, entities, cards, by_entity)
            prompt_chars += len(system) + len(user)
            house_calls += 1
            try:
                parsed = _parse_model_json(HTTPModelClient(call_cfg).complete(system, user))
                merge_valid(
                    parsed_by_entity,
                    parsed,
                    {str(e["entity_id"]) for e in entities},
                )
            except Exception:
                pass

        missing = [e for e in entities if str(e["entity_id"]) not in parsed_by_entity]
        if missing and house_calls < max_house_calls:
            remaining = config.unit_timeout_s - (time.monotonic() - started)
            if remaining > finalization_reserve_s + 10:
                call_cfg = replace(
                    config,
                    max_retries=1,
                    timeout_s=min(config.timeout_s, max(10.0, remaining - finalization_reserve_s)),
                )
                system, user = _prompt(task, missing, cards, by_entity, repair=True)
                prompt_chars += len(system) + len(user)
                house_calls += 1
                repair_calls += 1
                try:
                    parsed = _parse_model_json(HTTPModelClient(call_cfg).complete(system, user))
                    repaired = merge_valid(
                        parsed_by_entity,
                        parsed,
                        {str(e["entity_id"]) for e in missing},
                    )
                except Exception:
                    repaired = set()

    predictions = [
        prediction_row(
            task,
            entity,
            parsed_by_entity.get(str(entity["entity_id"]), {}),
            by_entity[str(entity["entity_id"])],
        )
        for entity in entities
    ]
    results = [
        EntityResult(prediction=row, dropped_claims=0, model_raw="v4_postearn_card")
        for row in predictions
    ]
    answer = build_answer(task, results, corpus)
    answer["target_type"] = "classification"
    fallback = [
        str(e["entity_id"]) for e in entities
        if str(e["entity_id"]) not in parsed_by_entity
    ]
    answer.setdefault("notes", {}).update(
        {
            "v4_postearn_card": True,
            "house_calls_attempted": house_calls,
            "house_repair_calls": repair_calls,
            "house_repaired_entities": sorted(repaired),
            "house_fallback_entities": fallback,
            "structured_card_count": len(cards),
            "prompt_chars": prompt_chars,
            "expectation_gap_status": {
                eid: cards[eid].expectation_gap_status for eid in sorted(cards)
            },
        }
    )
    return answer

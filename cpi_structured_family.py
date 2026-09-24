from __future__ import annotations

import json
import math
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping

import statistics

from strong_rag_baseline.agent import EntityResult, _parse_model_json
from strong_rag_baseline.client import HTTPModelClient, MockModelClient
from strong_rag_baseline.config import Config
from strong_rag_baseline.formatter import build_answer
from strong_rag_baseline.indexer import Chunk, IndexedCorpus, build_index

ALFRED_DOC = "ALFRED_CPI_COMPONENTS_20241031"
BLS_DOC = "BLS_CPI_RELEASE_20241010"
EIA_DOC = "EIA_GASREGW_20241028"

COL_ORDER = [
    "CPI_ALLITEMS",
    "CPI_CORE",
    "CPI_SHELTER",
    "CPI_ENERGY",
    "CPI_GASOLINE",
    "CPI_FOOD",
    "CPI_NEWVEH",
    "CPI_USEDCARS",
    "CPI_APPAREL",
    "CPI_MEDICAL",
    "CPI_TRANSPSVC",
]

GROUP_HINT = {
    "CPI_SHELTER": "The task explicitly flags shelter persistence as relevant.",
    "CPI_GASOLINE": "Use October pump-price evidence directionally; the target is seasonally adjusted CPI MoM, not the unadjusted pump-price percent change.",
    "CPI_ENERGY": "Gasoline market prices are relevant directional evidence for energy, but do not map them one-for-one into seasonally adjusted CPI.",
    "CPI_APPAREL": "The task explicitly flags apparel as volatile; avoid treating September's large move as automatically persistent.",
    "CPI_USEDCARS": "The task explicitly flags used vehicles as volatile; distinguish a one-month reversal from a persistent trend.",
    "CPI_MEDICAL": "The task explicitly flags medical care stability as relevant.",
}


def _doc_text(corpus_dir: Path, doc_id: str) -> str:
    data = json.loads((corpus_dir / f"{doc_id}.json").read_text(encoding="utf-8"))
    text = data.get("text")
    if not isinstance(text, str):
        raise ValueError(f"missing text for {doc_id}")
    return text


def _panel(corpus_dir: Path) -> dict[int, dict[str, float]]:
    text = _doc_text(corpus_dir, ALFRED_DOC)
    out: dict[int, dict[str, float]] = {}
    for m in re.finditer(r"2024-(\d{2})\s*\|\s*([^\n]+)", text):
        month = int(m.group(1))
        vals = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", m.group(2))]
        if len(vals) >= len(COL_ORDER):
            out[month] = dict(zip(COL_ORDER, vals[: len(COL_ORDER)]))
    if sorted(out) != list(range(1, 10)):
        raise ValueError(f"unexpected CPI panel months: {sorted(out)}")
    return out


def _chunks(corpus: IndexedCorpus, doc_id: str) -> list[Chunk]:
    xs = [c for c in corpus.chunks if c.doc_id == doc_id]
    xs.sort(key=lambda c: (c.span_start, c.span_end))
    if not xs:
        raise ValueError(f"missing indexed document {doc_id}")
    return xs


def _find_chunk(corpus: IndexedCorpus, doc_id: str, phrase: str) -> Chunk:
    phrase_l = phrase.lower()
    xs = [c for c in _chunks(corpus, doc_id) if phrase_l in c.text.lower()]
    if not xs:
        raise ValueError(f"phrase not found in {doc_id}: {phrase}")
    return min(xs, key=lambda c: (len(c.text), c.span_start))


def build_cpi_context(
    task: dict[str, Any],
    corpus_dir: Path,
    corpus: IndexedCorpus,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[Chunk]], dict[str, Any]]:
    panel = _panel(corpus_dir)
    alfred_table = _find_chunk(corpus, ALFRED_DOC, "month | All items")
    alfred_notes = _find_chunk(corpus, ALFRED_DOC, "NOTES (derived from the table above)")
    bls_sa = _find_chunk(corpus, BLS_DOC, "For analyzing short-term price trends")
    eia_note = _find_chunk(corpus, EIA_DOC, "NOTE (derived)")

    cards: dict[str, dict[str, Any]] = {}
    candidates: dict[str, list[Chunk]] = {}

    for entity in task.get("entities", []):
        eid = str(entity["entity_id"])
        hist = [panel[m][eid] for m in range(1, 10)]
        trailing3 = float(statistics.fmean(hist[-3:]))
        trailing6 = float(statistics.fmean(hist[-6:]))
        sd = float(statistics.pstdev(hist))
        card = {
            "entity_id": eid,
            "name": entity.get("name"),
            "series_fred": entity.get("series_fred"),
            "target_unit": entity.get("unit"),
            "target_ref_month": entity.get("ref_month"),
            "latest_published_ref_month": entity.get("latest_published_ref_month"),
            "latest_first_print_mom_pct": float(entity.get("latest_published_mom_pct")),
            "history_2024_jan_sep_sa_mom_pct": hist,
            "trailing3_mean": trailing3,
            "trailing6_mean": trailing6,
            "history_sd": sd,
            "history_min": float(min(hist)),
            "history_max": float(max(hist)),
            "latest_minus_trailing3": float(hist[-1] - trailing3),
            "interpretation_hint": GROUP_HINT.get(
                eid,
                "Use the component's own frozen history and the task definition; do not infer post-cutoff values.",
            ),
        }
        if eid in {"CPI_GASOLINE", "CPI_ENERGY"}:
            card["eia_guardrail"] = (
                "The EIA series is an unadjusted market-price signal. Use direction/magnitude as contextual evidence only. "
                "Do not equate its monthly percent change to the seasonally adjusted CPI target."
            )
            candidates[eid] = [alfred_notes, alfred_table, eia_note, bls_sa]
        else:
            candidates[eid] = [alfred_notes, alfred_table, bls_sa]
        cards[eid] = card

    diagnostics = {
        "card_count": len(cards),
        "candidate_docs_by_entity": {
            eid: [c.doc_id for c in chunks] for eid, chunks in candidates.items()
        },
        "eia_entities": [
            eid for eid, chunks in candidates.items() if any(c.doc_id == EIA_DOC for c in chunks)
        ],
        "history_months": list(range(1, 10)),
    }
    return cards, candidates, diagnostics


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


def cpi_batch_prompt(
    task: dict[str, Any],
    entities: list[dict[str, Any]],
    cards: dict[str, dict[str, Any]],
    by_entity: dict[str, list[Chunk]],
    repair: bool = False,
) -> tuple[str, str]:
    system = (
        "You are a CPI nowcasting agent in a sealed benchmark. Use ONLY the supplied pre-cutoff frozen evidence. "
        "Forecast OCTOBER 2024 FIRST-PUBLISHED seasonally adjusted month-over-month CPI percent changes. "
        "Each entity has a structured fact card derived mechanically from the frozen ALFRED table. "
        "Anchor on the component's own first-print history, persistence/mean reversion and volatility. "
        "For gasoline and energy, EIA retail pump prices are directional market-price evidence only: they are NOT "
        "seasonally adjusted CPI and must not be mapped one-for-one into the target. "
        "Do not use outside knowledge, revised CPI values, or any post-cutoff October CPI release. "
        "For every requested entity return a finite point_forecast, a 90% interval, and one or two candidate_id "
        "values that best support the forecast. Return one JSON object only."
    )
    if repair:
        system += (
            " This is a STRICT REPAIR CALL for entities omitted or malformed in an earlier response. "
            "Return every requested entity exactly once and preserve the same forecasting semantics."
        )

    items = []
    for entity in entities:
        eid = str(entity["entity_id"])
        items.append(
            {
                "entity_id": eid,
                "fact_card": cards[eid],
                "evidence_candidates": _payload_candidates(by_entity[eid]),
            }
        )

    user = json.dumps(
        {
            "task_prompt": task.get("prompt", ""),
            "family": task.get("family", ""),
            "cutoff_date": task.get("cutoff_date", ""),
            "target": task.get("target", {}),
            "interval_level": task.get("interval_level", 0.90),
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
            "entities": items,
        },
        ensure_ascii=False,
    )
    return system, user


def _mock_reply(entities: list[dict[str, Any]], cards: dict[str, dict[str, Any]]) -> str:
    rows = []
    for entity in entities:
        eid = str(entity["entity_id"])
        card = cards[eid]
        point = 0.6 * float(card["latest_first_print_mom_pct"]) + 0.4 * float(card["trailing3_mean"])
        half = max(0.35, 1.64 * max(0.10, float(card["history_sd"])))
        rows.append(
            {
                "entity_id": eid,
                "point_forecast": point,
                "interval": {"lo": point - half, "hi": point + half},
                "candidate_ids": ["c1"],
            }
        )
    return json.dumps({"predictions": rows})


def run_cpi_structured(
    task: dict[str, Any],
    corpus_dir: Path,
    config: Config,
    use_mock: bool,
    batch_size: int,
    max_house_calls: int,
    finalization_reserve_s: float,
    prediction_row: Callable[[dict, dict, Mapping[str, Any], list[Chunk]], dict[str, Any]],
    merge_valid: Callable[[dict[str, dict], Mapping[str, Any], set[str]], set[str]],
) -> dict[str, Any]:
    if str(task.get("family") or "") != "cpi_component_nowcast":
        raise ValueError("CPI structured runner only supports cpi_component_nowcast")

    corpus = build_index(corpus_dir, task["cutoff_date"])
    cards, by_entity, diagnostics = build_cpi_context(task, corpus_dir, corpus)
    entities = [e for e in task.get("entities", []) if isinstance(e, dict)]
    parsed_by_entity: dict[str, dict] = {}
    house_calls = 0
    repair_calls = 0
    repaired: set[str] = set()
    prompt_chars = 0
    started = time.monotonic()

    if use_mock:
        raw = _mock_reply(entities, cards)
        parsed = _parse_model_json(raw)
        merge_valid(parsed_by_entity, parsed, {str(e["entity_id"]) for e in entities})
    elif config.model_endpoint:
        batches = [entities[i : i + batch_size] for i in range(0, len(entities), batch_size)]
        for batch in batches[:max_house_calls]:
            remaining = config.unit_timeout_s - (time.monotonic() - started)
            if remaining <= finalization_reserve_s + 10:
                break
            call_config = replace(
                config,
                max_retries=1,
                timeout_s=min(config.timeout_s, max(10.0, remaining - finalization_reserve_s)),
            )
            house_calls += 1
            allowed = {str(e["entity_id"]) for e in batch}
            try:
                system, user = cpi_batch_prompt(task, batch, cards, by_entity)
                prompt_chars += len(system) + len(user)
                parsed = _parse_model_json(HTTPModelClient(call_config).complete(system, user))
                merge_valid(parsed_by_entity, parsed, allowed)
            except Exception:
                continue

        missing = [e for e in entities if str(e["entity_id"]) not in parsed_by_entity]
        if missing and house_calls < max_house_calls:
            remaining = config.unit_timeout_s - (time.monotonic() - started)
            if remaining > finalization_reserve_s + 10:
                house_calls += 1
                repair_calls += 1
                allowed = {str(e["entity_id"]) for e in missing}
                try:
                    system, user = cpi_batch_prompt(task, missing, cards, by_entity, repair=True)
                    prompt_chars += len(system) + len(user)
                    call_config = replace(
                        config,
                        max_retries=1,
                        timeout_s=min(config.timeout_s, max(10.0, remaining - finalization_reserve_s)),
                    )
                    parsed = _parse_model_json(HTTPModelClient(call_config).complete(system, user))
                    repaired = merge_valid(parsed_by_entity, parsed, allowed)
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
        EntityResult(prediction=p, dropped_claims=0, model_raw="cpi_structured_packet_v1")
        for p in predictions
    ]
    answer = build_answer(task, results, corpus)
    answer["target_type"] = "regression"
    fallback = [str(e["entity_id"]) for e in entities if str(e["entity_id"]) not in parsed_by_entity]
    answer.setdefault("notes", {}).update(
        {
            "cpi_structured_packet": True,
            "cpi_packet_version": "v1",
            "house_calls_attempted": house_calls,
            "house_repair_calls": repair_calls,
            "house_repaired_entities": sorted(repaired),
            "house_fallback_entities": fallback,
            "house_batch_size": batch_size,
            "structured_card_count": diagnostics["card_count"],
            "eia_entities": diagnostics["eia_entities"],
            "prompt_chars": prompt_chars,
        }
    )
    return answer

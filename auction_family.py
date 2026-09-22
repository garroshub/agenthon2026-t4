from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass
from typing import Any

from strong_rag_baseline.agent import _parse_model_json
from strong_rag_baseline.client import ModelClient
from strong_rag_baseline.indexer import IndexedCorpus
from strong_rag_baseline.retriever import BM25Index

ROW_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})\s*\|\s*"
    r"(?P<term>[^|]+?)\s*\|\s*"
    r"(?P<issue>new|reopen)\s*\|\s*"
    r"(?P<size>\d+(?:\.\d+)?)\s*\|\s*"
    r"(?P<btc>\d+(?:\.\d+)?)\s*\|\s*"
    r"(?P<yield>\d+(?:\.\d+)?)\s*\|\s*"
    r"(?P<indirect>\d+(?:\.\d+)?)"
)

TENOR_KEYS = {
    "2-Year": "_2Y_",
    "3-Year": "_3Y_",
    "5-Year": "_5Y_",
    "7-Year": "_7Y_",
    "10-Year": "_10Y_",
    "20-Year": "_20Y_",
    "30-Year": "_30Y_",
}


@dataclass(frozen=True)
class Obs:
    date: str
    issue: str
    size: float
    btc: float


def parse_history(text: str) -> list[Obs]:
    rows: list[Obs] = []
    for m in ROW_RE.finditer(text):
        rows.append(
            Obs(
                date=m.group("date"),
                issue=m.group("issue"),
                size=float(m.group("size")),
                btc=float(m.group("btc")),
            )
        )
    return rows


def _persistence(hist: list[Obs], issue: str) -> float | None:
    return hist[-1].btc if hist else None


def _mean(hist: list[Obs], n: int) -> float | None:
    if len(hist) < n:
        return None
    return statistics.mean(x.btc for x in hist[-n:])


def _same_type(hist: list[Obs], issue: str) -> float | None:
    vals = [x.btc for x in hist if x.issue == issue]
    if len(vals) < 2:
        return None
    return statistics.mean(vals[-3:])


def _trend6(hist: list[Obs], issue: str) -> float | None:
    if len(hist) < 6:
        return None
    y = [x.btc for x in hist[-6:]]
    n = len(y)
    xbar = (n - 1) / 2
    ybar = statistics.mean(y)
    denom = sum((i - xbar) ** 2 for i in range(n))
    if denom <= 0:
        return y[-1]
    slope = sum((i - xbar) * (v - ybar) for i, v in enumerate(y)) / denom
    intercept = ybar - slope * xbar
    return intercept + slope * n


METHODS = {
    "persistence": _persistence,
    "mean3": lambda h, i: _mean(h, 3),
    "mean6": lambda h, i: _mean(h, 6),
    "same_issue_type": _same_type,
    "trend6": _trend6,
}

PENALTY = {
    "persistence": 0.0,
    "mean3": 0.002,
    "mean6": 0.002,
    "same_issue_type": 0.004,
    "trend6": 0.006,
}


def _walk_errors(hist: list[Obs], method_name: str, min_train: int = 6) -> list[float]:
    method = METHODS[method_name]
    errors: list[float] = []
    for i in range(min_train, len(hist)):
        pred = method(hist[:i], hist[i].issue)
        if pred is not None and math.isfinite(pred):
            errors.append(hist[i].btc - float(pred))
    return errors


def _choose_at_origin(hist: list[Obs], origin: int) -> tuple[str, float] | None:
    target = hist[origin]
    ranked: list[tuple[float, str, float]] = []
    for name, method in METHODS.items():
        errors = _walk_errors(hist[:origin], name)
        if len(errors) < 3:
            continue
        pred = method(hist[:origin], target.issue)
        if pred is None or not math.isfinite(pred):
            continue
        mae = statistics.mean(abs(e) for e in errors)
        ranked.append((mae + PENALTY[name], name, float(pred)))
    if not ranked:
        return None
    _, name, pred = min(ranked)
    return name, pred


def nested_errors(hist: list[Obs], start: int = 9) -> list[float]:
    out: list[float] = []
    for origin in range(start, len(hist)):
        choice = _choose_at_origin(hist, origin)
        if choice is None:
            continue
        _, pred = choice
        out.append(hist[origin].btc - pred)
    return out


def final_point(hist: list[Obs], target_issue: str) -> tuple[str, float]:
    ranked: list[tuple[float, str, float]] = []
    for name, method in METHODS.items():
        pred = method(hist, target_issue)
        if pred is None or not math.isfinite(pred):
            continue
        errors = _walk_errors(hist, name)
        if len(errors) < 3:
            continue
        mae = statistics.mean(abs(e) for e in errors)
        ranked.append((mae + PENALTY[name], name, float(pred)))
    if not ranked:
        return "persistence", float(hist[-1].btc if hist else 2.5)
    _, name, point = min(ranked)
    return name, point


def _quantile_abs(errors: list[float], level: float) -> float:
    vals = sorted(abs(float(e)) for e in errors)
    if not vals:
        return 0.25
    pos = level * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    w = pos - lo
    return vals[lo] * (1 - w) + vals[hi] * w


def calibrated_half_width(histories: dict[str, list[Obs]]) -> float:
    by_tenor = {k: nested_errors(v) for k, v in histories.items()}
    usable = {k: v for k, v in by_tenor.items() if v}
    all_errors = [e for vals in usable.values() for e in vals]
    if len(usable) < 2 or len(all_errors) < 8:
        return max(0.15, _quantile_abs(all_errors, 0.90))

    levels = [0.85, 0.875, 0.90, 0.925, 0.95]
    scored: list[tuple[float, float, float]] = []
    for level in levels:
        hits: list[bool] = []
        widths: list[float] = []
        for tenor, held in usable.items():
            train = [e for other, vals in usable.items() if other != tenor for e in vals]
            if not train:
                continue
            q = _quantile_abs(train, level)
            hits.extend(abs(e) <= q for e in held)
            widths.extend([2 * q] * len(held))
        if hits:
            coverage = sum(hits) / len(hits)
            scored.append((abs(coverage - 0.90), statistics.mean(widths), level))
    level = min(scored)[2] if scored else 0.90
    return max(0.10, _quantile_abs(all_errors, level))


def _history_doc_for_tenor(corpus: IndexedCorpus, tenor: str) -> str | None:
    key = TENOR_KEYS.get(tenor)
    if not key:
        return None
    for doc_id in corpus.doc_texts:
        if "TDIRECT_AUCTIONS_" in doc_id and key in doc_id:
            return doc_id
    return None


def _support_candidates(
    entity: dict,
    index: BM25Index,
    corpus: IndexedCorpus,
    limit: int = 8,
) -> list[dict]:
    tenor = str(entity.get("tenor", ""))
    key = TENOR_KEYS.get(tenor, "")
    query = (
        f"{entity.get('name','')} {tenor} bid-to-cover ratio auction demand "
        "historical results average range"
    )
    ranked = [s.chunk for s in index.search(query, 60)]
    filtered = [
        c for c in ranked
        if (key and key in c.doc_id) or "TDIRECT_UPCOMING" in c.doc_id
    ]
    chunks = filtered or ranked
    out = []
    seen = set()
    for chunk in chunks:
        k = (chunk.doc_id, chunk.span_start, chunk.span_end)
        if k in seen:
            continue
        seen.add(k)
        out.append(
            {
                "candidate_id": f"{entity.get('entity_id')}__{len(out)+1}",
                "doc_id": chunk.doc_id,
                "span_start": chunk.span_start,
                "span_end": chunk.span_end,
                "text": chunk.text,
            }
        )
        if len(out) >= limit:
            break
    return out


def _house_select(
    tasks: list[dict],
    client: ModelClient,
) -> dict[str, str]:
    system = (
        "You select evidence for fixed financial forecasts. Forecast values and intervals "
        "are immutable. For each entity choose exactly one candidate_id that best supports "
        "the forecasted level and stated 90% uncertainty using only the provided text. "
        "Do not invent evidence or alter numbers. Return one JSON object only."
    )
    user = json.dumps(
        {
            "output_schema": {
                "selections": [
                    {"entity_id": "string", "candidate_id": "string"}
                ]
            },
            "tasks": tasks,
        },
        ensure_ascii=False,
    )
    raw = client.complete(system, user)
    parsed = _parse_model_json(raw)
    out: dict[str, str] = {}
    for row in parsed.get("selections", []):
        if isinstance(row, dict) and row.get("entity_id") and row.get("candidate_id"):
            out[str(row["entity_id"])] = str(row["candidate_id"])
    return out


def run_auction(
    task: dict,
    index: BM25Index,
    corpus: IndexedCorpus,
    client: ModelClient | None,
) -> dict:
    entities = list(task.get("entities", []))
    histories: dict[str, list[Obs]] = {}
    doc_ids: dict[str, str] = {}
    for entity in entities:
        tenor = str(entity.get("tenor", ""))
        doc_id = _history_doc_for_tenor(corpus, tenor)
        if not doc_id:
            continue
        hist = parse_history(corpus.doc_texts[doc_id])
        if hist:
            histories[tenor] = hist
            doc_ids[tenor] = doc_id

    half_width = calibrated_half_width(histories)
    predictions: list[dict] = []
    support_tasks: list[dict] = []
    candidates_by_entity: dict[str, list[dict]] = {}

    for entity in entities:
        entity_id = str(entity.get("entity_id", ""))
        tenor = str(entity.get("tenor", ""))
        hist = histories.get(tenor, [])
        method, point = final_point(hist, str(entity.get("new_or_reopening", "new")))
        lo, hi = max(1.0, point - half_width), min(5.0, point + half_width)
        candidates = _support_candidates(entity, index, corpus)
        candidates_by_entity[entity_id] = candidates
        support_tasks.append(
            {
                "entity_id": entity_id,
                "fixed_forecast": {
                    "point_forecast": point,
                    "interval": {"level": task.get("interval_level", 0.90), "lo": lo, "hi": hi},
                },
                "candidates": candidates,
            }
        )
        predictions.append(
            {
                "entity_id": entity_id,
                "label": None,
                "point_forecast": point,
                "interval": {
                    "level": task.get("interval_level", 0.90),
                    "lo": lo,
                    "hi": hi,
                },
                "claims": [],
                "_method": method,
            }
        )

    selected: dict[str, str] = {}
    if client is not None and support_tasks:
        try:
            selected = _house_select(support_tasks, client)
        except Exception:
            selected = {}

    for pred in predictions:
        eid = pred["entity_id"]
        cands = candidates_by_entity.get(eid, [])
        chosen = None
        wanted = selected.get(eid)
        if wanted:
            chosen = next((c for c in cands if c["candidate_id"] == wanted), None)
        if chosen is None and cands:
            chosen = cands[0]
        if chosen is not None:
            pred["claims"] = [
                {
                    "doc_id": chosen["doc_id"],
                    "span_start": chosen["span_start"],
                    "span_end": chosen["span_end"],
                    "claim": "Historical Treasury auction evidence used to support the fixed bid-to-cover forecast.",
                }
            ]
        pred.pop("_method", None)

    return {
        "task_id": task.get("task_id", ""),
        "schema_version": task.get("schema_version", "3"),
        "target_type": task.get("target", {}).get("type"),
        "entity_predictions": predictions,
        "evidence_trace": (
            "auction_demand adapter: tenor-specific historical forecasting with "
            "nested residual interval calibration and one batched House evidence-selection call."
        ),
        "notes": {
            "agent": "hybrid_t4_dev_v1",
            "family_adapter": "auction_demand",
            "house_calls_planned": 1 if client is not None else 0,
        },
    }

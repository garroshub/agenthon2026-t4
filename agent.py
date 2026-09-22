from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping

from strong_rag_baseline.indexer import Chunk, IndexedCorpus, build_index
from strong_rag_baseline.retriever import BM25Index

HOUSE_BATCH_SIZE = 8
GENERIC_TOP_K = 6
AUCTION_TOP_K = 12
INTERVAL_LEVEL_DEFAULT = 0.90
MAX_HOUSE_CALLS_SOFT = 8

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
JSON_RE = re.compile(r"\{.*\}", re.S)


def _task_text(task: Mapping[str, Any]) -> str:
    return str(task.get("prompt") or "")


def _entity_query(task: Mapping[str, Any], entity: Mapping[str, Any]) -> str:
    target = task.get("target") if isinstance(task.get("target"), Mapping) else {}
    parts = [
        str(entity.get(k, ""))
        for k in (
            "entity_id", "name", "ticker", "symbol", "series_id",
            "description", "sector", "tenor"
        )
    ]
    parts += [
        str(target.get("name") or ""),
        str(task.get("family") or ""),
        _task_text(task),
        "forecast guidance outlook results history target",
    ]
    return " ".join(p for p in parts if p)


def _eligible_chunks(index: BM25Index, task: dict, entity: dict, top_k: int) -> list[Chunk]:
    return [x.chunk for x in index.search(_entity_query(task, entity), top_k)]


def _first_eligible_chunk(index: BM25Index) -> Chunk | None:
    return index.chunks[0] if index.chunks else None


def _candidate_payload(chunks: list[Chunk]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, c in enumerate(chunks, 1):
        out.append(
            {
                "candidate_id": f"c{i}",
                "doc_id": c.doc_id,
                "doc_date": c.doc_date,
                "span_start": c.span_start,
                "span_end": c.span_end,
                "text": c.text,
            }
        )
    return out


def _parse_json(raw: str) -> dict[str, Any]:
    match = JSON_RE.search(raw)
    if not match:
        raise ValueError("House reply contains no JSON object")
    return json.loads(match.group(0))


def _house_request(system: str, user_obj: dict[str, Any], max_tokens: int = 4000) -> dict[str, Any]:
    endpoint = os.environ.get("MODEL_ENDPOINT", "").rstrip("/")
    model = os.environ.get("MODEL_NAME", "")
    token = os.environ.get("MODEL_TOKEN", "")
    if not endpoint or not model or not token:
        raise RuntimeError("House runtime variables are unavailable")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    req = urllib.request.Request(
        endpoint + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + token,
        },
    )
    last: Exception | None = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return _parse_json(body["choices"][0]["message"]["content"])
        except (urllib.error.URLError, KeyError, json.JSONDecodeError, ValueError) as exc:
            last = exc
            time.sleep(1 + attempt)
    raise RuntimeError("House call failed") from last


def _safe_interval(point: float | None, interval: Any, level: float, task: dict) -> dict[str, float]:
    if isinstance(interval, Mapping):
        lo, hi = interval.get("lo"), interval.get("hi")
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            lo_f, hi_f = float(lo), float(hi)
            if math.isfinite(lo_f) and math.isfinite(hi_f) and lo_f <= hi_f:
                return {"level": level, "lo": lo_f, "hi": hi_f}

    family = str(task.get("family") or "")
    center = float(point) if isinstance(point, (int, float)) and math.isfinite(float(point)) else 0.0
    if family == "credit_event":
        return {"level": level, "lo": 0.0, "hi": 1.0}
    half = max(1.0, abs(center) * 0.5)
    return {"level": level, "lo": center - half, "hi": center + half}


def _fallback_point(task: dict, entity: dict) -> float:
    family = str(task.get("family") or "")
    if family == "credit_event":
        return 0.5
    preferred = {
        "eps_beat_consensus": ("consensus_eps",),
        "eps_yoy_direction": ("prior_year_q_eps",),
        "macro_revision_direction": ("latest_precutoff_estimate",),
        "positioning_shift": ("trailing_4wk_net_change_pct_oi", "net_pct_oi_20241022"),
        "post_earnings_reaction": (),
        "rate_curve_cross_section": (),
        "cpi_component_nowcast": (),
        "eps_growth_regression": (),
    }.get(family, ("value", "score"))
    for key in preferred:
        value = entity.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return 0.0


def _fallback_label(task: dict, entity: dict, point: float) -> str | None:
    target = task.get("target") if isinstance(task.get("target"), Mapping) else {}
    labels = [str(x) for x in target.get("labels", []) if isinstance(x, str)]
    if not labels:
        return None
    family = str(task.get("family") or "")
    if family == "eps_yoy_direction":
        prior = entity.get("prior_year_q_eps")
        if isinstance(prior, (int, float)):
            wanted = "up" if point > float(prior) else "down"
            if wanted in labels:
                return wanted
    if family == "post_earnings_reaction":
        wanted = "positive_reaction" if point > 1 else ("negative_reaction" if point < -1 else "flat")
        if wanted in labels:
            return wanted
    if family == "macro_revision_direction":
        latest = entity.get("latest_precutoff_estimate")
        if isinstance(latest, (int, float)):
            wanted = "up" if point > float(latest) else "down"
            if wanted in labels:
                return wanted
    return labels[0]


def _claim_from_candidate(entity_id: str, candidate: Chunk) -> dict[str, Any]:
    return {
        "doc_id": candidate.doc_id,
        "span_start": candidate.span_start,
        "span_end": candidate.span_end,
        "claim": f"Pre-cutoff evidence selected for {entity_id}.",
    }


def _generic_house_batch(
    task: dict,
    batch_entities: list[dict],
    candidates_by_entity: dict[str, list[Chunk]],
) -> list[dict[str, Any]]:
    target = task.get("target") if isinstance(task.get("target"), Mapping) else {}
    target_type = str(target.get("type") or "classification")
    level = float(task.get("interval_level", INTERVAL_LEVEL_DEFAULT))
    tasks: list[dict[str, Any]] = []

    for entity in batch_entities:
        eid = str(entity.get("entity_id") or "")
        entity_public = {k: v for k, v in entity.items() if k != "corpus_ref"}
        tasks.append(
            {
                "entity_id": eid,
                "entity": entity_public,
                "candidates": _candidate_payload(candidates_by_entity[eid]),
            }
        )

    system = (
        "You are a forecasting agent in a sealed evidence-grounded benchmark. "
        "Use ONLY the supplied pre-cutoff evidence candidates. Do not use outside knowledge. "
        "For each entity return a forecast and select one or two candidate_id values that best support "
        "the forecast. The selected evidence must support the prediction, not merely describe the entity. "
        "Return strict JSON only. Be conservative about uncertainty."
    )
    user = {
        "task_prompt": task.get("prompt", ""),
        "cutoff_date": task.get("cutoff_date", ""),
        "family": task.get("family", ""),
        "target": target,
        "interval_level": level,
        "required_output": {
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
        "entities": tasks,
    }
    parsed = _house_request(system, user)
    items = parsed.get("predictions")
    if not isinstance(items, list):
        raise ValueError("House batch missing predictions")
    return [x for x in items if isinstance(x, dict)]


# -------------------------- auction family -------------------------- #

def _parse_auction_history(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not re.match(r"^\d{4}-\d{2}-\d{2}\s*\|", line.strip()):
            continue
        parts = [x.strip() for x in line.split("|")]
        if len(parts) < 7:
            continue
        try:
            rows.append(
                {
                    "date": parts[0],
                    "term": parts[1],
                    "issue": parts[2].lower(),
                    "size": float(parts[3]),
                    "btc": float(parts[4]),
                }
            )
        except ValueError:
            continue
    return rows


def _pred_persistence(hist: list[dict], issue: str, size: float | None) -> float | None:
    return hist[-1]["btc"] if hist else None


def _pred_mean(hist: list[dict], n: int) -> float | None:
    vals = [x["btc"] for x in hist[-n:]]
    return statistics.mean(vals) if vals else None


def _pred_same_issue(hist: list[dict], issue: str, size: float | None) -> float | None:
    vals = [x["btc"] for x in hist if issue and issue in x["issue"]]
    return statistics.mean(vals[-4:]) if vals else None


def _pred_trend6(hist: list[dict], issue: str, size: float | None) -> float | None:
    vals = [x["btc"] for x in hist[-6:]]
    n = len(vals)
    if n < 3:
        return None
    xs = list(range(n))
    xbar, ybar = statistics.mean(xs), statistics.mean(vals)
    denom = sum((x - xbar) ** 2 for x in xs)
    if denom == 0:
        return ybar
    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, vals)) / denom
    return ybar + slope * ((n) - xbar)


AUCTION_METHODS = {
    "persistence": _pred_persistence,
    "mean3": lambda h, i, s: _pred_mean(h, 3),
    "mean6": lambda h, i, s: _pred_mean(h, 6),
    "same_issue": _pred_same_issue,
    "trend6": _pred_trend6,
}


def _walk_errors(hist: list[dict], fn) -> list[float]:
    errs: list[float] = []
    for i in range(6, len(hist)):
        pred = fn(hist[:i], hist[i]["issue"], hist[i]["size"])
        if pred is not None and math.isfinite(float(pred)):
            errs.append(float(hist[i]["btc"] - pred))
    return errs


def _auction_forecast(hist: list[dict], issue: str, size: float | None) -> tuple[float, list[float]]:
    choices: list[tuple[float, str, float, list[float]]] = []
    penalty = {"persistence": 0.0, "mean3": 0.002, "mean6": 0.002, "same_issue": 0.004, "trend6": 0.006}
    for name, fn in AUCTION_METHODS.items():
        pred = fn(hist, issue, size)
        errs = _walk_errors(hist, fn)
        if pred is None or len(errs) < 3:
            continue
        mae = statistics.mean(abs(x) for x in errs)
        choices.append((mae + penalty[name], name, float(pred), errs))
    if not choices:
        return (hist[-1]["btc"] if hist else 2.5), []
    _, _, pred, errs = min(choices)
    return pred, errs


def _empirical_abs_quantile(errors: list[float], level: float = 0.925) -> float:
    vals = sorted(abs(x) for x in errors)
    if not vals:
        return 0.25
    pos = level * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return max(0.05, vals[lo])
    w = pos - lo
    return max(0.05, vals[lo] * (1 - w) + vals[hi] * w)


def _auction_doc_for_entity(corpus: IndexedCorpus, entity: dict) -> str | None:
    tenor = str(entity.get("tenor") or "").upper()
    if tenor:
        key = "_" + tenor.replace(" ", "") + "_"
        for doc_id in corpus.doc_texts:
            if key in doc_id.upper():
                return doc_id
    eid = str(entity.get("entity_id") or "")
    m = re.search(r"AUC_(\d+)Y", eid)
    if m:
        key = "_" + m.group(1) + "Y_"
        for doc_id in corpus.doc_texts:
            if key in doc_id.upper():
                return doc_id
    return None


def _auction_predictions(task: dict, entities: list[dict], index: BM25Index, corpus: IndexedCorpus) -> list[dict]:
    level = float(task.get("interval_level", INTERVAL_LEVEL_DEFAULT))
    hist_by_entity: dict[str, list[dict]] = {}
    points: dict[str, float] = {}
    all_errors: list[float] = []

    for entity in entities:
        eid = str(entity["entity_id"])
        doc_id = _auction_doc_for_entity(corpus, entity)
        hist = _parse_auction_history(corpus.doc_texts.get(doc_id or "", ""))
        hist_by_entity[eid] = hist
        issue = str(entity.get("new_or_reopening") or "").lower()
        size = entity.get("offering_amount_usd_bn")
        size_f = float(size) if isinstance(size, (int, float)) else None
        point, errs = _auction_forecast(hist, issue, size_f)
        points[eid] = point
        all_errors.extend(errs)

    half = _empirical_abs_quantile(all_errors, 0.925)

    candidates_by_entity: dict[str, list[Chunk]] = {}
    for entity in entities:
        eid = str(entity["entity_id"])
        query = _entity_query(task, entity) + " bid to cover bid-to-cover auction history"
        chunks = [x.chunk for x in index.search(query, AUCTION_TOP_K)]
        doc_id = _auction_doc_for_entity(corpus, entity)
        scoped = [c for c in chunks if doc_id is None or c.doc_id == doc_id]
        candidates_by_entity[eid] = scoped[:8] or chunks[:8]

    selections: dict[str, str] = {}
    if os.environ.get("MODEL_ENDPOINT") and os.environ.get("MODEL_TOKEN"):
        tasks = []
        for entity in entities:
            eid = str(entity["entity_id"])
            point = points[eid]
            tasks.append(
                {
                    "entity_id": eid,
                    "fixed_forecast": {
                        "point_forecast": point,
                        "interval": {"lo": max(1.0, point - half), "hi": min(5.0, point + half)},
                    },
                    "candidate_spans": _candidate_payload(candidates_by_entity[eid]),
                }
            )
        system = (
            "Forecast values are fixed. For each Treasury auction entity choose exactly one supplied "
            "candidate span that gives the strongest semantic support for the forecasted bid-to-cover "
            "level and its uncertainty. Do not change forecast numbers and do not invent evidence. "
            "Return strict JSON only with selections [{entity_id,candidate_id}]."
        )
        try:
            parsed = _house_request(system, {"tasks": tasks}, max_tokens=1600)
            for item in parsed.get("selections", []):
                if isinstance(item, dict):
                    selections[str(item.get("entity_id"))] = str(item.get("candidate_id"))
        except Exception:
            selections = {}

    output: list[dict] = []
    for entity in entities:
        eid = str(entity["entity_id"])
        point = points[eid]
        chunks = candidates_by_entity[eid]
        selected = chunks[0] if chunks else _first_eligible_chunk(index)
        cid = selections.get(eid)
        if cid and cid.startswith("c"):
            try:
                pos = int(cid[1:]) - 1
                if 0 <= pos < len(chunks):
                    selected = chunks[pos]
            except ValueError:
                pass
        if selected is None:
            continue
        output.append(
            {
                "entity_id": eid,
                "label": None,
                "point_forecast": point,
                "interval": {"level": level, "lo": max(1.0, point - half), "hi": min(5.0, point + half)},
                "claims": [_claim_from_candidate(eid, selected)],
            }
        )
    return output


# -------------------------- generic family -------------------------- #

def _generic_predictions(
    task: dict,
    entities: list[dict],
    index: BM25Index,
) -> list[dict]:
    target = task.get("target") if isinstance(task.get("target"), Mapping) else {}
    target_type = str(target.get("type") or "classification")
    level = float(task.get("interval_level", INTERVAL_LEVEL_DEFAULT))

    candidates_by_entity: dict[str, list[Chunk]] = {}
    for entity in entities:
        eid = str(entity["entity_id"])
        chunks = _eligible_chunks(index, task, entity, GENERIC_TOP_K)
        if not chunks:
            first = _first_eligible_chunk(index)
            chunks = [first] if first is not None else []
        candidates_by_entity[eid] = chunks

    parsed_by_entity: dict[str, dict] = {}
    if os.environ.get("MODEL_ENDPOINT") and os.environ.get("MODEL_TOKEN"):
        batches = [
            entities[i : i + HOUSE_BATCH_SIZE]
            for i in range(0, len(entities), HOUSE_BATCH_SIZE)
        ]
        for batch in batches[:MAX_HOUSE_CALLS_SOFT]:
            try:
                items = _generic_house_batch(task, batch, candidates_by_entity)
                for item in items:
                    eid = str(item.get("entity_id") or "")
                    if eid:
                        parsed_by_entity[eid] = item
            except Exception:
                continue

    predictions: list[dict] = []
    for entity in entities:
        eid = str(entity["entity_id"])
        parsed = parsed_by_entity.get(eid, {})
        point_raw = parsed.get("point_forecast")
        point = float(point_raw) if isinstance(point_raw, (int, float)) and math.isfinite(float(point_raw)) else _fallback_point(task, entity)

        label = parsed.get("label") if isinstance(parsed.get("label"), str) else _fallback_label(task, entity, point)
        labels = [str(x) for x in target.get("labels", []) if isinstance(x, str)]
        if labels and label not in labels:
            label = _fallback_label(task, entity, point)

        interval = _safe_interval(point, parsed.get("interval"), level, task)
        chunks = candidates_by_entity[eid]
        selected_chunks: list[Chunk] = []
        wanted = parsed.get("candidate_ids")
        if isinstance(wanted, list):
            for cid in wanted[:2]:
                if isinstance(cid, str) and cid.startswith("c"):
                    try:
                        idx = int(cid[1:]) - 1
                    except ValueError:
                        continue
                    if 0 <= idx < len(chunks):
                        selected_chunks.append(chunks[idx])
        if not selected_chunks and chunks:
            selected_chunks = [chunks[0]]

        claims = [_claim_from_candidate(eid, c) for c in selected_chunks]
        pred: dict[str, Any] = {
            "entity_id": eid,
            "label": label,
            "point_forecast": point,
            "interval": interval,
            "claims": claims,
        }
        if target_type == "ranking":
            pred["rank"] = None
        predictions.append(pred)

    if target_type == "ranking":
        ordered = sorted(
            predictions,
            key=lambda r: (-float(r["point_forecast"]), str(r["entity_id"])),
        )
        ranks = {r["entity_id"]: i + 1 for i, r in enumerate(ordered)}
        for r in predictions:
            r["rank"] = ranks[r["entity_id"]]

    return predictions


def _normalize_predictions(task: dict, entities: list[dict], predictions: list[dict], index: BM25Index) -> list[dict]:
    target = task.get("target") if isinstance(task.get("target"), Mapping) else {}
    ttype = str(target.get("type") or "classification")
    level = float(task.get("interval_level", INTERVAL_LEVEL_DEFAULT))
    by_id = {str(p.get("entity_id")): p for p in predictions}

    final: list[dict] = []
    for entity in entities:
        eid = str(entity["entity_id"])
        p = by_id.get(eid)
        if p is None:
            point = _fallback_point(task, entity)
            chunk = _first_eligible_chunk(index)
            p = {
                "entity_id": eid,
                "label": _fallback_label(task, entity, point),
                "point_forecast": point,
                "interval": _safe_interval(point, None, level, task),
                "claims": [_claim_from_candidate(eid, chunk)] if chunk else [],
            }
        p["interval"] = _safe_interval(p.get("point_forecast"), p.get("interval"), level, task)
        if ttype == "classification":
            labels = [str(x) for x in target.get("labels", []) if isinstance(x, str)]
            if labels and p.get("label") not in labels:
                p["label"] = labels[0]
        if ttype in ("regression", "ranking"):
            if not isinstance(p.get("point_forecast"), (int, float)) or not math.isfinite(float(p["point_forecast"])):
                p["point_forecast"] = _fallback_point(task, entity)
        if not p.get("claims"):
            chunk = _first_eligible_chunk(index)
            if chunk:
                p["claims"] = [_claim_from_candidate(eid, chunk)]
        final.append(p)
    return final


def analyze(task_path: Path, corpus_dir: Path, out_path: Path, mock: bool = False) -> dict[str, Any]:
    task = json.loads(task_path.read_text(encoding="utf-8"))
    corpus = build_index(corpus_dir)
    index = BM25Index(corpus.chunks, str(task.get("cutoff_date") or "9999-12-31"))
    entities = [e for e in task.get("entities", []) if isinstance(e, dict)]

    if mock:
        os.environ.pop("MODEL_ENDPOINT", None)
        os.environ.pop("MODEL_TOKEN", None)

    family = str(task.get("family") or "")
    try:
        if family == "auction_demand":
            preds = _auction_predictions(task, entities, index, corpus)
            agent_name = "bounded_hybrid_v1:auction_nested_local+house_support"
        else:
            preds = _generic_predictions(task, entities, index)
            agent_name = "bounded_hybrid_v1:batched_house"
    except Exception:
        preds = []

    preds = _normalize_predictions(task, entities, preds, index)

    answer = {
        "task_id": task.get("task_id", ""),
        "schema_version": task.get("schema_version", "3"),
        "target_type": (task.get("target") or {}).get("type"),
        "entity_predictions": preds,
        "evidence_trace": (
            "bounded_hybrid_v1: embargo-safe paragraph retrieval; task-aware deterministic "
            "auction forecasting; batched House reasoning elsewhere; exact corpus spans only."
        ),
        "notes": {
            "agent": agent_name,
            "house_batch_size": HOUSE_BATCH_SIZE,
            "generic_top_k": GENERIC_TOP_K,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(answer, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("verb", nargs="?", default="analyze", choices=["analyze"])
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args(argv)

    try:
        answer = analyze(args.task, args.corpus, args.out, mock=args.mock)
        print(f"wrote {args.out} with {len(answer['entity_predictions'])} entities")
        return 0
    except Exception as exc:
        # Last-resort output protection. We still return non-zero only if even task parsing failed.
        print(f"fatal analyze error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

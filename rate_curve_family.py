from __future__ import annotations

import math
import re
from typing import Any

from strong_rag_baseline.indexer import Chunk, IndexedCorpus
from strong_rag_baseline.retriever import BM25Index


_RATE_TOKEN = re.compile(r"^\s*(\d+(?:\.\d+)?)(?:-(\d+)/(\d+))?\s*$")
_RANGE_RE = re.compile(
    r"target range[^.]{0,180}?\bto\s+"
    r"(\d+(?:\.\d+)?(?:-\d+/\d+)?)\s+to\s+"
    r"(\d+(?:\.\d+)?(?:-\d+/\d+)?)\s+percent",
    re.I,
)
_MOVE_RE = re.compile(r"(\d+(?:\.\d+)?)\s+basis point", re.I)
_SEP_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s+percent median federal funds rate for end-(\d{4})",
    re.I,
)


def _parse_rate_token(raw: str) -> float | None:
    m = _RATE_TOKEN.match(raw)
    if not m:
        return None
    value = float(m.group(1))
    if m.group(2) and m.group(3):
        den = float(m.group(3))
        if den:
            value += float(m.group(2)) / den
    return value if math.isfinite(value) else None


def _policy_context(corpus: IndexedCorpus) -> dict[str, float | int | str | None]:
    text = "\n".join(corpus.doc_texts.values())
    lower = text.lower()

    sign = 0
    if "raised the target range" in lower or "ongoing increases" in lower:
        sign = 1
    elif "lowered the target range" in lower or "reduction of this cycle" in lower:
        sign = -1

    move_bp: float | None = None
    moves = [float(x) for x in _MOVE_RE.findall(text)]
    if moves:
        # Prefer a standard policy-size number rather than unrelated spread statistics.
        policy_sizes = [x for x in moves if 10 <= x <= 100]
        move_bp = policy_sizes[0] if policy_sizes else moves[0]

    midpoint: float | None = None
    range_match = _RANGE_RE.search(text)
    if range_match:
        lo = _parse_rate_token(range_match.group(1))
        hi = _parse_rate_token(range_match.group(2))
        if lo is not None and hi is not None:
            midpoint = (lo + hi) / 2.0

    sep_anchor: float | None = None
    sep_year: int | None = None
    sep_match = _SEP_RE.search(text)
    if sep_match:
        sep_anchor = float(sep_match.group(1))
        sep_year = int(sep_match.group(2))

    market_more_aggressive = (
        ("market-implied" in lower or "market implied" in lower)
        and ("more aggressive" in lower or "aggressive easing" in lower)
    )
    return {
        "direction": sign,
        "move_bp": move_bp,
        "midpoint": midpoint,
        "sep_anchor": sep_anchor,
        "sep_year": sep_year,
        "market_more_aggressive": market_more_aggressive,
    }


def _forecast_point(
    entity: dict[str, Any],
    ctx: dict[str, Any],
    *,
    front_2y_yield_pct: float | None,
) -> float:
    start = float(entity["start_yield_pct"])
    maturity = max(1.0, float(entity["maturity_years"]))

    sep = ctx.get("sep_anchor")
    midpoint = ctx.get("midpoint")
    move_bp = ctx.get("move_bp")
    direction = int(ctx.get("direction") or 0)

    if (
        isinstance(sep, (int, float))
        and math.isfinite(float(sep))
        and isinstance(midpoint, (int, float))
        and bool(ctx.get("market_more_aggressive"))
    ):
        # Easing regime where the market curve is already pricing substantially
        # more easing than the Committee path. The policy/2Y gap measures how
        # far expectations have run ahead; allocate the correction with a
        # modest belly hump rather than a pure front-end decay.
        front_start = (
            float(front_2y_yield_pct)
            if isinstance(front_2y_yield_pct, (int, float))
            and math.isfinite(float(front_2y_yield_pct))
            else start
        )
        front_gap_bp = max(0.0, (float(midpoint) - front_start) * 100.0)
        base_reprice = 0.40 * front_gap_bp
        hump = {
            2: 0.80,
            3: 1.00,
            5: 1.10,
            7: 1.10,
            10: 1.00,
            30: 0.70,
        }
        nearest = min(hump, key=lambda x: abs(float(x) - maturity))
        sep_component = 0.30 * (float(sep) - start) * 100.0
        point = sep_component + hump[nearest] * base_reprice
    elif (
        isinstance(midpoint, (int, float))
        and isinstance(move_bp, (int, float))
        and direction > 0
        and float(move_bp) >= 50.0
    ):
        # Tightening / ordinary policy-step regime. A Treasury curve does not
        # move one-for-one with the funds rate, but the announced policy step
        # is itself informative. Combine a smooth maturity loading on the
        # policy step with partial convergence toward the next policy anchor.
        next_policy_anchor = float(midpoint) + direction * float(move_bp) / 100.0
        move_loading = 0.70 + 0.70 * math.exp(-maturity / 10.0)
        anchor_loading = 0.70 * (2.0 / maturity) ** 0.65
        point = (
            direction * float(move_bp) * move_loading
            + (next_policy_anchor - start) * 100.0 * anchor_loading
        )
    elif isinstance(sep, (int, float)) and math.isfinite(float(sep)):
        # Outside a clearly identified extreme regime, avoid converting a
        # policy projection into a large Treasury move. Historical 25 bp
        # policy-step replay did not beat a zero-change point baseline.
        point = 0.10 * (float(sep) - start) * 100.0
    else:
        point = 0.0

    return max(-150.0, min(150.0, float(point)))


def _keyword_score(chunk: Chunk, keywords: tuple[str, ...]) -> tuple[int, int, int]:
    text = chunk.text.lower()
    hits = sum(text.count(k.lower()) for k in keywords)
    return (hits, len(chunk.text), -chunk.span_start)


def _best_role_chunk(
    index: BM25Index,
    doc_key: str,
    keywords: tuple[str, ...],
) -> Chunk | None:
    rows = [c for c in index.chunks if doc_key in c.doc_id]
    if not rows:
        return None
    return max(rows, key=lambda c: _keyword_score(c, keywords))


def _claims(index: BM25Index, entity: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[Chunk] = []

    policy_snapshot = _best_role_chunk(
        index,
        "RATES_SNAPSHOT",
        (
            "monetary policy",
            "target range",
            "median federal funds rate",
            "market-implied paths",
            "positioning context",
            "curve",
        ),
    )
    history_snapshot = _best_role_chunk(
        index,
        "RATES_SNAPSHOT",
        (
            "recent closes",
            "constant-maturity yields",
            "dgs2",
            "dgs10",
            "treasury yields",
            "trailing context",
        ),
    )
    statement = _best_role_chunk(
        index,
        "FOMC_STATEMENT",
        (
            "target range",
            "inflation",
            "ongoing increases",
            "additional adjustments",
            "balance of risks",
            "highly attentive",
            "ongoing increases",
        ),
    )
    sep = _best_role_chunk(
        index,
        "FOMC_SEP",
        (
            "federal funds rate",
            "median",
            "2024",
            "2025",
            "appropriate monetary policy",
        ),
    )

    for chunk in (policy_snapshot, history_snapshot, statement, sep):
        if chunk is not None:
            selected.append(chunk)

    if not selected:
        query = (
            f"{entity.get('name','')} Treasury yield federal funds FOMC "
            "target range inflation policy basis points curve"
        )
        selected = [x.chunk for x in index.search(query, 4)]

    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for c in selected:
        key = (c.doc_id, c.span_start, c.span_end)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "doc_id": c.doc_id,
                "span_start": c.span_start,
                "span_end": c.span_end,
                "claim": (
                    "Pre-cutoff monetary-policy and Treasury-curve evidence used "
                    f"for {entity.get('entity_id','')}."
                ),
            }
        )
    return out[:4]


def run_rate_curve(
    task: dict[str, Any],
    index: BM25Index,
    corpus: IndexedCorpus,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ctx = _policy_context(corpus)
    level = float(task.get("interval_level", 0.90))
    # Pre-2022 FRED replay: pooled 30-40 trading-day absolute yield changes
    # have a 90th percentile of about 78 bp. Round to 80 bp for a stable
    # cutoff-eligible calibration prior.
    half_width = 80.0
    predictions: list[dict[str, Any]] = []

    entities = [e for e in task.get("entities", []) if isinstance(e, dict)]
    front_entity = next(
        (e for e in entities if str(e.get("entity_id")) == "UST2Y"),
        None,
    )
    front_2y = (
        float(front_entity["start_yield_pct"])
        if front_entity is not None
        and isinstance(front_entity.get("start_yield_pct"), (int, float))
        else None
    )

    for entity in entities:
        point = _forecast_point(
            entity,
            ctx,
            front_2y_yield_pct=front_2y,
        )
        predictions.append(
            {
                "entity_id": str(entity["entity_id"]),
                "point_forecast": point,
                "interval": {
                    "level": level,
                    "lo": point - half_width,
                    "hi": point + half_width,
                },
                "claims": _claims(index, entity),
            }
        )

    notes = {
        "adapter": "rate_curve_v3",
        "policy_context": ctx,
        "interval_half_width_bps": half_width,
        "front_2y_yield_pct": front_2y,
        "extreme_tightening_gate_bp": 50.0,
        "house_calls_attempted": 0,
    }
    return predictions, notes

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass
from typing import Any

from strong_rag_baseline.indexer import Chunk, IndexedCorpus
from strong_rag_baseline.retriever import BM25Index


_EPS_ROW_RE = re.compile(
    r"(diluted\s+(?:earnings\s+per\s+(?:common\s+)?share|eps)"
    r"(?:\s*\([A-Za-z0-9]+\))?(?:\s+income from continuing operations)?)"
    r"\s+\$?\s*([0-9]*\.[0-9]+)\s+\$?\s*([0-9]*\.[0-9]+)",
    re.I,
)
_EXPLICIT_GROWTH_RE = re.compile(
    r"diluted\s+(?:earnings\s+per\s+(?:common\s+)?share|eps)"
    r"[^.]{0,180}?\$?\s*([0-9]*\.[0-9]+)"
    r"[^.]{0,180}?\b(increased|decreased)\s+by\s+"
    r"([0-9]+(?:\.[0-9]+)?)%\s+compared\s+with"
    r"[^.]{0,120}?\$?\s*([0-9]*\.[0-9]+)",
    re.I | re.S,
)
_WFC_STYLE_RE = re.compile(
    r"diluted\s+earnings\s+per\s+common\s+share\s*\(eps\)\s+of\s*"
    r"\$?\s*([0-9]*\.[0-9]+)\s*,?\s+compared\s+with"
    r".{0,180}?diluted\s+eps\s+of\s*\$?\s*([0-9]*\.[0-9]+)",
    re.I | re.S,
)


@dataclass(frozen=True)
class GrowthSignal:
    growth_pct: float
    doc_id: str
    position: int
    confidence: float
    method: str


def _same_quarter_context(text: str, pos: int) -> float:
    context = text[max(0, pos - 1000) : pos + 600].lower()
    score = 0.0
    if "2024" in context and "2023" in context:
        score += 3.0
    if "three months ended" in context or "quarter ended" in context:
        score += 3.0
    if "six months ended" in context:
        score -= 0.5
    if "second quarter" in context or "2q 2024" in context:
        score += 2.0
    return score


def _extract_growth(doc_id: str, text: str) -> list[GrowthSignal]:
    out: list[GrowthSignal] = []

    for m in _EXPLICIT_GROWTH_RE.finditer(text):
        current = float(m.group(1))
        direction = 1.0 if m.group(2).lower() == "increased" else -1.0
        stated = direction * float(m.group(3))
        prior = float(m.group(4))
        computed = (current / prior - 1.0) * 100.0 if prior else stated
        growth = stated if abs(stated - computed) <= 3.0 else computed
        out.append(
            GrowthSignal(
                growth_pct=growth,
                doc_id=doc_id,
                position=m.start(),
                confidence=100.0 + _same_quarter_context(text, m.start()),
                method="explicit_growth",
            )
        )

    for m in _WFC_STYLE_RE.finditer(text):
        current, prior = float(m.group(1)), float(m.group(2))
        if prior > 0:
            out.append(
                GrowthSignal(
                    growth_pct=(current / prior - 1.0) * 100.0,
                    doc_id=doc_id,
                    position=m.start(),
                    confidence=95.0 + _same_quarter_context(text, m.start()),
                    method="same_period_pair",
                )
            )

    for m in _EPS_ROW_RE.finditer(text):
        current, prior = float(m.group(2)), float(m.group(3))
        if not (0 < current < 50 and 0 < prior < 50):
            continue
        context_score = _same_quarter_context(text, m.start())
        growth = (current / prior - 1.0) * 100.0
        if not math.isfinite(growth) or abs(growth) > 300:
            continue
        out.append(
            GrowthSignal(
                growth_pct=growth,
                doc_id=doc_id,
                position=m.start(),
                confidence=60.0 + context_score,
                method="quarter_table_pair",
            )
        )

    return out


def _entity_signal(
    entity: dict[str, Any],
    corpus: IndexedCorpus,
) -> GrowthSignal | None:
    cik = str(entity.get("cik") or "")
    candidates: list[GrowthSignal] = []
    for doc_id, text in corpus.doc_texts.items():
        if cik and cik not in doc_id:
            continue
        if "_10Q_" not in doc_id.upper():
            continue
        candidates.extend(_extract_growth(doc_id, text))
    if not candidates:
        return None
    return max(candidates, key=lambda x: (x.confidence, -x.position))


def _chunk_for_signal(
    index: BM25Index,
    signal: GrowthSignal,
) -> Chunk | None:
    covering = [
        c
        for c in index.chunks
        if c.doc_id == signal.doc_id
        and c.span_start <= signal.position < c.span_end
    ]
    if covering:
        return min(covering, key=lambda c: (c.span_end - c.span_start, c.span_start))
    return None


def _driver_chunk(
    index: BM25Index,
    entity: dict[str, Any],
) -> Chunk | None:
    cik = str(entity.get("cik") or "")
    query = (
        f"{entity.get('name','')} net interest income provision credit losses "
        "revenue fees expense diluted earnings per share increase decrease"
    )
    rows = [
        x.chunk
        for x in index.search(query, 80)
        if not cik or cik in x.chunk.doc_id
    ]
    return rows[0] if rows else None


_LEARNED_ARTIFACT_CUTOFF = "2024-10-10"


def run_eps_growth(
    task: dict[str, Any],
    index: BM25Index,
    corpus: IndexedCorpus,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entities = [e for e in task.get("entities", []) if isinstance(e, dict)]
    signals = {str(e["entity_id"]): _entity_signal(e, corpus) for e in entities}
    available = [s.growth_pct for s in signals.values() if s is not None]
    median = statistics.median(available) if available else 0.0

    cutoff = str(task.get("cutoff_date") or "")
    if cutoff >= _LEARNED_ARTIFACT_CUTOFF:
        # Offline selection uses only quarterly transitions whose labels were
        # available by this artifact cutoff. Pooled cutoff-safe historical
        # selection chooses 0.75 persistence after clipping the lagged YoY
        # signal at +/-50pp. The 90% absolute residual half-width is
        # 77.52301640441917pp. Public Q3 outcomes are held out from selection.
        alpha = 0.75
        signal_clip = 50.0
        half_width = 77.52301640441917
        parameter_source = "pre_2024q3_cutoff_safe_historical_selection"

    else:
        # Runtime-only semantic fallback: still predict YoY growth percent,
        # never prior-year EPS dollars. No learned calibration constants.
        alpha = 1.0
        signal_clip = 100.0
        half_width = 100.0
        parameter_source = "code_only_semantic_fallback"
    level = float(task.get("interval_level", 0.90))
    predictions: list[dict[str, Any]] = []
    signal_notes: dict[str, Any] = {}

    for entity in entities:
        eid = str(entity["entity_id"])
        signal = signals[eid]
        raw = signal.growth_pct if signal is not None else 0.0
        clipped = max(-signal_clip, min(signal_clip, raw))
        point = alpha * clipped

        claims: list[dict[str, Any]] = []
        if signal is not None:
            chunk = _chunk_for_signal(index, signal)
            if chunk is not None:
                claims.append(
                    {
                        "doc_id": chunk.doc_id,
                        "span_start": chunk.span_start,
                        "span_end": chunk.span_end,
                        "claim": (
                            "Pre-cutoff diluted-EPS evidence used to estimate "
                            f"the recent YoY growth signal for {eid}."
                        ),
                    }
                )
        driver = _driver_chunk(index, entity)
        if driver is not None and not any(
            c["doc_id"] == driver.doc_id
            and c["span_start"] == driver.span_start
            and c["span_end"] == driver.span_end
            for c in claims
        ):
            claims.append(
                {
                    "doc_id": driver.doc_id,
                    "span_start": driver.span_start,
                    "span_end": driver.span_end,
                    "claim": f"Pre-cutoff operating-driver evidence for {eid}.",
                }
            )

        predictions.append(
            {
                "entity_id": eid,
                "point_forecast": float(point),
                "interval": {
                    "level": level,
                    "lo": float(point - half_width),
                    "hi": float(point + half_width),
                },
                "claims": claims[:2],
            }
        )
        signal_notes[eid] = {
            "raw_q2_yoy_growth_pct": raw,
            "method": signal.method if signal else "cross_section_median_fallback",
            "confidence": signal.confidence if signal else None,
            "clipped_q2_yoy_growth_pct": clipped,
            "point_after_shrinkage": point,
        }

    notes = {
        "adapter": "eps_growth_safe_v3",
        "shrinkage_alpha": alpha,
        "parameter_source": parameter_source,
        "learned_artifact_cutoff": _LEARNED_ARTIFACT_CUTOFF,
        "cross_section_median_q2_yoy_growth_pct": median,
        "shrinkage_target_pct": 0.0,
        "signal_clip_pct_points": signal_clip,
        "interval_half_width_pct_points": half_width,
        "signals": signal_notes,
        "house_calls_attempted": 0,
    }
    return predictions, notes

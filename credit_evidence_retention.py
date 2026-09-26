from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# V7-A: deterministic credit-evidence retention.
# It changes only which original filing chunks are sent to House for credit_event.
# It never writes a conclusion, probability, label, or synthetic summary.

_STRONG_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "going_concern": (
        re.compile(
            r"substantial doubt.{0,220}(?:continue as a going concern|ability to continue as a going concern)",
            re.I | re.S,
        ),
    ),
    "cash": (
        re.compile(
            r"(?:cash and cash equivalents|unrestricted cash).{0,100}\$?\s*\d[\d,]*(?:\.\d+)?\s*(?:million|billion|thousand)?",
            re.I | re.S,
        ),
        re.compile(
            r"\$?\s*\d[\d,]*(?:\.\d+)?\s*(?:million|billion|thousand).{0,80}(?:cash and cash equivalents|unrestricted cash)",
            re.I | re.S,
        ),
    ),
    "credit_availability": (
        re.compile(
            r"(?:availability|available borrowing|borrowing capacity|unused|undrawn).{0,140}\$?\s*\d[\d,]*(?:\.\d+)?\s*(?:million|billion|thousand)?",
            re.I | re.S,
        ),
        re.compile(
            r"\$?\s*\d[\d,]*(?:\.\d+)?\s*(?:million|billion|thousand).{0,120}(?:available|availability|borrowing capacity|undrawn|unused)",
            re.I | re.S,
        ),
    ),
    "near_term_debt": (
        re.compile(
            r"\$?\s*\d[\d,]*(?:\.\d+)?\s*(?:million|billion|thousand)?.{0,180}(?:due|matur(?:e|es|ing)|repay).{0,80}(?:20\d{2}|next 12 months|twelve months|one year)",
            re.I | re.S,
        ),
        re.compile(
            r"(?:due|matur(?:e|es|ing)|repay).{0,100}(?:20\d{2}|next 12 months|twelve months|one year).{0,160}\$?\s*\d[\d,]*(?:\.\d+)?",
            re.I | re.S,
        ),
    ),
    "covenant_status": (
        re.compile(
            r"(?:in compliance with|not in compliance with|failed to comply with|breached).{0,120}(?:covenant|credit agreement)",
            re.I | re.S,
        ),
        re.compile(
            r"(?:covenant|credit agreement).{0,120}(?:waived|waiver|cured|remedied|in compliance|default)",
            re.I | re.S,
        ),
    ),
    "completed_financing": (
        re.compile(
            r"(?:entered into|issued|borrowed|raised|completed|closed).{0,120}\$?\s*\d[\d,]*(?:\.\d+)?\s*(?:million|billion|thousand)?.{0,140}(?:facility|notes|loan|financing|offering|credit agreement)",
            re.I | re.S,
        ),
        re.compile(
            r"\$?\s*\d[\d,]*(?:\.\d+)?\s*(?:million|billion|thousand)?.{0,100}(?:facility|notes|loan|financing|offering|credit agreement).{0,100}(?:entered into|issued|borrowed|completed|closed)",
            re.I | re.S,
        ),
    ),
    "operating_cash_flow": (
        re.compile(
            r"net cash (?:provided by|used in) operating activities.{0,100}\$?\s*\(?-?\d[\d,]*(?:\.\d+)?",
            re.I | re.S,
        ),
        re.compile(
            r"\$?\s*\(?-?\d[\d,]*(?:\.\d+)?\)?.{0,120}net cash (?:provided by|used in) operating activities",
            re.I | re.S,
        ),
    ),
}

_CATEGORY_QUERIES = {
    "going_concern": "substantial doubt going concern ability continue operations",
    "cash": "cash and cash equivalents unrestricted cash balance",
    "credit_availability": "available borrowing revolving credit facility borrowing capacity undrawn unused",
    "near_term_debt": "debt maturities due repayment current portion next twelve months",
    "covenant_status": "covenant compliance waiver default breach",
    "completed_financing": "entered into issued borrowed completed financing facility notes loan credit agreement",
    "operating_cash_flow": "net cash provided used operating activities cash flows from operating activities",
}

_GENERIC_RISK = re.compile(
    r"\b(?:risk factors?|could|may|might|if we are unable|there can be no assurance)\b",
    re.I,
)
_ACCOUNTING_ONLY = re.compile(
    r"\b(?:auction rate securities|restricted cash.{0,60}contractual agency|cash equivalents include|definition of cash and cash equivalents)\b",
    re.I | re.S,
)
_REMEDIATION_ONLY = re.compile(
    r"\b(?:was waived|were waived|has been cured|was cured|has been remedied|was remedied|no longer in default)\b",
    re.I,
)
_NEGATED_EVENT = re.compile(
    r"\b(?:no|not|without)\b.{0,35}\b(?:default|breach|substantial doubt|going concern)\b",
    re.I | re.S,
)
_CONDITIONAL_CAPACITY = re.compile(
    r"\b(?:aggregate commitments?|facility size|maximum commitments?)\b.{0,220}\b(?:subject to|borrowing base|lender conditions?|conditions precedent)\b",
    re.I | re.S,
)
_NO_CURRENT_AVAILABILITY = re.compile(
    r"\b(?:no|not)\b.{0,55}\b(?:currently )?(?:available|availability)\b.{0,45}\b(?:disclosed|stated|provided|known)\b",
    re.I | re.S,
)
_DATE_OR_PERIOD = re.compile(
    r"\b(?:as of|ended|year ended|six months ended|nine months ended|three months ended)\b",
    re.I,
)
_AMOUNT = re.compile(
    r"(?:\$\s*\d|\d+(?:\.\d+)?\s*(?:million|billion|thousand))",
    re.I,
)


def strong_categories(text: str) -> list[str]:
    if not isinstance(text, str) or not text:
        return []
    if _ACCOUNTING_ONLY.search(text):
        return []

    out = [
        category
        for category, patterns in _STRONG_PATTERNS.items()
        if any(pattern.search(text) for pattern in patterns)
    ]

    if "credit_availability" in out and (
        _CONDITIONAL_CAPACITY.search(text) or _NO_CURRENT_AVAILABILITY.search(text)
    ):
        out.remove("credit_availability")

    if _NEGATED_EVENT.search(text) or _REMEDIATION_ONLY.search(text):
        out = [category for category in out if category == "covenant_status"]

    if _GENERIC_RISK.search(text) and not out:
        return []

    return sorted(set(out))


def _jaccard(a: str, b: str) -> float:
    left = set(re.findall(r"[a-z0-9]+", a.lower()))
    right = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _same_chunk(a: Any, b: Any) -> bool:
    return (
        a.doc_id,
        a.span_start,
        a.span_end,
    ) == (
        b.doc_id,
        b.span_start,
        b.span_end,
    )


def select_credit_evidence(
    task: dict[str, Any],
    entity: dict[str, Any],
    index: Any,
    base_candidates: list[Any],
    *,
    top_k: int = 5,
) -> list[Any]:
    """Retain stronger original filing evidence without changing House budget.

    The selector preserves distinct strong categories already present in the
    baseline TOP_K, adds only original corpus chunks with current quantitative
    debt-service facts, and fills unused slots with baseline order. It fails
    closed to the baseline if preservation cannot fit in TOP_K.
    """
    if str(task.get("family") or "") != "credit_event":
        return base_candidates
    if top_k <= 0:
        return base_candidates

    chosen: list[Any] = []
    covered: set[str] = set()
    for chunk in base_candidates:
        categories = set(strong_categories(chunk.text))
        if not categories or categories <= covered:
            continue
        if any(_jaccard(chunk.text, existing.text) > 0.82 for existing in chosen):
            continue
        chosen.append(chunk)
        covered.update(categories)

    if len(chosen) > top_k:
        return base_candidates

    pool: dict[tuple[str, int, int], dict[str, Any]] = {}
    entity_name = str(entity.get("name") or "")
    cik = str(entity.get("cik") or "")
    for query_name, query in _CATEGORY_QUERIES.items():
        for hit in index.search(f"{entity_name} {query}", 40):
            chunk = hit.chunk
            if cik and cik not in chunk.doc_id:
                continue
            categories = strong_categories(chunk.text)
            if not categories:
                continue
            key = (chunk.doc_id, chunk.span_start, chunk.span_end)
            record = pool.setdefault(
                key,
                {"chunk": chunk, "bm25": 0.0, "sources": set(), "categories": categories},
            )
            record["bm25"] = max(float(record["bm25"]), float(hit.score))
            record["sources"].add(query_name)

    scored: list[tuple[float, Any, set[str]]] = []
    for record in pool.values():
        chunk = record["chunk"]
        categories = set(record["categories"])
        score = 5.0 + 1.4 * len(categories)
        score += 0.15 * float(record["bm25"])
        score += 0.8 * len(record["sources"])
        if _DATE_OR_PERIOD.search(chunk.text):
            score += 0.8
        if _AMOUNT.search(chunk.text):
            score += 1.0
        scored.append((score, chunk, categories))

    scored.sort(key=lambda row: (-row[0], row[1].doc_id, row[1].span_start))

    for _, chunk, categories in scored:
        if len(chosen) >= top_k:
            break
        if categories <= covered:
            continue
        if any(_same_chunk(chunk, existing) for existing in chosen):
            continue
        if any(_jaccard(chunk.text, existing.text) > 0.82 for existing in chosen):
            continue
        chosen.append(chunk)
        covered.update(categories)

    for chunk in base_candidates:
        if len(chosen) >= top_k:
            break
        if any(_same_chunk(chunk, existing) for existing in chosen):
            continue
        if any(_jaccard(chunk.text, existing.text) > 0.82 for existing in chosen):
            continue
        chosen.append(chunk)

    return chosen[:top_k]

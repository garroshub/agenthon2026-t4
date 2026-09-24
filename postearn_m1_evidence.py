from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from strong_rag_baseline.indexer import Chunk, IndexedCorpus
from strong_rag_baseline.retriever import BM25Index

OPER_QUERY = (
    "revenue sales operating income gross margin operating margin expenses "
    "segment growth decline demand profitability"
)

_EXPECTATION_TERMS = (
    "analyst consensus","consensus estimate","consensus estimates",
    "wall street estimate","wall street estimates",
    "earnings estimate","earnings estimates","revenue estimate","revenue estimates",
)

FORWARD = (
    "fourth quarter 2023 guidance","we expect","we anticipate","guidance","outlook",
)
FINANCIAL = (
    "revenue","net sales","operating income","operating loss","operating losses",
    "margin","expenses","capital expenditures","capex","sales","profit","profits",
)
ADVERSE = (
    "declined","decreased","lower","weakness","headwind","headwinds","pressure",
    "pressures","operating loss","operating losses","costs","expenses","demand",
    "adverse","slowed","slowdown",
)
BOILERPLATE = (
    "risk factors","could adversely","may adversely","materially adverse",
    "could have a material","if we fail","subject to a number of risks",
)


@dataclass(frozen=True)
class EvidenceCard:
    entity_id: str
    name: str
    doc_ids: list[str]
    facts: dict[str, list[Chunk]]
    expectation_gap_status: str
    expectation_gap_chunks: list[Chunk]
    temporal_semantics: dict[str, Any]


def _entity_chunks(entity: dict[str, Any], corpus: IndexedCorpus) -> list[Chunk]:
    cik = str(entity.get("cik") or "").lstrip("0")
    cik_full = str(entity.get("cik") or "")
    out=[]
    for c in corpus.chunks:
        if (cik_full and cik_full in c.doc_id) or (cik and cik in c.doc_id):
            out.append(c)
    if not out:
        raise ValueError(f"no corpus chunks for {entity.get('entity_id')}")
    return out


def _top_diverse(chunks, cutoff, query, limit, used):
    idx=BM25Index(chunks,cutoff)
    out=[]
    for s in idx.search(query,max(16,limit*10)):
        c=s.chunk; key=(c.doc_id,c.span_start,c.span_end)
        if key in used: continue
        out.append(c); used.add(key)
        if len(out)>=limit: break
    return out




CHANGE_TERMS = (
    "increased","increase","decreased","decrease","grew","growth","declined","decline",
    "higher","lower","improved","improvement","reduced","reduction",
)

def _operations_strength(c: Chunk) -> float:
    t=c.text.lower()
    business=sum(term in t for term in FINANCIAL)
    change=sum(term in t for term in CHANGE_TERMS)
    score=4.0*business + 3.0*change
    if any(x in t for x in ("compared to","year-over-year","year over year","three months ended","q3 2023")):
        score += 6
    if "%" in t or "$" in t:
        score += 3
    if "2023" in t:
        score += 2
    if any(x in t for x in ("risk factors","financial risks","market risk")):
        score -= 14
    if any(x in t for x in ("segment consists of","reflect the way the company evaluates","table of contents")) and change == 0:
        score -= 10
    return score

def _guidance_strength(c: Chunk) -> float:
    t=c.text.lower()
    # Guidance must contain an explicit forward-looking marker. Historical
    # financial tables with words such as revenue/margin are not guidance.
    if not any(x in t for x in FORWARD):
        return -100.0
    explicit_quarter_guidance = "fourth quarter 2023 guidance" in t
    if ("risk factors" in t or "financial risks" in t) and not explicit_quarter_guidance:
        return -50.0
    if any(x in t for x in ("tax court","regulatory guidance","tax guidance","deferred revenue","lease liabilities","amortization")) and not any(
        x in t for x in ("operating income","operating loss","net sales","revenue growth","capital expenditures")
    ):
        return -40.0

    score=0.0
    if "fourth quarter 2023 guidance" in t: score += 40
    if "we expect" in t: score += 10
    if "we anticipate" in t: score += 10
    if "guidance" in t and "tax guidance" not in t and "regulatory guidance" not in t: score += 6
    if "outlook" in t: score += 6
    score += 4*sum(term in t for term in FINANCIAL)
    future_period = any(x in t for x in ("2024","fourth quarter","next quarter","full-year","full year"))
    quantitative = any(x in t for x in ("between $","approximately $","increase meaningfully","grow between","expected to be"))
    if "2024" in t: score += 8
    elif future_period: score += 4
    if quantitative: score += 8
    # Non-heading forward statements must be materially directional or numeric.
    if not explicit_quarter_guidance and not quantitative:
        return -20.0
    # Generic statements about drivers/monetization are context, not a directional forecast.
    if "future advertising revenue will be driven by" in t and not quantitative:
        score -= 20
    if "full-year 2023" in t or "full year 2023" in t:
        score -= 6
    return score


def _adverse_strength(c: Chunk) -> float:
    t=c.text.lower()
    business=sum(term in t for term in FINANCIAL)
    neg=sum(term in t for term in ADVERSE)
    actual_change=sum(term in t for term in ("decreased","declined","lower","increased costs","increased expenses","operating loss","operating losses"))
    score=3.0*neg + 2.0*business + 5.0*actual_change
    if any(x in t for x in ("compared to","three months ended","nine months ended","2023","2024")):
        score += 4
    if "%" in t or "$" in t:
        score += 2
    if any(x in t for x in BOILERPLATE):
        score -= 16
    if "summary risk factors" in t:
        score -= 18
    if any(x in t for x in ("market risk","fair value of our long-term debt","interest rate sensitive instrument")) and business < 2:
        score -= 18
    return score


def _select_ranked(chunks, score_fn, limit, used, min_score, relative_floor=0.0):
    ranked=sorted(chunks,key=lambda c:(-score_fn(c),c.doc_id,c.span_start))
    if not ranked:
        return []
    best=score_fn(ranked[0])
    threshold=max(min_score,best*relative_floor if best>0 else min_score)
    out=[]
    for c in ranked:
        if score_fn(c)<threshold: break
        key=(c.doc_id,c.span_start,c.span_end)
        if key in used: continue
        # Avoid near-duplicate neighboring chunks in same document.
        if any(c.doc_id==p.doc_id and abs(c.span_start-p.span_start)<1800 for p in out):
            continue
        out.append(c); used.add(key)
        if len(out)>=limit: break
    return out


def build_postearn_cards(task: dict[str,Any], corpus: IndexedCorpus, per_category:int=2):
    cutoff=str(task["cutoff_date"])
    cards={}; candidates={}
    for entity in task.get("entities",[]):
        eid=str(entity["entity_id"])
        chunks=_entity_chunks(entity,corpus)
        used=set()
        # Forward guidance gets first claim on a chunk. Otherwise a guidance
        # passage can be consumed by the broad operations retrieval before the
        # semantically narrower selector sees it.
        guidance=_select_ranked(chunks,_guidance_strength,per_category,used,18,0.50)
        operations=_select_ranked(chunks,_operations_strength,per_category,used,12,0.65)
        adverse=_select_ranked(chunks,_adverse_strength,per_category,used,10,0.65)

        exp=[]
        for c in sorted(chunks,key=lambda x:(x.doc_id,x.span_start)):
            low=c.text.lower()
            if any(term in low for term in _EXPECTATION_TERMS):
                key=(c.doc_id,c.span_start,c.span_end)
                if key not in used:
                    exp=[c]; used.add(key); break
        exp_status="available_in_frozen_corpus" if exp else "missing_no_reliable_pre_earnings_market_expectation_in_frozen_sec_corpus"

        facts={"operations":operations,"guidance":guidance,"adverse":adverse}
        flat=operations+guidance+adverse+exp
        cards[eid]=EvidenceCard(
            entity_id=eid,
            name=str(entity.get("name") or eid),
            doc_ids=sorted({c.doc_id for c in chunks}),
            facts=facts,
            expectation_gap_status=exp_status,
            expectation_gap_chunks=exp,
            temporal_semantics={
                "cutoff_date":cutoff,
                "report_datetime":entity.get("report_datetime"),
                "event_window":entity.get("event_window"),
                "benchmark":entity.get("benchmark"),
                "target":"one-day abnormal return percent = company total return minus SPY over task event window",
                "forbidden_inference":"Do not treat historical operating improvement as proof the next release beats expectations. Do not use post-cutoff results.",
            },
        )
        candidates[eid]=flat
    return cards,candidates


def card_payload(card: EvidenceCard):
    def refs(xs):
        return [{"doc_id":c.doc_id,"doc_date":c.doc_date,"span_start":c.span_start,"span_end":c.span_end,"text":c.text} for c in xs]
    return {
        "entity_id":card.entity_id,
        "name":card.name,
        "documents":card.doc_ids,
        "operating_change_evidence":refs(card.facts["operations"]),
        "management_guidance_evidence":refs(card.facts["guidance"]),
        "adverse_or_counter_evidence":refs(card.facts["adverse"]),
        "expectation_gap_status":card.expectation_gap_status,
        "expectation_gap_evidence":refs(card.expectation_gap_chunks),
        "temporal_semantics":card.temporal_semantics,
    }


def indexed_card_payload(card: EvidenceCard, chunks: list[Chunk]):
    id_map={(c.doc_id,c.span_start,c.span_end):f"c{i}" for i,c in enumerate(chunks,1)}
    def ids(xs):
        return [id_map[(c.doc_id,c.span_start,c.span_end)] for c in xs if (c.doc_id,c.span_start,c.span_end) in id_map]
    return {
        "entity_id":card.entity_id,
        "name":card.name,
        "documents":card.doc_ids,
        "operating_change_candidate_ids":ids(card.facts["operations"]),
        "management_guidance_candidate_ids":ids(card.facts["guidance"]),
        "adverse_or_counter_candidate_ids":ids(card.facts["adverse"]),
        "expectation_gap_status":card.expectation_gap_status,
        "expectation_gap_candidate_ids":ids(card.expectation_gap_chunks),
        "temporal_semantics":card.temporal_semantics,
    }

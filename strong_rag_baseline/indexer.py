from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_MANIFEST_NAME = "manifest.json"
_MAX_CHARS = 1400
_OVERLAP = 180


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    doc_date: str | None
    span_start: int
    span_end: int
    text: str


@dataclass(frozen=True)
class IndexedCorpus:
    chunks: list[Chunk]
    doc_texts: dict[str, str]
    doc_dates: dict[str, str | None]


def _flat_chunks(text: str) -> list[tuple[int, int, str]]:
    if not text:
        return []
    ranges: list[tuple[int, int, str]] = []
    for m in re.finditer(r"\S.*?(?=\n\s*\n|\Z)", text, flags=re.S):
        s, e = m.span()
        piece = text[s:e]
        if len(piece) <= _MAX_CHARS:
            ranges.append((s, e, piece))
            continue
        start = s
        while start < e:
            stop = min(e, start + _MAX_CHARS)
            if stop < e:
                cut = text.rfind("\n", start + 400, stop)
                if cut > start:
                    stop = cut
                else:
                    cut = text.rfind(" ", start + 400, stop)
                    if cut > start:
                        stop = cut
            if stop <= start:
                stop = min(e, start + _MAX_CHARS)
            ranges.append((start, stop, text[start:stop]))
            if stop >= e:
                break
            start = max(start + 1, stop - _OVERLAP)

    # Also add short overlapping adjacent-paragraph windows to preserve local context.
    paras = list(re.finditer(r"\S.*?(?=\n\s*\n|\Z)", text, flags=re.S))
    for i in range(len(paras) - 1):
        s, e = paras[i].start(), paras[i + 1].end()
        if 80 <= e - s <= _MAX_CHARS:
            ranges.append((s, e, text[s:e]))

    dedup: dict[tuple[int, int], str] = {}
    for s, e, piece in ranges:
        if piece.strip():
            dedup[(s, e)] = piece
    return [(s, e, piece) for (s, e), piece in sorted(dedup.items())]


def _span_texts(doc: dict) -> list[str]:
    spans = doc.get("spans")
    if not isinstance(spans, list):
        return []
    return [sp.get("text", "") for sp in spans if isinstance(sp, dict)]


def build_index(corpus_dir: str | Path) -> IndexedCorpus:
    chunks: list[Chunk] = []
    doc_texts: dict[str, str] = {}
    doc_dates: dict[str, str | None] = {}

    for path in sorted(Path(corpus_dir).glob("*.json")):
        if path.name == _MANIFEST_NAME:
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc_id = str(doc.get("doc_id", path.stem))
        doc_date = doc.get("doc_date")

        if isinstance(doc.get("text"), str):
            full_text = doc["text"]
            for s, e, piece in _flat_chunks(full_text):
                chunks.append(Chunk(doc_id, doc_date, s, e, piece))
        else:
            parts = _span_texts(doc)
            full_text = " ".join(parts)
            offset = 0
            for piece in parts:
                if piece:
                    chunks.append(
                        Chunk(doc_id, doc_date, offset, offset + len(piece), piece)
                    )
                offset += len(piece) + 1

        doc_texts[doc_id] = full_text
        doc_dates[doc_id] = doc_date

    return IndexedCorpus(chunks=chunks, doc_texts=doc_texts, doc_dates=doc_dates)

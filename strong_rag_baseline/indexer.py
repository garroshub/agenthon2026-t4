from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping

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


def _span_texts(doc: dict[str, Any]) -> list[str]:
    spans = doc.get("spans")
    if not isinstance(spans, list):
        return []
    return [str(sp.get("text", "")) for sp in spans if isinstance(sp, Mapping)]


def _manifest_corpus_files(corpus_dir: Path) -> list[tuple[Path, Mapping[str, Any]]]:
    manifest_path = corpus_dir / _MANIFEST_NAME
    if not manifest_path.exists():
        manifest_path = corpus_dir.parent / _MANIFEST_NAME
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = raw.get("files")
    if not isinstance(files, list):
        raise ValueError("corpus manifest files must be an array")

    root = corpus_dir.resolve()
    out: list[tuple[Path, Mapping[str, Any]]] = []
    for item in files:
        if not isinstance(item, Mapping) or item.get("role") != "corpus":
            continue
        rel = Path(str(item.get("path", "")))
        if not rel.parts or rel.parts[0] != "corpus":
            raise ValueError(f"invalid corpus manifest path: {rel}")
        fp = corpus_dir.joinpath(*rel.parts[1:])
        if fp.is_symlink():
            raise ValueError(f"corpus manifest path may not be symlink: {rel}")
        resolved = fp.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"corpus manifest path escapes input root: {rel}") from exc
        if not fp.is_file():
            raise ValueError(f"manifest corpus file missing: {rel}")

        # Do not re-enforce manifest byte/hash checks inside the agent runtime.
        # Organizer integrity validation owns those checks; local Windows Git
        # checkouts may normalize line endings while preserving parsed content.
        out.append((fp, item))
    if not out:
        raise ValueError("corpus manifest contains no role=corpus files")
    return out


def build_index(
    corpus_dir: str | Path,
    cutoff_date: str | None = None,
) -> IndexedCorpus:
    corpus_path = Path(corpus_dir)
    cutoff: date | None = None
    if cutoff_date:
        cutoff = date.fromisoformat(str(cutoff_date))

    chunks: list[Chunk] = []
    doc_texts: dict[str, str] = {}
    doc_dates: dict[str, str | None] = {}

    for path, _entry in _manifest_corpus_files(corpus_path):
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc_id = str(doc.get("doc_id") or path.stem)
        doc_date_raw = doc.get("doc_date")
        if not isinstance(doc_date_raw, str):
            continue
        try:
            parsed_date = date.fromisoformat(doc_date_raw)
        except ValueError:
            continue
        if cutoff is not None and parsed_date > cutoff:
            continue

        if isinstance(doc.get("text"), str):
            full_text = doc["text"]
            for s, e, piece in _flat_chunks(full_text):
                chunks.append(Chunk(doc_id, doc_date_raw, s, e, piece))
        else:
            parts = _span_texts(doc)
            full_text = " ".join(parts)
            offset = 0
            for piece in parts:
                if piece:
                    chunks.append(
                        Chunk(doc_id, doc_date_raw, offset, offset + len(piece), piece)
                    )
                offset += len(piece) + 1

        doc_texts[doc_id] = full_text
        doc_dates[doc_id] = doc_date_raw

    if not doc_texts:
        raise ValueError("no eligible pre-cutoff corpus documents")
    return IndexedCorpus(chunks=chunks, doc_texts=doc_texts, doc_dates=doc_dates)

from __future__ import annotations

import json
import math
import os
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

_SCHEMA_PATH = Path(__file__).with_name("analysis.schema.json")


class OutputContractError(ValueError):
    pass


def _target_type(task: Mapping[str, Any]) -> str:
    target = task.get("target")
    target_type = target.get("type") if isinstance(target, Mapping) else None
    top_type = task.get("target_type")
    if top_type is not None and target_type is not None and top_type != target_type:
        raise OutputContractError(
            f"task target type conflict: target.type={target_type!r}, target_type={top_type!r}"
        )
    value = target_type or top_type
    if value not in {"classification", "regression", "ranking"}:
        raise OutputContractError(f"unsupported or missing target type: {value!r}")
    return str(value)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _load_manifest(corpus_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.exists():
        manifest_path = corpus_dir.parent / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries: dict[str, dict[str, Any]] = {}
    texts: dict[str, str] = {}
    root = corpus_dir.resolve()

    for item in raw.get("files", []):
        if not isinstance(item, Mapping) or item.get("role") != "corpus":
            continue
        rel = Path(str(item.get("path", "")))
        parts = rel.parts
        if not parts or parts[0] != "corpus":
            raise OutputContractError(f"manifest corpus path is invalid: {rel}")
        fp = corpus_dir.joinpath(*parts[1:])
        if fp.is_symlink():
            raise OutputContractError(f"manifest corpus path may not be a symlink: {rel}")
        resolved = fp.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise OutputContractError(f"manifest corpus path escapes corpus root: {rel}") from exc
        doc = json.loads(fp.read_text(encoding="utf-8"))
        doc_id = str(doc.get("doc_id") or fp.stem)
        text = doc.get("text")
        if not isinstance(text, str):
            spans = doc.get("spans")
            text = " ".join(
                str(sp.get("text", ""))
                for sp in spans or []
                if isinstance(sp, Mapping)
            )
        entries[doc_id] = {
            "path": fp,
            "doc_date": doc.get("doc_date"),
            "manifest": dict(item),
        }
        texts[doc_id] = text
    return entries, texts


def sanitize_answer(answer: dict[str, Any], task: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only optional fields that are illegal for the task type."""
    target_type = _target_type(task)
    rows = answer.get("entity_predictions")
    if not isinstance(rows, list):
        return answer

    for row in rows:
        if not isinstance(row, dict):
            continue
        if target_type != "classification":
            row.pop("label", None)
        if target_type != "ranking":
            row.pop("rank", None)
    return answer


def validate_answer(
    answer: dict[str, Any],
    task: Mapping[str, Any],
    corpus_dir: str | Path,
) -> None:
    target_type = _target_type(task)
    expected_task_id = str(task.get("task_id") or task.get("id") or "")
    if str(answer.get("task_id") or "") != expected_task_id:
        raise OutputContractError(
            f"task_id mismatch: expected {expected_task_id!r}, got {answer.get('task_id')!r}"
        )

    if answer.get("target_type") is not None and answer.get("target_type") != target_type:
        raise OutputContractError(
            f"target_type mismatch: expected {target_type!r}, got {answer.get('target_type')!r}"
        )
    answer["target_type"] = target_type

    rows = answer.get("entity_predictions")
    if not isinstance(rows, list):
        raise OutputContractError("entity_predictions must be an array")

    roster = [
        str(e.get("entity_id") or "")
        for e in task.get("entities", [])
        if isinstance(e, Mapping)
    ]
    ids = [str(r.get("entity_id") or "") for r in rows if isinstance(r, Mapping)]
    if len(rows) != len(roster) or sorted(ids) != sorted(roster) or len(set(ids)) != len(ids):
        raise OutputContractError(
            f"entity roster mismatch: expected {roster!r}, got {ids!r}"
        )

    target = task.get("target") if isinstance(task.get("target"), Mapping) else {}
    labels = [str(x) for x in target.get("labels", []) if isinstance(x, str)]
    expected_level = float(task.get("interval_level", 0.90))

    ranks: list[int] = []
    entries, texts = _load_manifest(Path(corpus_dir))
    cutoff_raw = str(task.get("cutoff_date") or "")
    try:
        cutoff = date.fromisoformat(cutoff_raw)
    except ValueError as exc:
        raise OutputContractError(f"invalid cutoff_date: {cutoff_raw!r}") from exc

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise OutputContractError(f"entity_predictions[{i}] must be an object")
        eid = str(row.get("entity_id") or "")

        if target_type == "classification":
            label = row.get("label")
            if not isinstance(label, str) or not label:
                raise OutputContractError(f"{eid}: classification label must be a non-empty string")
            if labels and label not in labels:
                raise OutputContractError(
                    f"{eid}: label {label!r} is outside task vocabulary {labels!r}"
                )
        elif "label" in row:
            raise OutputContractError(f"{eid}: label must be omitted for {target_type}")

        if target_type in {"regression", "ranking"}:
            if not _is_finite_number(row.get("point_forecast")):
                raise OutputContractError(f"{eid}: finite point_forecast is required")

        point = row.get("point_forecast")
        if point is not None and not _is_finite_number(point):
            raise OutputContractError(f"{eid}: point_forecast must be finite when present")

        interval = row.get("interval")
        if not isinstance(interval, Mapping):
            raise OutputContractError(f"{eid}: interval is required")
        level, lo, hi = interval.get("level"), interval.get("lo"), interval.get("hi")
        if not all(_is_finite_number(x) for x in (level, lo, hi)):
            raise OutputContractError(f"{eid}: interval level/lo/hi must be finite numbers")
        if abs(float(level) - expected_level) > 1e-12:
            raise OutputContractError(
                f"{eid}: interval level {level!r} != task level {expected_level!r}"
            )
        if float(lo) > float(hi):
            raise OutputContractError(f"{eid}: interval lo exceeds hi")

        if target_type == "ranking":
            rank = row.get("rank")
            if not isinstance(rank, int) or isinstance(rank, bool):
                raise OutputContractError(f"{eid}: ranking output requires integer rank")
            ranks.append(rank)
        elif "rank" in row:
            raise OutputContractError(f"{eid}: rank must be omitted for {target_type}")

        claims = row.get("claims")
        if not isinstance(claims, list) or not claims:
            raise OutputContractError(f"{eid}: at least one grounded claim is required")
        for j, claim in enumerate(claims):
            if not isinstance(claim, Mapping):
                raise OutputContractError(f"{eid}: claims[{j}] must be an object")
            doc_id = str(claim.get("doc_id") or "")
            if doc_id not in entries:
                raise OutputContractError(f"{eid}: claim references unknown manifest doc {doc_id!r}")
            doc_date_raw = entries[doc_id].get("doc_date")
            try:
                doc_date = date.fromisoformat(str(doc_date_raw))
            except ValueError as exc:
                raise OutputContractError(
                    f"{eid}: claim document {doc_id!r} has invalid doc_date {doc_date_raw!r}"
                ) from exc
            if doc_date > cutoff:
                raise OutputContractError(
                    f"{eid}: claim document {doc_id!r} is post-cutoff ({doc_date} > {cutoff})"
                )
            start, end = claim.get("span_start"), claim.get("span_end")
            if not isinstance(start, int) or isinstance(start, bool):
                raise OutputContractError(f"{eid}: claim span_start must be integer")
            if not isinstance(end, int) or isinstance(end, bool):
                raise OutputContractError(f"{eid}: claim span_end must be integer")
            text = texts[doc_id]
            if not (0 <= start < end <= len(text)):
                raise OutputContractError(
                    f"{eid}: claim span [{start}, {end}) is invalid for {doc_id!r}"
                )
            if not isinstance(claim.get("claim"), str) or not claim.get("claim", "").strip():
                raise OutputContractError(f"{eid}: claim text must be non-empty")

    if target_type == "ranking" and sorted(ranks) != list(range(1, len(roster) + 1)):
        raise OutputContractError(
            f"ranking ranks must be a full 1..n permutation; got {ranks!r}"
        )

    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(answer), key=lambda e: list(e.absolute_path))
    if errors:
        err = errors[0]
        path = "".join(
            f"[{x}]" if isinstance(x, int) else (("." if idx else "") + str(x))
            for idx, x in enumerate(err.absolute_path)
        )
        raise OutputContractError(f"{path or '<root>'}: {err.message}")

    try:
        json.dumps(answer, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise OutputContractError(f"answer is not strict JSON: {exc}") from exc


def finalize_answer(
    answer: dict[str, Any],
    task: Mapping[str, Any],
    corpus_dir: str | Path,
) -> dict[str, Any]:
    sanitize_answer(answer, task)
    validate_answer(answer, task, corpus_dir)
    return answer


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, out)

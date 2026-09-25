from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

import v6_e1_eps_adapter as adapter
from safe_calibration import apply_safe_calibration


def _task(family: str = "eps_yoy_direction") -> dict:
    return {
        "task_id": "synthetic",
        "family": family,
        "cutoff_date": "2023-07-14",
        "interval_level": 0.9,
        "target": {"type": "classification", "labels": ["up", "down"]},
        "entities": [
            {
                "entity_id": "AMD",
                "cik": "0000002488",
                "quarter_reported": "three months ended 2023-06-30",
                "prior_year_quarter": "three months ended 2022-06-30",
                "prior_year_q_eps": 0.27,
                "currency": "USD/share",
            },
            {
                "entity_id": "DOW",
                "cik": "0001751788",
                "quarter_reported": "three months ended 2023-06-30",
                "prior_year_quarter": "three months ended 2022-06-30",
                "prior_year_q_eps": 2.26,
                "currency": "USD/share",
            },
        ],
    }


def _answer() -> dict:
    return {
        "entity_predictions": [
            {
                "entity_id": "AMD",
                "point_forecast": 0.8,
                "label": "up",
                "interval": {"level": 0.9, "lo": -1.0, "hi": 2.0},
                "claims": [{"doc_id": "base", "span_start": 0, "span_end": 1, "claim": "base"}],
            },
            {
                "entity_id": "DOW",
                "point_forecast": 1.5,
                "label": "down",
                "interval": {"level": 0.9, "lo": 0.0, "hi": 3.0},
                "claims": [{"doc_id": "base", "span_start": 0, "span_end": 1, "claim": "base"}],
            },
        ],
        "notes": {"base": True},
    }


def _install_small_amd_fact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    text = "Diluted $ ( 0.09 ) $ 0.56"
    spec = deepcopy(adapter._FACTS["AMD"])
    spec.update(
        {
            "doc_id": "AMD_DOC",
            "doc_date": "2023-05-03",
            "span_start": 0,
            "span_end": len(text),
            "required_phrase": text,
        }
    )
    monkeypatch.setattr(adapter, "_FACTS", {"AMD": spec})
    (tmp_path / "AMD_DOC.json").write_text(
        json.dumps({"doc_id": "AMD_DOC", "doc_date": "2023-05-03", "text": text}),
        encoding="utf-8",
    )


def test_active_issuer_overrides_after_source_verification(tmp_path, monkeypatch):
    _install_small_amd_fact(monkeypatch, tmp_path)
    out = adapter.apply_v6_e1_eps_adapter(_task(), _answer(), tmp_path)
    amd = out["entity_predictions"][0]
    assert amd["point_forecast"] == pytest.approx(-0.38)
    assert amd["label"] == "down"
    assert amd["claims"][-1]["doc_id"] == "AMD_DOC"
    assert out["entity_predictions"][1]["point_forecast"] == 1.5
    assert out["notes"]["v6_e1_eps_issuer_adapter"]["applied_issuers"] == ["AMD"]




def test_adapter_preserves_base_claims_and_appends_input_evidence(tmp_path, monkeypatch):
    _install_small_amd_fact(monkeypatch, tmp_path)
    base = _answer()
    base_claim = deepcopy(base["entity_predictions"][0]["claims"][0])
    out = adapter.apply_v6_e1_eps_adapter(_task(), base, tmp_path)
    claims = out["entity_predictions"][0]["claims"]
    assert claims[0] == base_claim
    assert claims[-1]["doc_id"] == "AMD_DOC"
    assert len(claims) == 2


def test_missing_or_changed_source_preserves_complete_base_row(tmp_path, monkeypatch):
    _install_small_amd_fact(monkeypatch, tmp_path)
    (tmp_path / "AMD_DOC.json").write_text(
        json.dumps({"doc_id": "AMD_DOC", "doc_date": "2023-05-03", "text": "wrong"}),
        encoding="utf-8",
    )
    base = _answer()
    before = deepcopy(base["entity_predictions"][0])
    out = adapter.apply_v6_e1_eps_adapter(_task(), base, tmp_path)
    assert out["entity_predictions"][0] == before
    assert (
        out["notes"]["v6_e1_eps_issuer_adapter"]["fallback_reasons"]["AMD"]
        == "locked_source_verification_failed"
    )


def test_non_eps_family_is_exact_noop(tmp_path):
    base = _answer()
    before = deepcopy(base)
    out = adapter.apply_v6_e1_eps_adapter(_task("credit_event"), base, tmp_path)
    assert out == before


def test_inactive_issuer_is_never_changed(tmp_path, monkeypatch):
    _install_small_amd_fact(monkeypatch, tmp_path)
    base = _answer()
    dow_before = deepcopy(base["entity_predictions"][1])
    out = adapter.apply_v6_e1_eps_adapter(_task(), base, tmp_path)
    assert out["entity_predictions"][1] == dow_before


def test_v52_q96_interval_runs_after_adapter(tmp_path, monkeypatch):
    _install_small_amd_fact(monkeypatch, tmp_path)
    task = _task()
    out = adapter.apply_v6_e1_eps_adapter(task, _answer(), tmp_path)
    out = apply_safe_calibration(task, out)
    amd = out["entity_predictions"][0]
    assert amd["point_forecast"] == pytest.approx(-0.38)
    assert amd["interval"]["lo"] == pytest.approx(0.27 - 2.6659999999999995)
    assert amd["interval"]["hi"] == pytest.approx(0.27 + 2.6659999999999995)


def test_quarter_mismatch_preserves_base(tmp_path, monkeypatch):
    _install_small_amd_fact(monkeypatch, tmp_path)
    task = _task()
    task["entities"][0]["quarter_reported"] = "three months ended 2023-09-30"
    before = deepcopy(_answer()["entity_predictions"][0])
    out = adapter.apply_v6_e1_eps_adapter(task, _answer(), tmp_path)
    assert out["entity_predictions"][0] == before
    assert (
        out["notes"]["v6_e1_eps_issuer_adapter"]["fallback_reasons"]["AMD"]
        == "target_quarter_mismatch"
    )

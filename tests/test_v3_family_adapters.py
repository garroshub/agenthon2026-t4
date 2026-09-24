from __future__ import annotations

import copy
import json
import math
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
SUBMISSION = ROOT / "submission"
if str(SUBMISSION) not in sys.path:
    sys.path.insert(0, str(SUBMISSION))

import main as submission_main
from eps_growth_family import _entity_signal as eps_entity_signal
from eps_growth_family import run_eps_growth
from safe_calibration import apply_safe_calibration
from strong_rag_baseline.agent import _parse_model_json
from strong_rag_baseline.indexer import build_index
from strong_rag_baseline.retriever import BM25Index

UNITS = ROOT / "reference" / "track4-analysis-public" / "units"


def load_task(unit: str) -> dict:
    return json.loads((UNITS / unit / "task.json").read_text(encoding="utf-8"))


def load_index(unit: str):
    task = load_task(unit)
    corpus = build_index(UNITS / unit / "corpus", task["cutoff_date"])
    return task, corpus, BM25Index(corpus.chunks, task["cutoff_date"])


class V3SafeUpgradeTests(unittest.TestCase):
    def test_only_semantic_families_bypass_house(self) -> None:
        self.assertIn("eps_growth_regression", submission_main.SPECIALIZED_RUNNERS)
        self.assertNotIn("positioning_shift", submission_main.SPECIALIZED_RUNNERS)
        self.assertNotIn("rate_curve_cross_section", submission_main.SPECIALIZED_RUNNERS)
        self.assertNotIn("post_earnings_reaction", submission_main.SPECIALIZED_RUNNERS)

    def test_rate_calibration_changes_only_interval(self) -> None:
        task = load_task("t4-fomc-curve-20220728")
        answer = {
            "entity_predictions": [
                {
                    "entity_id": "UST2Y",
                    "point_forecast": 17.25,
                    "interval": {"level": 0.9, "lo": 16.25, "hi": 18.25},
                    "claims": [{"doc_id": "D", "span_start": 1, "span_end": 2}],
                }
            ],
            "notes": {},
        }
        before = copy.deepcopy(answer["entity_predictions"][0])
        out = apply_safe_calibration(task, answer)
        row = out["entity_predictions"][0]
        self.assertEqual(row["point_forecast"], before["point_forecast"])
        self.assertEqual(row["claims"], before["claims"])
        self.assertAlmostEqual(row["interval"]["hi"] - row["point_forecast"], 112.915)
        self.assertAlmostEqual(row["point_forecast"] - row["interval"]["lo"], 112.915)
        self.assertEqual(
            out["notes"]["rate_interval_calibration"]["selected_train_quantile"],
            0.995,
        )

    def test_rate_calibration_is_cutoff_gated(self) -> None:
        task = load_task("t4-fomc-curve-20220728")
        task["cutoff_date"] = "2021-12-31"
        answer = {
            "entity_predictions": [
                {
                    "entity_id": "UST2Y",
                    "point_forecast": 0.0,
                    "interval": {"level": 0.9, "lo": -1.0, "hi": 1.0},
                    "claims": [],
                }
            ],
            "notes": {},
        }
        out = apply_safe_calibration(task, answer)
        self.assertEqual(out["entity_predictions"][0]["interval"]["lo"], -1.0)
        self.assertNotIn("rate_interval_calibration", out["notes"])

    def test_postearn_calibration_preserves_house_decision(self) -> None:
        task = load_task("t4-postearn-20240201-megacap")
        answer = {
            "entity_predictions": [
                {
                    "entity_id": "AAPL",
                    "label": "negative_reaction",
                    "point_forecast": -2.5,
                    "interval": {"level": 0.9, "lo": -3.5, "hi": -1.5},
                    "claims": [{"doc_id": "D", "span_start": 1, "span_end": 2}],
                }
            ],
            "notes": {},
        }
        before = copy.deepcopy(answer["entity_predictions"][0])
        out = apply_safe_calibration(task, answer)
        row = out["entity_predictions"][0]
        self.assertEqual(row["label"], before["label"])
        self.assertEqual(row["point_forecast"], before["point_forecast"])
        self.assertEqual(row["claims"], before["claims"])
        self.assertAlmostEqual(row["interval"]["lo"], -12.5)
        self.assertAlmostEqual(row["interval"]["hi"], 7.5)
        self.assertEqual(
            out["notes"]["postearn_interval_calibration"]["threshold_multiplier"],
            10.0,
        )

    def test_eps_growth_extracts_public_q2_yoy_signals(self) -> None:
        unit = "t4-eps-growth-2024Q3-banks"
        task, corpus, _ = load_index(unit)
        by_id = {e["entity_id"]: e for e in task["entities"]}
        expected = {
            "BAC": -5.681818,
            "C": 14.285714,
            "GS": 179.870130,
            "JPM": 28.842105,
            "MS": 47.0,
            "PNC": 0.892857,
            "USB": 15.476190,
            "WFC": 6.4,
        }
        for eid, value in expected.items():
            signal = eps_entity_signal(by_id[eid], corpus)
            self.assertIsNotNone(signal, eid)
            assert signal is not None
            self.assertAlmostEqual(signal.growth_pct, value, places=3, msg=eid)

    def test_eps_growth_uses_reproducible_cutoff_safe_parameters(self) -> None:
        unit = "t4-eps-growth-2024Q3-banks"
        task, corpus, index = load_index(unit)
        rows, notes = run_eps_growth(task, index, corpus)
        self.assertEqual(notes["adapter"], "eps_growth_safe_v3")
        self.assertAlmostEqual(notes["shrinkage_alpha"], 0.75)
        self.assertAlmostEqual(notes["signal_clip_pct_points"], 50.0)
        self.assertAlmostEqual(
            notes["interval_half_width_pct_points"], 77.52301640441917
        )
        by_id = {r["entity_id"]: r for r in rows}
        self.assertAlmostEqual(by_id["BAC"]["point_forecast"], -4.2613636, places=3)
        self.assertAlmostEqual(by_id["GS"]["point_forecast"], 37.5, places=6)
        self.assertAlmostEqual(
            by_id["BAC"]["interval"]["hi"] - by_id["BAC"]["point_forecast"],
            77.52301640441917,
        )
        for row in rows:
            self.assertNotIn("label", row)
            self.assertTrue(math.isfinite(float(row["point_forecast"])))
            self.assertTrue(row["claims"])

    def test_specialized_eps_emergency_never_recreates_dollar_as_growth_bug(self) -> None:
        unit = "t4-eps-growth-2024Q3-banks"
        task = load_task(unit)
        answer = submission_main.emergency_answer(task, UNITS / unit / "corpus")
        prior = {e["entity_id"]: e["prior_year_q_eps"] for e in task["entities"]}
        self.assertTrue(
            any(
                abs(float(r["point_forecast"]) - float(prior[r["entity_id"]])) > 1.0
                for r in answer["entity_predictions"]
            )
        )

    def test_house_parser_recovers_final_json_after_reasoning_braces(self) -> None:
        raw = (
            '<think>Compare candidate {A} with {B}; neither is final.</think>\n'
            '{"predictions":[{"entity_id":"E1","point_forecast":1.2}]}'
        )
        parsed = _parse_model_json(raw)
        self.assertEqual(parsed["predictions"][0]["entity_id"], "E1")
        self.assertAlmostEqual(parsed["predictions"][0]["point_forecast"], 1.2)


if __name__ == "__main__":
    unittest.main()

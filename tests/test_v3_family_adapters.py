from __future__ import annotations

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

from eps_growth_family import _entity_signal as eps_entity_signal
from eps_growth_family import run_eps_growth
from cpi_family import run_cpi_component
from strong_rag_baseline.agent import _parse_model_json
from postearn_family import run_postearn
from rate_curve_family import run_rate_curve
from strong_rag_baseline.indexer import build_index
from strong_rag_baseline.retriever import BM25Index


UNITS = ROOT / "reference" / "track4-analysis-public" / "units"


def load_task(unit: str) -> dict:
    return json.loads((UNITS / unit / "task.json").read_text(encoding="utf-8"))


def load_index(unit: str):
    task = load_task(unit)
    corpus = build_index(UNITS / unit / "corpus", task["cutoff_date"])
    return task, corpus, BM25Index(corpus.chunks, task["cutoff_date"])


class V3FamilyAdapterTests(unittest.TestCase):
    def test_rate_curve_adapter_is_finite_wide_and_complete(self) -> None:
        for unit in ("t4-fomc-curve-20220728", "t4-fomc-curve-20240918"):
            task, corpus, index = load_index(unit)
            rows, notes = run_rate_curve(task, index, corpus)
            self.assertEqual(len(rows), 6)
            self.assertEqual(notes["adapter"], "rate_curve_v3")
            for row in rows:
                self.assertNotIn("label", row)
                self.assertTrue(math.isfinite(float(row["point_forecast"])))
                interval = row["interval"]
                self.assertAlmostEqual(
                    float(interval["hi"]) - float(row["point_forecast"]), 80.0
                )
                self.assertAlmostEqual(
                    float(row["point_forecast"]) - float(interval["lo"]), 80.0
                )
                self.assertGreaterEqual(len(row["claims"]), 1)

    def test_rate_curve_easing_uses_shared_front_2y_gap(self) -> None:
        unit = "t4-fomc-curve-20240918"
        task, corpus, index = load_index(unit)
        rows, notes = run_rate_curve(task, index, corpus)
        self.assertEqual(notes["adapter"], "rate_curve_v3")
        self.assertAlmostEqual(notes["front_2y_yield_pct"], 3.59, places=2)
        by_id = {r["entity_id"]: r for r in rows}
        # Regression guard for the v3 semantic fix: the easing repricing gap is
        # defined from the front-end 2Y yield once, then shared across maturities.
        self.assertAlmostEqual(by_id["UST30Y"]["point_forecast"], 46.18, places=2)
        self.assertAlmostEqual(by_id["UST10Y"]["point_forecast"], 71.50, places=2)

    def test_eps_growth_extracts_all_public_q2_yoy_signals(self) -> None:
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

    def test_postearn_adapter_is_entity_scoped_and_wide(self) -> None:
        unit = "t4-postearn-20240201-megacap"
        task, corpus, index = load_index(unit)
        rows, notes = run_postearn(task, index, corpus)
        self.assertEqual(notes["adapter"], "postearn_v1")
        by_entity = {e["entity_id"]: e for e in task["entities"]}
        expected = {
            "AAPL": "negative_reaction",
            "AMZN": "positive_reaction",
            "META": "positive_reaction",
        }
        self.assertEqual({r["entity_id"] for r in rows}, set(expected))
        for row in rows:
            eid = row["entity_id"]
            self.assertEqual(row["label"], expected[eid])
            self.assertAlmostEqual(
                row["interval"]["hi"] - row["point_forecast"], 20.0
            )
            self.assertAlmostEqual(
                row["point_forecast"] - row["interval"]["lo"], 20.0
            )
            cik = by_entity[eid]["cik"]
            self.assertTrue(row["claims"])
            self.assertTrue(all(cik in c["doc_id"] for c in row["claims"]))

    def test_specialized_fallbacks_do_not_recreate_v2_bad_defaults(self) -> None:
        import main as submission_main

        # FOMC fallback must not collapse back to 0 +/- 1 bps.
        unit = "t4-fomc-curve-20220728"
        task = load_task(unit)
        answer = submission_main.emergency_answer(task, UNITS / unit / "corpus")
        for row in answer["entity_predictions"]:
            width = row["interval"]["hi"] - row["interval"]["lo"]
            self.assertGreaterEqual(width, 160.0 - 1e-9)

        # EPS growth fallback must not use prior-year EPS dollar level as growth %.
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


    def test_eps_growth_uses_historically_calibrated_shrinkage(self) -> None:
        unit = "t4-eps-growth-2024Q3-banks"
        task, corpus, index = load_index(unit)
        rows, notes = run_eps_growth(task, index, corpus)
        self.assertAlmostEqual(notes["shrinkage_alpha"], 0.75)
        self.assertAlmostEqual(notes["shrinkage_target_pct"], 0.0)
        self.assertAlmostEqual(notes["signal_clip_pct_points"], 60.0)
        by_id = {r["entity_id"]: r for r in rows}
        self.assertAlmostEqual(by_id["BAC"]["point_forecast"], -4.2613636, places=3)
        # GS has an extreme prior-quarter growth rate; nested historical
        # validation uses a +60pp cap before 0.75 persistence.
        self.assertAlmostEqual(by_id["GS"]["point_forecast"], 45.0, places=6)
        self.assertAlmostEqual(
            by_id["BAC"]["interval"]["hi"] - by_id["BAC"]["point_forecast"],
            46.0,
        )

    def test_cpi_family_remains_on_generic_v2_path(self) -> None:
        import main as submission_main
        self.assertNotIn("cpi_component_nowcast", submission_main.SPECIALIZED_RUNNERS)

    def test_cpi_adapter_uses_latest_published_mom_with_shrinkage(self) -> None:
        unit = "t4-cpicomp-202410-us11"
        task, corpus, index = load_index(unit)
        rows, notes = run_cpi_component(task, index, corpus)
        self.assertEqual(notes["adapter"], "cpi_component_v1")
        by_id = {r["entity_id"]: r for r in rows}
        self.assertAlmostEqual(by_id["CPI_CORE"]["point_forecast"], 0.31 * 0.25)
        self.assertAlmostEqual(by_id["CPI_GASOLINE"]["point_forecast"], -4.1 * 0.25)
        self.assertNotEqual(by_id["CPI_GASOLINE"]["point_forecast"], 0.0)
        self.assertTrue(all(r["claims"] for r in rows))

    def test_house_parser_recovers_final_json_after_reasoning_braces(self) -> None:
        raw = (
            '<think>Compare candidate {A} with {B}; neither is final.</think>\n'
            '```json\n{"predictions":[{"entity_id":"E1","point_forecast":1.2}]}\n```'
        )
        parsed = _parse_model_json(raw)
        self.assertEqual(parsed["predictions"][0]["entity_id"], "E1")
        self.assertAlmostEqual(parsed["predictions"][0]["point_forecast"], 1.2)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
SUBMISSION = HERE.parents[1]
if str(SUBMISSION) not in sys.path:
    sys.path.insert(0, str(SUBMISSION))

import main as submission_main
from output_contract import sanitize_answer
from strong_rag_baseline.indexer import Chunk
from strong_rag_baseline.retriever import BM25Index


class SubmissionV2ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunk = Chunk(
            doc_id="DOC_A",
            doc_date="2024-01-01",
            span_start=0,
            span_end=50,
            text="Revenue and operating income guidance improved year over year.",
        )

    def test_regression_prediction_omits_label_even_if_house_returns_null(self) -> None:
        task = {
            "target": {"type": "regression"},
            "interval_level": 0.90,
            "family": "test",
        }
        row = submission_main._prediction_row(
            task,
            {"entity_id": "E1"},
            {"label": None, "point_forecast": 1.5},
            [self.chunk],
        )
        self.assertNotIn("label", row)
        self.assertEqual(row["point_forecast"], 1.5)

    def test_ranking_prediction_omits_label(self) -> None:
        task = {
            "target": {"type": "ranking"},
            "interval_level": 0.90,
            "family": "test",
        }
        row = submission_main._prediction_row(
            task,
            {"entity_id": "E1"},
            {"label": None, "point_forecast": 2.0},
            [self.chunk],
        )
        self.assertNotIn("label", row)

    def test_classification_prediction_keeps_valid_label(self) -> None:
        task = {
            "target": {"type": "classification", "labels": ["up", "down"]},
            "interval_level": 0.90,
            "family": "test",
        }
        row = submission_main._prediction_row(
            task,
            {"entity_id": "E1"},
            {"label": "down", "point_forecast": 0.0},
            [self.chunk],
        )
        self.assertEqual(row["label"], "down")

    def test_sanitizer_removes_only_nonapplicable_optional_fields(self) -> None:
        answer = {
            "entity_predictions": [
                {
                    "entity_id": "E1",
                    "label": None,
                    "point_forecast": 1.0,
                    "interval": {"level": 0.9, "lo": 0.0, "hi": 2.0},
                    "claims": [{"x": 1}],
                }
            ]
        }
        sanitize_answer(answer, {"target": {"type": "regression"}})
        self.assertNotIn("label", answer["entity_predictions"][0])
        self.assertIn("point_forecast", answer["entity_predictions"][0])

    def test_postearn_candidates_never_cross_cik(self) -> None:
        chunks = [
            Chunk(
                doc_id="EDGAR_0000320193_10K_20231103",
                doc_date="2024-01-01",
                span_start=0,
                span_end=100,
                text="Revenue and diluted earnings per share improved.",
            ),
            Chunk(
                doc_id="EDGAR_0001018724_10Q_20231027",
                doc_date="2024-01-01",
                span_start=0,
                span_end=100,
                text="Revenue guidance and operating income increased.",
            ),
        ]
        index = BM25Index(chunks, "2024-01-31")
        task = {
            "family": "post_earnings_reaction",
            "target": {"type": "classification", "name": "abnormal return"},
            "prompt": "Forecast abnormal return after earnings.",
        }
        entity = {
            "entity_id": "AAPL",
            "name": "Apple Inc.",
            "cik": "0000320193",
        }
        result = submission_main._candidates(task, entity, index)
        self.assertTrue(result)
        self.assertTrue(all("0000320193" in x.doc_id for x in result))

    def test_postearn_business_span_ranks_ahead_of_noise(self) -> None:
        chunks = [
            Chunk(
                doc_id="EDGAR_0000320193_10K_20231103",
                doc_date="2024-01-01",
                span_start=0,
                span_end=80,
                text="Tax Court and signature information for the filing.",
            ),
            Chunk(
                doc_id="EDGAR_0000320193_10K_20231103",
                doc_date="2024-01-01",
                span_start=100,
                span_end=220,
                text=(
                    "Net sales increased year over year. Operating income and "
                    "gross margin improved. Diluted earnings per share rose."
                ),
            ),
        ]
        index = BM25Index(chunks, "2024-01-31")
        task = {
            "family": "post_earnings_reaction",
            "target": {"type": "classification", "name": "abnormal return"},
            "prompt": "Forecast abnormal return after earnings.",
        }
        entity = {
            "entity_id": "AAPL",
            "name": "Apple Inc.",
            "cik": "0000320193",
        }
        result = submission_main._candidates(task, entity, index)
        self.assertTrue(result)
        self.assertIn("Net sales", result[0].text)


if __name__ == "__main__":
    unittest.main()

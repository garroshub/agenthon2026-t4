from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cpi_structured_family as cpi
import main
from output_contract import finalize_answer
from safe_calibration import apply_safe_calibration
from strong_rag_baseline.config import Config
from strong_rag_baseline.indexer import build_index

REF = ROOT.parent / "reference" / "track4-analysis-public" / "units" / "t4-cpicomp-202410-us11"
TASK = json.loads((REF / "task.json").read_text(encoding="utf-8"))
CORPUS = REF / "corpus"


def cfg() -> Config:
    return Config(
        model_endpoint="http://house.test",
        model_id="house",
        model_token=None,
        seed=20260731,
        top_k=10,
        timeout_s=60.0,
        max_retries=1,
        temperature=0.0,
        max_tokens=3000,
        unit_timeout_s=540.0,
    )


def prediction(eid: str, point: float) -> dict:
    return {
        "entity_id": eid,
        "point_forecast": point,
        "interval": {"lo": point - 1.0, "hi": point + 1.0},
        "candidate_ids": ["c1"],
    }


class SequenceClient:
    calls: list[tuple[str, str]] = []
    replies: list[object] = []

    def __init__(self, _config: Config):
        pass

    def complete(self, system: str, user: str) -> str:
        type(self).calls.append((system, user))
        if not type(self).replies:
            raise RuntimeError("no reply")
        value = type(self).replies.pop(0)
        if isinstance(value, Exception):
            raise value
        return str(value)


class CPIStructuredPacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = build_index(CORPUS, TASK["cutoff_date"])
        cls.cards, cls.by_entity, cls.diag = cpi.build_cpi_context(
            TASK, CORPUS, cls.corpus
        )

    def setUp(self) -> None:
        SequenceClient.calls = []
        SequenceClient.replies = []

    def test_all_component_cards_match_task_and_frozen_september(self) -> None:
        self.assertEqual(len(self.cards), 11)
        for entity in TASK["entities"]:
            eid = entity["entity_id"]
            card = self.cards[eid]
            self.assertEqual(card["series_fred"], entity["series_fred"])
            self.assertEqual(card["target_unit"], "mom_pct_change_sa")
            self.assertEqual(len(card["history_2024_jan_sep_sa_mom_pct"]), 9)
            self.assertAlmostEqual(
                card["history_2024_jan_sep_sa_mom_pct"][-1],
                float(entity["latest_published_mom_pct"]),
                places=9,
            )

    def test_eia_is_exposed_only_to_energy_and_gasoline(self) -> None:
        eia_entities = {
            eid
            for eid, chunks in self.by_entity.items()
            if any(c.doc_id == cpi.EIA_DOC for c in chunks)
        }
        self.assertEqual(eia_entities, {"CPI_ENERGY", "CPI_GASOLINE"})
        for eid, chunks in self.by_entity.items():
            docs = [c.doc_id for c in chunks]
            self.assertIn(cpi.ALFRED_DOC, docs)
            self.assertIn(cpi.BLS_DOC, docs)

    def test_prompt_contains_component_fact_cards_and_sa_guardrail(self) -> None:
        batch = TASK["entities"][:6]
        system, user = cpi.cpi_batch_prompt(
            TASK, batch, self.cards, self.by_entity
        )
        payload = json.loads(user)
        self.assertEqual(
            [x["entity_id"] for x in payload["entities"]],
            [x["entity_id"] for x in batch],
        )
        self.assertIn("seasonally adjusted", system.lower())
        self.assertIn("not", system.lower())
        for item in payload["entities"]:
            self.assertIn("history_2024_jan_sep_sa_mom_pct", item["fact_card"])
            self.assertIn("series_fred", item["fact_card"])

    def test_mock_path_returns_valid_eleven_entity_answer(self) -> None:
        answer = cpi.run_cpi_structured(
            TASK,
            CORPUS,
            cfg(),
            True,
            6,
            4,
            45.0,
            main._prediction_row,
            main._merge_valid_prediction_items,
        )
        answer = apply_safe_calibration(TASK, answer)
        answer = finalize_answer(answer, TASK, CORPUS)
        self.assertEqual(len(answer["entity_predictions"]), 11)
        self.assertTrue(answer["notes"]["cpi_structured_packet"])
        self.assertEqual(answer["notes"]["house_fallback_entities"], [])

    def test_partial_second_batch_gets_targeted_repair(self) -> None:
        first_ids = [e["entity_id"] for e in TASK["entities"][:6]]
        second_ids = [e["entity_id"] for e in TASK["entities"][6:]]
        SequenceClient.replies = [
            json.dumps({"predictions": [prediction(eid, 0.1 + i/100) for i, eid in enumerate(first_ids)]}),
            json.dumps({"predictions": [prediction(eid, 0.2 + i/100) for i, eid in enumerate(second_ids[:2])]}),
            json.dumps({"predictions": [prediction(eid, 0.3 + i/100) for i, eid in enumerate(second_ids[2:])]}),
        ]
        with patch.object(cpi, "HTTPModelClient", SequenceClient):
            answer = cpi.run_cpi_structured(
                TASK,
                CORPUS,
                cfg(),
                False,
                6,
                4,
                45.0,
                main._prediction_row,
                main._merge_valid_prediction_items,
            )
        self.assertEqual(len(SequenceClient.calls), 3)
        self.assertEqual(answer["notes"]["house_repair_calls"], 1)
        self.assertEqual(
            set(answer["notes"]["house_repaired_entities"]),
            set(second_ids[2:]),
        )
        self.assertEqual(answer["notes"]["house_fallback_entities"], [])
        self.assertIn("STRICT REPAIR CALL", SequenceClient.calls[2][0])

    def test_claim_spans_exist_in_index(self) -> None:
        answer = cpi.run_cpi_structured(
            TASK,
            CORPUS,
            cfg(),
            True,
            6,
            4,
            45.0,
            main._prediction_row,
            main._merge_valid_prediction_items,
        )
        valid = {(x.doc_id, x.span_start, x.span_end) for x in self.corpus.chunks}
        for row in answer["entity_predictions"]:
            self.assertGreaterEqual(len(row["claims"]), 1)
            for claim in row["claims"]:
                key = (claim["doc_id"], claim["span_start"], claim["span_end"])
                self.assertIn(key, valid)


if __name__ == "__main__":
    unittest.main()

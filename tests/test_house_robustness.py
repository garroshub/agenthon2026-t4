from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main
from strong_rag_baseline.config import Config

REF = ROOT.parent / "reference" / "track4-analysis-public" / "units" / "t4-postearn-20240201-megacap"
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


def row(eid: str, point: float, label: str | None = None) -> dict:
    out = {
        "entity_id": eid,
        "point_forecast": point,
        "interval": {"lo": point - 4.0, "hi": point + 4.0},
        "candidate_ids": ["c1"],
    }
    if label is not None:
        out["label"] = label
    return out


class SequenceClient:
    replies: list[object] = []
    calls: list[tuple[str, str]] = []

    def __init__(self, _config: Config):
        pass

    def complete(self, system: str, user: str) -> str:
        type(self).calls.append((system, user))
        if not type(self).replies:
            raise RuntimeError("no reply configured")
        value = type(self).replies.pop(0)
        if isinstance(value, Exception):
            raise value
        return str(value)


class HouseRobustnessTests(unittest.TestCase):
    def setUp(self) -> None:
        SequenceClient.replies = []
        SequenceClient.calls = []

    def _run(self, replies: list[object]) -> dict:
        SequenceClient.replies = list(replies)
        with patch.object(main, "HTTPModelClient", SequenceClient):
            return main.generic_run(TASK, CORPUS, cfg(), False)

    def test_complete_first_pass_is_unchanged_and_no_repair_call(self) -> None:
        reply = json.dumps(
            {
                "predictions": [
                    row("AAPL", 2.5, "positive_reaction"),
                    row("AMZN", -2.0, "negative_reaction"),
                    row("META", 0.4, "flat"),
                ]
            }
        )
        ans = self._run([reply])
        self.assertEqual(len(SequenceClient.calls), 1)
        self.assertEqual(ans["notes"]["house_repair_calls"], 0)
        self.assertEqual(ans["notes"]["house_fallback_entities"], [])
        got = {r["entity_id"]: r["point_forecast"] for r in ans["entity_predictions"]}
        self.assertEqual(got, {"AAPL": 2.5, "AMZN": -2.0, "META": 0.4})

    def test_partial_roster_gets_one_targeted_repair(self) -> None:
        first = json.dumps(
            {
                "predictions": [
                    row("AAPL", 2.5, "positive_reaction"),
                    row("AMZN", -2.0, "negative_reaction"),
                ]
            }
        )
        repair = json.dumps({"predictions": [row("META", 3.1, "positive_reaction")]})
        ans = self._run([first, repair])
        self.assertEqual(len(SequenceClient.calls), 2)
        self.assertEqual(ans["notes"]["house_repair_calls"], 1)
        self.assertEqual(ans["notes"]["house_repaired_entities"], ["META"])
        self.assertEqual(ans["notes"]["house_fallback_entities"], [])
        self.assertIn("STRICT REPAIR CALL", SequenceClient.calls[1][0])
        payload = json.loads(SequenceClient.calls[1][1])
        self.assertEqual([x["entity_id"] for x in payload["entities"]], ["META"])

    def test_malformed_first_response_repairs_entire_roster(self) -> None:
        repair = json.dumps(
            {
                "predictions": [
                    row("AAPL", 1.1, "positive_reaction"),
                    row("AMZN", 1.2, "positive_reaction"),
                    row("META", 1.3, "positive_reaction"),
                ]
            }
        )
        ans = self._run(["reasoning only, no JSON", repair])
        self.assertEqual(ans["notes"]["house_repair_calls"], 1)
        self.assertEqual(ans["notes"]["house_fallback_entities"], [])
        self.assertEqual(set(ans["notes"]["house_repaired_entities"]), {"AAPL", "AMZN", "META"})

    def test_nonfinite_and_duplicate_bad_row_do_not_block_valid_repair(self) -> None:
        first = json.dumps(
            {
                "predictions": [
                    row("AAPL", 2.0, "positive_reaction"),
                    {"entity_id": "AMZN", "point_forecast": "bad", "candidate_ids": ["c1"]},
                    {"entity_id": "AMZN", "point_forecast": None, "candidate_ids": ["c1"]},
                    row("META", -2.0, "negative_reaction"),
                ]
            }
        )
        repair = json.dumps({"predictions": [row("AMZN", 4.0, "positive_reaction")]})
        ans = self._run([first, repair])
        self.assertEqual(ans["notes"]["house_repaired_entities"], ["AMZN"])
        got = {r["entity_id"]: r["point_forecast"] for r in ans["entity_predictions"]}
        self.assertEqual(got["AMZN"], 4.0)

    def test_failed_repair_falls_back_without_overwriting_good_rows(self) -> None:
        first = json.dumps({"predictions": [row("AAPL", 2.5, "positive_reaction")]})
        ans = self._run([first, RuntimeError("repair timeout")])
        self.assertEqual(ans["notes"]["house_repair_calls"], 1)
        self.assertEqual(
            set(ans["notes"]["house_fallback_entities"]),
            {"AMZN", "META"},
        )
        got = {r["entity_id"]: r for r in ans["entity_predictions"]}
        self.assertEqual(got["AAPL"]["point_forecast"], 2.5)
        self.assertEqual(got["AMZN"]["point_forecast"], 0.0)
        self.assertEqual(got["META"]["point_forecast"], 0.0)


if __name__ == "__main__":
    unittest.main()

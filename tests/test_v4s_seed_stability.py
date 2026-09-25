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

UNITS = ROOT.parent / "reference" / "track4-analysis-public" / "units"


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


class SeedAwareClient:
    calls: list[dict] = []

    def __init__(self, config: Config):
        self.config = config

    def complete(self, system: str, user: str) -> str:
        payload = json.loads(user)
        ents = payload["entities"]
        family = payload["family"]
        type(self).calls.append(
            {
                "seed": self.config.seed,
                "system": system,
                "user": user,
                "ids": [x["entity_id"] for x in ents],
                "family": family,
            }
        )
        rows = []
        for i, item in enumerate(ents):
            eid = item["entity_id"]
            if family == "positioning_shift":
                # Base and alt seeds deliberately disagree on scale/order.
                if self.config.seed == 20260731:
                    point = float(i + (0 if eid in {"CORN_CBT","ES_SP500","EURO_FX","GOLD_CMX","JPY_CME","NATGAS_NYMEX"} else 10))
                else:
                    point = float(20 - i if eid in {"CORN_CBT","ES_SP500","EURO_FX","GOLD_CMX","JPY_CME","NATGAS_NYMEX"} else 5 - i)
                rows.append(
                    {
                        "entity_id": eid,
                        "point_forecast": point,
                        "interval": {"lo": point - 2, "hi": point + 2},
                        "rank": i + 1,
                        "candidate_ids": ["c1"],
                    }
                )
            elif family == "credit_event":
                base = self.config.seed == 20260731
                # BBBY agrees positive; BBY deliberately disagrees; rest agree no_event.
                if eid == "BBBY":
                    label = "credit_event"
                    point = 0.8 if base else 0.6
                elif eid == "BBY":
                    label = "no_event" if base else "credit_event"
                    point = 0.2 if base else 0.7
                else:
                    label = "no_event"
                    point = 0.25 if base else 0.15
                rows.append(
                    {
                        "entity_id": eid,
                        "point_forecast": point,
                        "interval": {"lo": max(0.0, point - 0.1), "hi": min(1.0, point + 0.1)},
                        "label": label,
                        "candidate_ids": ["c1"],
                    }
                )
            else:
                rows.append(
                    {
                        "entity_id": eid,
                        "point_forecast": 0.1,
                        "interval": {"lo": -1.0, "hi": 1.0},
                        "candidate_ids": ["c1"],
                    }
                )
        return json.dumps({"predictions": rows})


class V4SSeedStabilityRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        SeedAwareClient.calls = []

    def _task(self, unit: str) -> tuple[dict, Path]:
        p = UNITS / unit
        return json.loads((p / "task.json").read_text(encoding="utf-8")), p / "corpus"

    def test_cot_uses_exactly_two_seeds_and_four_calls(self) -> None:
        task, corpus = self._task("t4-cotpos-202411-us10")
        with patch.object(main, "HTTPModelClient", SeedAwareClient):
            ans = main.generic_run(task, corpus, cfg(), False)

        self.assertEqual(len(SeedAwareClient.calls), 4)
        seeds = [x["seed"] for x in SeedAwareClient.calls]
        self.assertEqual(seeds[:2], [20260731, 20260731])
        self.assertEqual(seeds[2:], [20260731 + main._STABILITY_ALT_SEED_OFFSET] * 2)
        self.assertEqual(ans["notes"]["house_calls_attempted"], 4)
        self.assertTrue(ans["notes"]["house_stability_ensemble"])
        self.assertEqual(len(ans["entity_predictions"]), 10)

        # Prompts for corresponding batches must be identical except seed lives outside prompt.
        self.assertEqual(SeedAwareClient.calls[0]["system"], SeedAwareClient.calls[2]["system"])
        self.assertEqual(SeedAwareClient.calls[0]["user"], SeedAwareClient.calls[2]["user"])
        self.assertEqual(SeedAwareClient.calls[1]["system"], SeedAwareClient.calls[3]["system"])
        self.assertEqual(SeedAwareClient.calls[1]["user"], SeedAwareClient.calls[3]["user"])

        ranks = sorted(r["rank"] for r in ans["entity_predictions"])
        self.assertEqual(ranks, list(range(1, 11)))

    def test_credit_agreement_averages_and_disagreement_keeps_base(self) -> None:
        task, corpus = self._task("t4-credit-event-2023")
        with patch.object(main, "HTTPModelClient", SeedAwareClient):
            ans = main.generic_run(task, corpus, cfg(), False)

        self.assertEqual(len(SeedAwareClient.calls), 4)
        got = {r["entity_id"]: r for r in ans["entity_predictions"]}
        self.assertEqual(got["BBBY"]["label"], "credit_event")
        self.assertAlmostEqual(got["BBBY"]["point_forecast"], 0.7)
        self.assertEqual(
            ans["notes"]["house_stability_actions"]["BBBY"],
            "agreed_label_average_point",
        )

        self.assertEqual(got["BBY"]["label"], "no_event")
        self.assertAlmostEqual(got["BBY"]["point_forecast"], 0.2)
        self.assertEqual(
            ans["notes"]["house_stability_actions"]["BBY"],
            "base_on_label_disagreement",
        )

    def test_non_stability_family_remains_single_seed_v3_path(self) -> None:
        task, corpus = self._task("t4-postearn-20240201-megacap")
        with patch.object(main, "HTTPModelClient", SeedAwareClient):
            ans = main.generic_run(task, corpus, cfg(), False)

        self.assertEqual(len(SeedAwareClient.calls), 1)
        self.assertEqual(SeedAwareClient.calls[0]["seed"], 20260731)
        self.assertNotIn("house_stability_ensemble", ans["notes"])

    def test_mock_mode_does_not_fake_ensemble(self) -> None:
        task, corpus = self._task("t4-credit-event-2023")
        ans = main.generic_run(task, corpus, cfg(), True)
        self.assertEqual(ans["notes"]["house_calls_attempted"], 0)
        self.assertNotIn("house_stability_ensemble", ans["notes"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import main
from safe_calibration import apply_safe_calibration
from strong_rag_baseline.config import Config

UNITS=ROOT.parent/"reference"/"track4-analysis-public"/"units"

def cfg():
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

class NarrowIntervalClient:
    calls=[]
    def __init__(self,config):
        self.config=config
    def complete(self,system,user):
        payload=json.loads(user)
        family=payload["family"]
        rows=[]
        for i,e in enumerate(payload["entities"]):
            eid=e["entity_id"]
            if family=="credit_event":
                if eid in {"BBBY","RAD","WE","YELL"}:
                    label="credit_event"; point=.70 if self.config.seed==20260731 else .60
                else:
                    label="no_event"; point=.20 if self.config.seed==20260731 else .30
                rows.append({
                    "entity_id":eid,
                    "label":label,
                    "point_forecast":point,
                    "interval":{"lo":max(0,point-.05),"hi":min(1,point+.05)},
                    "candidate_ids":["c1"],
                })
            elif family=="eps_yoy_direction":
                prior=float(e.get("prior_year_q_eps",1.0))
                point=prior+(.25 if i%2==0 else -.25)
                label="up" if point>prior else "down"
                rows.append({
                    "entity_id":eid,
                    "label":label,
                    "point_forecast":point,
                    "interval":{"lo":point-.05,"hi":point+.05},
                    "candidate_ids":["c1"],
                })
            else:
                rows.append({
                    "entity_id":eid,
                    "point_forecast":0.0,
                    "interval":{"lo":-1.0,"hi":1.0},
                    "candidate_ids":["c1"],
                })
        return json.dumps({"predictions":rows})

class V52RuntimeOrderTests(unittest.TestCase):
    def _load(self,unit):
        p=UNITS/unit
        return json.loads((p/"task.json").read_text(encoding="utf-8")),p/"corpus"

    def test_credit_house_decision_survives_interval_rail(self):
        task,corpus=self._load("t4-credit-event-2023")
        with patch.object(main,"HTTPModelClient",NarrowIntervalClient):
            raw=main.generic_run(task,corpus,cfg(),False)
        before={r["entity_id"]:(r["label"],r["point_forecast"],r["claims"]) for r in raw["entity_predictions"]}
        out=apply_safe_calibration(task,raw)
        for r in out["entity_predictions"]:
            self.assertEqual((r["label"],r["point_forecast"],r["claims"]),before[r["entity_id"]])
            self.assertEqual(r["interval"],{"level":0.9,"lo":0.0,"hi":1.0})
        self.assertTrue(out["notes"]["house_stability_ensemble"])
        self.assertEqual(out["notes"]["house_calls_attempted"],4)

    def test_eps_house_decision_survives_historical_interval_rail(self):
        task,corpus=self._load("t4-eps-yoy-2023Q2-mixed")
        with patch.object(main,"HTTPModelClient",NarrowIntervalClient):
            raw=main.generic_run(task,corpus,cfg(),False)
        before={r["entity_id"]:(r["label"],r["point_forecast"],r["claims"]) for r in raw["entity_predictions"]}
        entities={e["entity_id"]:e for e in task["entities"]}
        out=apply_safe_calibration(task,raw)
        half=0.9440000000000011
        for r in out["entity_predictions"]:
            self.assertEqual((r["label"],r["point_forecast"],r["claims"]),before[r["entity_id"]])
            prior=float(entities[r["entity_id"]]["prior_year_q_eps"])
            self.assertLessEqual(r["interval"]["lo"],prior-half+1e-12)
            self.assertGreaterEqual(r["interval"]["hi"],prior+half-1e-12)
            self.assertLessEqual(r["interval"]["lo"],float(r["point_forecast"]))
            self.assertGreaterEqual(r["interval"]["hi"],float(r["point_forecast"]))
        self.assertEqual(raw["notes"]["house_calls_attempted"],1)

if __name__=="__main__":
    unittest.main()

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import safe_calibration

ROOT=Path(__file__).resolve().parents[1]
UNITS=ROOT.parent/"reference"/"track4-analysis-public"/"units"

class V52SafeCalibrationTests(unittest.TestCase):
    def _task(self,unit):
        return json.loads((UNITS/unit/"task.json").read_text(encoding="utf-8"))

    def test_credit_full_probability_support_preserves_decision(self):
        task=self._task("t4-credit-event-2023")
        answer={
            "entity_predictions":[{
                "entity_id":"BBBY",
                "point_forecast":0.72,
                "label":"credit_event",
                "interval":{"level":0.9,"lo":0.60,"hi":0.82},
                "claims":[{"doc_id":"x","span_start":1,"span_end":2,"claim":"x"}],
            }],
            "notes":{"keep":"yes"},
        }
        before=copy.deepcopy(answer)
        out=safe_calibration.apply_safe_calibration(task,answer)
        row=out["entity_predictions"][0]
        self.assertEqual(row["point_forecast"],before["entity_predictions"][0]["point_forecast"])
        self.assertEqual(row["label"],before["entity_predictions"][0]["label"])
        self.assertEqual(row["claims"],before["entity_predictions"][0]["claims"])
        self.assertEqual(row["interval"],{"level":0.9,"lo":0.0,"hi":1.0})
        self.assertEqual(out["notes"]["keep"],"yes")

    def test_eps_q96_band_preserves_point_label_claims(self):
        task=self._task("t4-eps-yoy-2023Q2-mixed")
        answer={
            "entity_predictions":[{
                "entity_id":"AMD",
                "point_forecast":0.12,
                "label":"down",
                "interval":{"level":0.9,"lo":0.0,"hi":0.3},
                "claims":[{"doc_id":"x","span_start":1,"span_end":2,"claim":"x"}],
            }],
            "notes":{},
        }
        before=copy.deepcopy(answer)
        out=safe_calibration.apply_safe_calibration(task,answer)
        row=out["entity_predictions"][0]
        prior=0.27
        half=safe_calibration._EPS_YOY_PRIOR_BAND_HALF_WIDTH
        self.assertEqual(row["point_forecast"],before["entity_predictions"][0]["point_forecast"])
        self.assertEqual(row["label"],before["entity_predictions"][0]["label"])
        self.assertEqual(row["claims"],before["entity_predictions"][0]["claims"])
        self.assertAlmostEqual(row["interval"]["lo"],min(prior-half,0.12))
        self.assertAlmostEqual(row["interval"]["hi"],max(prior+half,0.12))

    def test_eps_band_expands_to_contain_extreme_house_point(self):
        task=self._task("t4-eps-yoy-2023Q2-mixed")
        answer={
            "entity_predictions":[{
                "entity_id":"AMD","point_forecast":3.0,"label":"up",
                "interval":{"level":0.9,"lo":2.5,"hi":3.5},
                "claims":[{"doc_id":"x","span_start":1,"span_end":2,"claim":"x"}],
            }],
            "notes":{},
        }
        out=safe_calibration.apply_safe_calibration(task,answer)
        row=out["entity_predictions"][0]
        self.assertEqual(row["interval"]["hi"],3.0)
        self.assertLess(row["interval"]["lo"],0.0)

    def test_eps_artifact_does_not_apply_before_availability(self):
        task=self._task("t4-eps-yoy-2023Q2-mixed")
        task["cutoff_date"]="2023-01-01"
        answer={
            "entity_predictions":[{
                "entity_id":"AMD","point_forecast":0.12,"label":"down",
                "interval":{"level":0.9,"lo":-0.2,"hi":0.4},
                "claims":[{"doc_id":"x","span_start":1,"span_end":2,"claim":"x"}],
            }],
            "notes":{},
        }
        before=copy.deepcopy(answer)
        out=safe_calibration.apply_safe_calibration(task,answer)
        self.assertEqual(out,before)

    def test_postearn_rule_unchanged(self):
        task=self._task("t4-postearn-20240201-megacap")
        answer={
            "entity_predictions":[{
                "entity_id":"AAPL","point_forecast":2.0,"label":"positive_reaction",
                "interval":{"level":0.9,"lo":1.0,"hi":3.0},
                "claims":[{"doc_id":"x","span_start":1,"span_end":2,"claim":"x"}],
            }],
            "notes":{},
        }
        out=safe_calibration.apply_safe_calibration(task,answer)
        self.assertEqual(out["entity_predictions"][0]["interval"],{"level":0.9,"lo":-8.0,"hi":12.0})

if __name__=="__main__":
    unittest.main()

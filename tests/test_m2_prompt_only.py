from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main
from strong_rag_baseline.indexer import build_index
from strong_rag_baseline.retriever import BM25Index

UNITS = ROOT.parent / "reference" / "track4-analysis-public" / "units"


class M2PromptOnlyTests(unittest.TestCase):
    def test_postearn_uses_original_generic_evidence_and_schema(self) -> None:
        unit = UNITS / "t4-postearn-20240201-megacap"
        task = json.loads((unit / "task.json").read_text(encoding="utf-8"))
        corpus = build_index(unit / "corpus", task["cutoff_date"])
        index = BM25Index(corpus.chunks, task["cutoff_date"])
        entities = task["entities"]
        by = {e["entity_id"]: main._candidates(task, e, index) for e in entities}
        system, user = main._batch_prompt(task, entities, by)
        payload = json.loads(user)
        row = payload["output_schema"]["predictions"][0]

        self.assertEqual(row["label"], "one allowed task label")
        self.assertNotIn("probabilities", row)
        self.assertEqual(payload["target_type"], "classification")
        self.assertIn("do not invent it", system)
        self.assertIn("Distinguish backward-looking actual results", system)

        for item, entity in zip(payload["entities"], entities):
            self.assertEqual(item["entity_id"], entity["entity_id"])
            expected = [
                (c.doc_id, c.span_start, c.span_end)
                for c in by[entity["entity_id"]]
            ]
            actual = [
                (c["doc_id"], c["span_start"], c["span_end"])
                for c in item["evidence_candidates"]
            ]
            self.assertEqual(actual, expected)

    def test_non_postearn_prompt_does_not_receive_m2_language(self) -> None:
        unit = UNITS / "t4-cotpos-202411-us10"
        task = json.loads((unit / "task.json").read_text(encoding="utf-8"))
        corpus = build_index(unit / "corpus", task["cutoff_date"])
        index = BM25Index(corpus.chunks, task["cutoff_date"])
        entities = task["entities"][:2]
        by = {e["entity_id"]: main._candidates(task, e, index) for e in entities}
        system, _ = main._batch_prompt(task, entities, by)
        self.assertNotIn("do not invent it", system)
        self.assertNotIn("Distinguish backward-looking actual results", system)


if __name__ == "__main__":
    unittest.main()

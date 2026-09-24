from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main
from postearn_m1_evidence import build_postearn_cards
from strong_rag_baseline.indexer import build_index
from strong_rag_baseline.retriever import BM25Index

UNITS = ROOT.parent / "reference" / "track4-analysis-public" / "units"
POST = UNITS / "t4-postearn-20240201-megacap"


def sig(by):
    return {
        eid: [(c.doc_id, c.span_start, c.span_end) for c in chunks]
        for eid, chunks in by.items()
    }


class M1EvidenceOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.task = json.loads((POST / "task.json").read_text(encoding="utf-8"))
        cls.corpus = build_index(POST / "corpus", cls.task["cutoff_date"])
        cls.index = BM25Index(cls.corpus.chunks, cls.task["cutoff_date"])
        cls.entities = [e for e in cls.task["entities"] if isinstance(e, dict)]
        cls.v3_by = {
            str(e["entity_id"]): main._candidates(cls.task, e, cls.index)
            for e in cls.entities
        }
        _cards, cls.m1_by = build_postearn_cards(cls.task, cls.corpus)

    def test_only_evidence_changes_inside_batch_prompt(self) -> None:
        v3_system, v3_user = main._batch_prompt(self.task, self.entities, self.v3_by)
        m1_system, m1_user = main._batch_prompt(self.task, self.entities, self.m1_by)

        self.assertEqual(v3_system, m1_system)

        v3_payload = json.loads(v3_user)
        m1_payload = json.loads(m1_user)
        self.assertEqual(v3_payload["output_schema"], m1_payload["output_schema"])
        self.assertEqual(v3_payload["target"], m1_payload["target"])
        self.assertEqual(v3_payload["target_type"], m1_payload["target_type"])
        self.assertEqual(v3_payload["task_prompt"], m1_payload["task_prompt"])
        self.assertEqual(v3_payload["family"], m1_payload["family"])
        self.assertEqual(v3_payload["cutoff_date"], m1_payload["cutoff_date"])

        v3_entities = v3_payload["entities"]
        m1_entities = m1_payload["entities"]
        self.assertEqual(len(v3_entities), len(m1_entities))
        for a, b in zip(v3_entities, m1_entities):
            self.assertEqual(a["entity_id"], b["entity_id"])
            self.assertEqual(a["entity"], b["entity"])

        self.assertNotEqual(sig(self.v3_by), sig(self.m1_by))

    def test_no_probability_or_concrete_flat_schema(self) -> None:
        _, user = main._batch_prompt(self.task, self.entities, self.m1_by)
        row = json.loads(user)["output_schema"]["predictions"][0]
        self.assertNotIn("probabilities", row)
        self.assertEqual(row["label"], "one allowed task label")

    def test_aapl_guidance_is_not_invented(self) -> None:
        cards, _ = build_postearn_cards(self.task, self.corpus)
        self.assertEqual(cards["AAPL"].facts["guidance"], [])

    def test_amazon_explicit_guidance_is_present(self) -> None:
        cards, _ = build_postearn_cards(self.task, self.corpus)
        text = " ".join(c.text for c in cards["AMZN"].facts["guidance"]).lower()
        self.assertIn("fourth quarter 2023 guidance", text)
        self.assertIn("net sales are expected", text)

    def test_other_family_generic_candidates_unchanged(self) -> None:
        unit = UNITS / "t4-cotpos-202411-us10"
        task = json.loads((unit / "task.json").read_text(encoding="utf-8"))
        corpus = build_index(unit / "corpus", task["cutoff_date"])
        index = BM25Index(corpus.chunks, task["cutoff_date"])
        entity = task["entities"][0]
        a = main._candidates(task, entity, index)
        b = main._candidates(task, entity, index)
        self.assertEqual(
            [(x.doc_id, x.span_start, x.span_end) for x in a],
            [(x.doc_id, x.span_start, x.span_end) for x in b],
        )


if __name__ == "__main__":
    unittest.main()

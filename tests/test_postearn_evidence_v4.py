from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import postearn_evidence as pe
from strong_rag_baseline.indexer import build_index

UNIT = ROOT.parent / "reference" / "track4-analysis-public" / "units" / "t4-postearn-20240201-megacap"
TASK = json.loads((UNIT / "task.json").read_text(encoding="utf-8"))
CORPUS = build_index(UNIT / "corpus", TASK["cutoff_date"])


class PostearnEvidenceV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards, cls.by = pe.build_postearn_cards(TASK, CORPUS)

    def test_entity_roster_and_expectation_gap(self) -> None:
        self.assertEqual(set(self.cards), {"AAPL", "AMZN", "META"})
        for card in self.cards.values():
            self.assertEqual(
                card.expectation_gap_status,
                "missing_no_reliable_pre_earnings_market_expectation_in_frozen_sec_corpus",
            )

    def test_aapl_does_not_invent_guidance(self) -> None:
        self.assertEqual(self.cards["AAPL"].facts["guidance"], [])

    def test_amazon_contains_explicit_q4_guidance(self) -> None:
        text = " ".join(x.text for x in self.cards["AMZN"].facts["guidance"]).lower()
        self.assertIn("fourth quarter 2023 guidance", text)
        self.assertIn("net sales are expected", text)
        self.assertIn("operating income is expected", text)

    def test_meta_contains_material_forward_operating_information(self) -> None:
        text = " ".join(x.text for x in self.cards["META"].facts["guidance"]).lower()
        self.assertIn("2024", text)
        self.assertTrue(
            "operating losses to increase meaningfully" in text
            or "capital expenditures" in text
        )

    def test_selected_chunks_are_real_and_entity_specific(self) -> None:
        valid = {(x.doc_id, x.span_start, x.span_end) for x in CORPUS.chunks}
        cik_by_entity = {e["entity_id"]: e["cik"].lstrip("0") for e in TASK["entities"]}
        for eid, chunks in self.by.items():
            self.assertGreaterEqual(len(chunks), 3)
            for c in chunks:
                self.assertIn((c.doc_id, c.span_start, c.span_end), valid)
                self.assertIn(cik_by_entity[eid], c.doc_id)

    def test_evidence_categories_are_not_empty_except_aapl_guidance(self) -> None:
        for eid, card in self.cards.items():
            self.assertGreaterEqual(len(card.facts["operations"]), 1)
            self.assertGreaterEqual(len(card.facts["adverse"]), 1)
            if eid != "AAPL":
                self.assertGreaterEqual(len(card.facts["guidance"]), 1)


if __name__ == "__main__":
    unittest.main()

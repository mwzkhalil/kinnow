"""Tests for separate curated / one-edit decision cutoffs."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from evaluate import applies, cut_json, measure, pick, pick_split, verify_policy  # noqa: E402
from urdu_core import SplitCut, cutoff_for, suggestions  # noqa: E402


class FakeFixer:
    fingerprint = "m"
    asset = "r"


def doc(errors, preds, words=100):
    # (domain, eligible_words, gold_errors, covered, [(margin, status, kind)])
    return ("d", words, errors, errors, preds)


class SplitThreshold(unittest.TestCase):
    def test_suggestions_use_the_class_cutoff(self):
        d = [{"margin": 5.0, "curated": True}, {"margin": 5.0, "curated": False}, {"margin": 9.0, "curated": False}]
        self.assertEqual(len(suggestions(d, SplitCut(4.0, 8.0))), 2)
        self.assertEqual(len(suggestions(d, SplitCut(None, 8.0))), 1)
        self.assertEqual(len(suggestions(d, 4.0)), 3)  # a single number still works
        self.assertIsNone(cutoff_for(d[0], None))

    def test_policy_round_trip_and_validation(self):
        pol = {"model_sha256": "m", "resources_sha256": "r", "threshold": 9.0, "thresholds": {"curated": 5.0, "edit": None}}
        self.assertEqual(verify_policy(FakeFixer, pol), SplitCut(5.0, None))
        self.assertEqual(cut_json(SplitCut(5.0, None)), {"curated": 5.0, "edit": None})
        self.assertEqual(verify_policy(FakeFixer, {"model_sha256": "m", "resources_sha256": "r", "threshold": 9.0}), 9.0)
        with self.assertRaises(ValueError):
            verify_policy(FakeFixer, dict(pol, thresholds={"curated": -1.0, "edit": 2.0}))

    def test_applies(self):
        self.assertTrue(applies((5.0, "fixed", False, "curated"), SplitCut(4.0, 8.0)))
        self.assertFalse(applies((5.0, "fixed", False, "edit"), SplitCut(4.0, 8.0)))

    def test_split_finds_more_fixes_than_one_cutoff(self):
        # Curated fixes sit at margin 5, curated errors at 3; generic fixes at 9, generic harm at 6.
        rows = [doc(1, [(5.0, "fixed", "curated")]) for _ in range(20)]
        rows += [doc(1, [(9.0, "fixed", "edit")]) for _ in range(20)]
        rows += [doc(0, [(6.0, "harm", "edit")], 1000) for _ in range(5)]
        rows += [doc(0, [(3.0, "harm", "curated")], 1000) for _ in range(5)]
        single = pick(rows, 0.95, 0.5, 5)
        split = pick_split(rows, 0.95, 0.5, 5)
        self.assertEqual(single, 9.0)
        self.assertTrue(3.0 < split.curated <= 5.0 and 6.0 < split.edit <= 9.0)
        self.assertEqual(measure(rows, split)["fixed"], 40)
        self.assertEqual(measure(rows, single)["fixed"], 20)


if __name__ == "__main__":
    unittest.main()

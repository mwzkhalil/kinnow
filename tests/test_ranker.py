"""Regression tests for Urdu span safety, gradient flow, and artifact parity."""
import importlib.util
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import unittest
import numpy as np
from evaluate import measure, pick, verify_policy
from gold import edits_for
from pilot_train import init_weights, loss_grad
from prepare import BASE
from score_asr import distance, words
from urdu_core import BOS, PAD, Fixer, apply_edits, build_resources, contexts, forward, lookup, one_edit, read_json, sentences, suggestions
from collections import Counter

class Core(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = Fixer(BASE / 'tests/fixtures/tiny_model/model.npz', BASE / 'tests/fixtures/tiny_model/resources.json')
    def test_protected_text(self):
        text="🙂 https://example.pk/نظر \u200c #نظر @نذر `عصر` urduنظر نظر2 کِتاب می\u200cرا foo@example.pk عام"
        tokens = [t for s in sentences(text) for t in s]
        self.assertEqual([t.raw for t in tokens if t.editable], ['عام'])
        self.assertTrue(all((t.raw == text[t.start:t.end] for t in tokens)))
        self.assertEqual(self.fx.decisions('کراچی ۲۰۲۶ پاکستان'), [])
    def test_lookup_and_spans(self):
        self.assertEqual(lookup('كي'), 'کی')
        self.assertNotEqual(lookup('ی'), lookup('ے'))
        s = '🙂  اس کا عصر، آج بھی ہے۔'
        target = s.replace('عصر', 'اثر')
        edits = edits_for(s, target)
        self.assertEqual(edits[0]['start'], s.index('عصر'))
        self.assertEqual(apply_edits(s, [dict(edits[0], original='عصر')]), target)
        with self.assertRaises(ValueError):
            edits_for(s, target.replace('،', ','))
        with self.assertRaises(ValueError):
            apply_edits('الف', [dict(start=0, end=100, original='الف', replacement='ب')])
    def test_distances_and_graph(self):
        self.assertFalse(one_edit('اثر', 'عصر'))
        self.assertTrue(one_edit('اب', 'با'))
        self.assertTrue(one_edit('حال', 'محال'))
        self.assertFalse(one_edit('نظر', 'نظر'))
        self.assertFalse(one_edit('کتاب', 'کام'))
        counts = Counter({'سات': 8, 'ساتھ': 7, 'حال': 6, 'ہال': 5, 'نام': 4, 'کام': 3})
        r = build_resources(counts, [['سات', 'ساتھ'], ['حال', 'ہال']], {'نام'}, 16, 1)
        i = r['words'].index('سات')
        j = r['words'].index('ساتھ')
        p = r['words'].index('نام')
        self.assertEqual(r['candidates'][i][:2], [i, j])
        self.assertEqual(r['candidates'][p], [p])
        self.assertFalse(any((p in row[1:] for row in r['candidates'])))
    def test_context_boundaries(self):
        seq = np.array([5, 6, 7, 8, 9, 10])
        ends = np.array([3, 6])
        left, right = contexts(seq, ends, np.array([1, 3, 5]))
        self.assertEqual(left[0, -2:].tolist(), [BOS, 5])
        self.assertEqual(right[0].tolist(), [PAD, PAD, PAD, 7])
        self.assertEqual(left[1, -1], BOS)
        self.assertEqual(right[1].tolist(), [PAD, PAD, 10, 9])
        self.assertEqual(right[2].tolist(), [0, 0, 0, 0])
    def test_threshold_and_artifact_binding(self):
        fx = self.fx
        policy = read_json(BASE / 'tests/fixtures/tiny_model/policy.json')
        self.assertGreater(verify_policy(fx, policy), 0)
        with self.assertRaises(ValueError):
            verify_policy(fx, dict(policy, model_sha256='different'))
        self.assertEqual(suggestions([{'margin': -1.0}], 0.1), [])
        self.assertEqual(suggestions([{'margin': 8.0}], None), [])
        rows = [('x', 8, 0, 0, [(3.0, 'harm')]), ('x', 8, 1, 1, [(1.0, 'fixed')])]
        self.assertIsNone(pick(rows, 0.95, 0.5, 1))
        self.assertEqual(measure(rows, None)['suggestions'], 0)
    def test_lexical_counts(self):
        self.assertEqual(distance(words('آج AI اچھا ہے۔'), words('آج AI اچھا تھا!')), 1)
        self.assertEqual(distance([], ['لفظ']), 1)
    def test_gradient(self):
        rng = np.random.default_rng(15)
        weights = init_weights(16, 5, 4, 5)
        batch=(rng.integers(1,16,(4,16)),rng.integers(1,16,(4,4)),rng.integers(1,16,(4,9)),np.array([0,1,1,2]))
        _, grad = loss_grad(weights, batch)
        for key in ("proj","l_wi","r_wh","l_bh","emb"):
            at = np.unravel_index(np.abs(grad[key]).argmax(), grad[key].shape)
            orig = weights[key][at].copy()
            eps = 0.015
            weights[key][at] = orig + eps
            plus = loss_grad(weights, batch, False)
            weights[key][at] = orig - eps
            minus = loss_grad(weights, batch, False)
            weights[key][at] = orig
            self.assertAlmostEqual(float(grad[key][at]), (plus - minus) / (2 * eps), delta=0.00012)
    @unittest.skipUnless(importlib.util.find_spec("torch"),"requires PyTorch")
    def test_torch_portable_parity(self):
        import torch
        from train import Ranker
        torch.manual_seed(57)
        m = Ranker(25, 8, 9).eval()
        g = np.random.default_rng(12)
        batch = [g.integers(0, 25, (6, n)) for n in (16, 4, 9)]
        with torch.inference_mode():
            expected = m(*(torch.from_numpy(x) for x in batch)).numpy()
        np.testing.assert_allclose(forward(m.arrays(), *batch)[0], expected, atol=1e-05)
    @unittest.skipUnless(importlib.util.find_spec("onnxruntime"),"requires Python ONNX Runtime")
    def test_shipped_graphs(self):
        rng = np.random.default_rng(9)
        batch = [rng.integers(0, 1100, (8, n)) for n in (16, 4, 9)]
        expected = self.fx.score(*batch)
        for tag,tolerance in (("fp32",1e-4),("int8",.20)):
            model = Fixer(BASE / ('tests/fixtures/tiny_model/model.' + tag + '.onnx'), BASE / 'tests/fixtures/tiny_model/resources.json')
            self.assertGreater(verify_policy(model, read_json(BASE / ('tests/fixtures/tiny_model/' + tag + '/policy.json'))), 0)
            np.testing.assert_allclose(model.score(*batch), expected, rtol=0, atol=tolerance)
if __name__=="__main__":
    unittest.main()

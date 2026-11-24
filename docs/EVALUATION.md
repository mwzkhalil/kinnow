# Evaluation report: Kinnow 1.0.0

| | |
|---|---|
| INT8 model SHA-256 | `6afc16400429974d26b9ed8eb887ed0347db1a8d0a565a4054faf89b5280b37c` |
| FP32 model SHA-256 | `41d87336aef3a3f8dd310deb5a2a60fabba8fbbebb826a647c7745a7119ff339` |
| Checkpoint | step 99,000 of 100,000 (lowest dev loss, 0.280) |
| Default threshold (INT8 / FP32) | 8.705 / 8.706 |
| Balanced threshold (INT8 / FP32) | 7.328 / 7.328 |

All raw numbers are in [`eval/results/`](../eval/results), one folder per model format and policy. `tools/evaluate_release.sh` regenerates all of them.

## Rules followed

- Thresholds were chosen only on the synthetic **development** split and then frozen. No test set was used to pick anything, including the choice between single and split thresholds (section 6).
- Hand-written set A (`eval/handwritten_a.tsv`) was used during development to diagnose errors, which led to adding the grammatical paradigms to the confusion list. Its 12 lexical pairs outside the hand-picked list were deliberately left out of the confusion list.
- Hand-written set B (`eval/handwritten_b.tsv`) was written before the released model was trained and was not used for any decision.
- Every hand-written sentence was checked against the training corpus by exact normalized-sentence match. In set A, `c32` and `e18` occur verbatim in training text. Set B has no overlap.
- "Precision" is the share of the model's changes that were right. "Recall" is the share of wrong words it fixed.

## 1. Synthetic held-out Wikipedia

5,000 held-out sentences, each used once clean and once with one word swapped for one of its candidates (so swaps include the agreement paradigms).

| | INT8 default | INT8 balanced | FP32 default | FP32 balanced |
|---|---:|---:|---:|---:|
| Suggestions / correct | 2,106 / 1,993 | 2,838 / 2,573 | 2,103 / 1,991 | 2,839 / 2,577 |
| Precision (95% Wilson) | 94.6% (93.6–95.5) | 90.7% (89.5–91.7) | 94.7% (93.6–95.6) | 90.8% (89.7–91.8) |
| Recall | 39.9% | 51.5% | 39.8% | 51.6% |
| Right word in candidate list | 93.3% | 93.3% | 93.3% | 93.3% |
| Wrong changes on untouched sentences (91,978 words) | 29 (0.32 per 1,000) | 72 (0.78 per 1,000) | 29 (0.32 per 1,000) | 70 (0.76 per 1,000) |

## 2. Hand-written benchmarks

INT8 and FP32 give the same results on both sets with the default policy.

### Set A (`handwritten_a.tsv`): 62 errors, 46 correct controls

| | Default | Balanced |
|---|---:|---:|
| Curated word pairs fixed (32) | 9 | 12 |
| Everyday swaps fixed (30) | 2 (`سی→سے`, `کی→کہ`) | 4 |
| Precision | 11 / 11 | 16 / 16 |
| Correct controls changed | 0 / 46 | 0 / 46 |
| Right word not in candidate list | 1 | 1 |

18 of the 30 everyday errors are covered by the grammatical paradigms. Split by confusion-list membership (default policy), Kinnow fixed 11 / 50 listed errors and 0 / 12 unlisted lexical pairs.

### Set B (`handwritten_b.tsv`, written before training): 35 errors, 30 correct controls

| | Default | Balanced (INT8) |
|---|---:|---:|
| Agreement fixed (18) | 1 (`گئی→گئے`) | 1 |
| Function words fixed (6) | 2 (`کے→کہ`, `نہ→نے`) | 4 |
| Lexical fixed (11) | 1 (`سیر→سر`) | 1 |
| Wrong replacement | 1 (`محال→محاذ`, should be `محل`) | 1 |
| Correct controls changed | 0 / 30 | 0 / 30 |
| Agreement: right word not in candidate list | 0 | 0 |
| Agreement: right word ranked first (any margin) | 16 / 18 | 16 / 18 |
| Lexical: right word not in candidate list | 5 | 5 |

The model ranks the right agreement form first in 16 / 18 cases, with margins mostly between 3 and 7. Those are below both thresholds. Section 6 shows why lowering the threshold for these words is not supported by the development data. The five unreachable lexical pairs are two-letter edits outside the confusion list (`سیر/شیر`, `حل/ہل`, `پیاس/پیاز`).

Per-sentence traces with margins: `eval/results/<policy>/handwritten_trace.json` (set A) and `eval/results/<policy>/handwritten_b/handwritten_trace.json` (set B).

## 3. Clean out-of-domain text (FLEURS references)

All 299 human reference transcripts of Urdu FLEURS test (news-style read text, 230 distinct prompts; some prompts are recorded more than once), 8,288 eligible words. Every change was reviewed by hand.

| Policy | Changes | Fixed a typo in the reference | Wrong |
|---|---:|---:|---:|
| **Default** | 6 | 2 (`پھلے → پہلے`, `دینا → دنیا`) | 4 |
| Balanced (INT8) | 15 | 4 (adds `ہوتے یی → ہی` ×2) | 11 |

Default wrong changes: `بہت سی → سے` before `چھوٹے قد کی بلیاں` (one sentence, three recordings) and `ڈائنو سور → اور` (a split word). Balanced adds `دانوں → دانتوں` ×2, `لیں → میں` ×2, `پُر زور → اور` ×2 and `ہیں → میں`. FP32 balanced also makes `بچانے → بنانے`.

## 4. ASR transcripts (Zipformer, FLEURS test, 299 utterances, 230 prompt groups)

| Policy | Utterances changed | WER | 95% CI of WER change | CER |
|---|---:|---|---|---|
| **Default** | 24 | 28.43% → 28.35% | −0.17 to 0.00 | 13.078% → 13.081% |
| Balanced (INT8) | 58 | 28.43% → 28.35% | −0.21 to +0.05 | 13.078% → 13.100% |

No significant effect. Most ASR errors are outside the 20,000-word vocabulary or are split/merged words; see `tools/asr_audit/`.

## 5. Speed

Full suggestion path (tokenize, candidates, ONNX Runtime 1.28 CPU), one thread, AMD Ryzen 5 8500G, 299 sentences / 8,449 words, after 30 warm-up sentences.

| Format | File | Median / sentence | p95 / sentence |
|---|---:|---:|---:|
| INT8 | 1.93 MB | 3.0 ms | 5.2 ms |
| FP32 | 7.23 MB | 2.1 ms | 4.1 ms |

## 6. Separate thresholds for confusion-list pairs (tried, not used)

`src/evaluate.py calibrate --split-thresholds` chooses one cutoff for suggestions from the confusion list and another for generic one-edit suggestions, jointly, on dev. The hypothesis was that confusion-list pairs, like likely slips in the English tinyfix, deserve a lower bar.

The development data says the opposite. Confusion-list suggestions on dev (INT8), by cutoff:

| Cutoff | Fixed | Wrong replacement | Changed a correct word | Precision |
|---:|---:|---:|---:|---:|
| 10 | 173 | 1 | 4 | 97.2% |
| 9 | 215 | 3 | 15 | 92.3% |
| 8 | 260 | 6 | 30 | 87.8% |
| 7 | 311 | 7 | 59 | 82.5% |
| 6 | 353 | 12 | 111 | 74.2% |
| 5 | 395 | 14 | 226 | 62.2% |

Generic one-edit suggestions at the same cutoffs are more precise (93.0% at 8 vs 87.8%). Agreement and function words (`ہے/ہیں`, `کی/کے`, `سی/سے`) are frequent and often ambiguous within a 16 + 4 word window. Calibrated split cutoffs came out as list 9.26 / generic 8.52 (default) and list 7.71 / generic 7.16 (balanced). On test they gave recall 40.6% vs 39.9% and precision 94.2% vs 94.6%: no meaningful gain. The released policies therefore use one threshold. Results are in `eval/results/split-thresholds/`.

## 7. Recommended next steps

1. **Collect real errors.** A few hundred naturally occurring, editor-corrected Urdu errors (`src/gold.py` defines the format) would replace synthetic precision estimates and give a fair calibration set for agreement words.
2. **More context for agreement.** Subject–verb agreement often depends on words outside the window or on noun gender the model cannot see. A longer right context or a gender feature are the obvious experiments.
3. **Broader candidates.** Two-letter lexical pairs (`سیر/شیر`, `حل/ہل`, `پیاس/پیاز`) need a phonetic candidate generator or more curated groups.
4. **More varied text.** Wikipedia is encyclopaedic; add conversational and news Urdu.

## Reproduce

```bash
conda activate mahwiz
python -m unittest discover -s tests
python src/pipeline.py --work work/wiki --device cuda       # train (resumable)
tools/evaluate_release.sh work/wiki work/fleurs299.jsonl    # everything in this report
```

`work/fleurs299.jsonl` holds the FLEURS references and Zipformer hypotheses (`id`, `group`, `reference`, `hypothesis`); it is not redistributed.

Test environment: Python 3.11.15, NumPy 2.4.6, ONNX Runtime 1.28.0, PyTorch 2.13.0+cu130, Linux, RTX 4060.

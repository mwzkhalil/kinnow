---
language:
  - ur
license: mit
library_name: onnx
pipeline_tag: token-classification
tags:
  - urdu
  - spelling-correction
  - contextual-spellchecking
  - real-word-errors
  - onnx
  - on-device
  - kinnow
datasets:
  - wikimedia/wikipedia
---

<p align="center">
  <img src="kinnow.png" width="180" alt="Kinnow logo: a cartoon orange with big eyes and two buck teeth">
</p>

# Kinnow: Urdu autocorrect that reads both sides of a word

**params:** 1.8M · **size:** 1.9 MB model + 1.1 MB vocabulary · **speed:** ~3 ms per sentence on one CPU thread · **runs:** offline, on device · **data sent:** none

A spell checker can't tell that `اثر` was meant to be `عصر`: both are real Urdu words. This model can.

```
You type:  ہم اثر کی نماز کے بعد بازار گئے۔
You get:   ہم عصر کی نماز کے بعد بازار گئے۔
```

Kinnow reads up to 16 words before and 4 words after a correctly spelled word, and scores it against up to 8 nearby real words. It only suggests a swap when the best alternative wins by a clear margin, so it stays quiet far more often than it speaks.

Kinnow is named after the small citrus from Punjab. Its design follows [paperweight/tinyfix](https://huggingface.co/paperweight/tinyfix), an English model of the same kind; Kinnow was trained from scratch with a new Urdu vocabulary, tokenizer and confusion list.

## What it does

| You type | It becomes |
|---|---|
| ہم اثر کی نماز کے بعد بازار گئے۔ | ہم **عصر** کی نماز کے بعد بازار گئے۔ |
| اس نے بینک سے فرض لے کر گھر خریدا۔ | اس نے بینک سے **قرض** لے کر گھر خریدا۔ |
| میں اپنے دوستوں کے سات کرکٹ کھیلتا ہوں۔ | میں اپنے دوستوں کے **ساتھ** کرکٹ کھیلتا ہوں۔ |
| اس نے کہا کی وہ کل آئے گا۔ | اس نے کہا **کہ** وہ کل آئے گا۔ |
| میں نے لاہور سی کراچی تک سفر کیا۔ | میں نے لاہور **سے** کراچی تک سفر کیا۔ |
| ہم سب کل شادی میں گئی تھے۔ | ہم سب کل شادی میں **گئے** تھے۔ |
| اس کی باتوں کا مجھ پر گہرا اثر ہوا۔ | No change. `اثر` is right here. |
| اس کی بیٹی ڈاکٹر بن گیا ہے۔ | No change. It ranks `گئی` first, but not by enough to act (see *What it misses*). |

Every row is an actual output of this model with the default `policy.json`, taken from the hand-written test sets.

## How well it works

| | Default (`policy.json`) | Balanced (`policy.balanced.json`) |
|---|---:|---:|
| Fixes that were right, synthetic test | **94.6%** (1,993 / 2,106) | 90.7% |
| Wrong words fixed, synthetic test | **39.9%** | 51.5% |
| Wrong changes on clean text, synthetic test | **1 in 3,200 words** | 1 in 1,300 words |
| Hand-written set A: errors fixed (62) | 11, all right | 16, all right |
| Hand-written set B: errors fixed (35, written before training) | 4 of 5 suggestions right | 6 of 7 suggestions right |
| Hand-written: correct sentences changed | 0 / 76 | 0 / 76 |
| Clean news-style text (8,288 words), wrong changes after manual review | 4 | 11 |
| ASR transcripts: word error rate | 28.43% → 28.35% | 28.43% → 28.35% |

Both thresholds were picked on a development split and then frozen. No test set was used for tuning.

**Test sets**

- **Synthetic (9,998 sentences, held-out Urdu Wikipedia).** Half the sentences have one word swapped for one of its candidates, half are untouched. The largest test, but the errors are artificial.
- **Hand-written set A (108 sentences)** and **set B (65 sentences)**: one real-word mistake per error sentence, plus correct sentences that use the easily confused words. Set B was written before this model was trained, and covers gender/number agreement, function words and word mix-ups.
- **Clean out-of-domain text.** The 299 human reference transcripts of Urdu FLEURS test, every change reviewed by hand. Of the default policy's 6 changes, 2 corrected real typos in the reference (`پھلے → پہلے`, `دینا → دنیا`) and 4 were wrong (`بہت سی → سے` on one sentence that is recorded three times, and the split word `ڈائنو سور`).
- **ASR.** Zipformer transcripts of the same 299 recordings, scored before and after. The change (−0.08 points, 95% interval −0.17 to 0.00) is within noise.

Speed, measured end to end (tokenize, build candidates, run ONNX) on an AMD Ryzen 5 8500G, one thread, 299 sentences averaging 28 words: INT8 median 3.0 ms per sentence (p95 5.2 ms); FP32 median 2.1 ms (p95 4.1 ms).

## What it misses

- **It still misses most wrong words**: 60% on synthetic errors, and most hand-written ones.
- **Gender/number agreement is ranked but rarely fixed.** For 16 of 18 agreement errors in set B the right word now comes first, but by too small a margin. On the development data, accepting smaller margins for these words would wrongly change about 1 in 1,200 correct words, because forms like `ہے / ہیں` and `کی / کے` often can't be decided from the nearby words alone.
- **It only knows 20,000 words**, and only suggests words one letter away or in its confusion list. Mix-ups like `سیر / شیر` and `حل / ہل` are out of reach.
- **Split and joined words** (`کار آمد`, `ڈائنو سور`) are not handled and can cause a wrong change.
- **It needs the words that come after**, so a fix can only be made once up to four more words are typed.
- **Urdu (Perso-Arabic script) only.** Latin text, numbers, URLs, emails, hashtags and words with diacritics are never changed.

## How it works

1. The text is split into words. Arabic/Persian `ي/ك` are folded to Urdu `ی/ک` for lookup; `ی/ے`, `ہ/ھ` and `ن/ں` stay distinct. Original offsets and spellings are kept.
2. For each editable word in the vocabulary, up to 8 alternatives are taken from a precomputed list: the 72 confusion groups first, then words one edit away, ordered by frequency.
3. The model scores the typed word and each alternative against the 16 words before and 4 after.
4. A swap is suggested only if `score(best) − score(typed)` is at least the policy threshold (8.71 by default, 7.33 for balanced).

## How it was trained

**Task.** Take a sentence, pick a word, and about half the time replace it with one of its candidates. The model learns to score the original word highest. About a quarter of training positions are centred on a word from the confusion groups.

**Data.** Urdu Wikipedia (2023-11-01 snapshot, 200,154 articles). After sentence deduplication: 1.56 M training sentences (30.6 M tokens), split 90/5/5 by article so no article appears in two splits.

**Model.** Two GRUs, one reading left-to-right up to the word and one right-to-left back to it. Their states are joined, projected, and compared with each candidate's embedding.

| Setting | Value |
|---|---|
| Parameters | 1,807,424 |
| Vocabulary | 20,000 words, plus number and other-script tokens |
| Confusion groups | 72 (18 word pairs, 54 grammatical paradigms) |
| Embedding / hidden size | 64 / 256 |
| Context | 16 words before, 4 after |
| Candidates | typed word + up to 8 alternatives |
| Training | 100,000 steps, batch 2,048, AdamW, lr 0.002 with cosine decay to 0.0002 |
| Best checkpoint | step 99,000, dev loss 0.280 |
| Hardware / time | one NVIDIA RTX 4060 (8 GB, shared), about 25 minutes of training |
| Export | ONNX; INT8 weight-only quantization (top-1 agreement with FP32: 512 / 512) |

## Files

| File | What it is |
|---|---|
| `model.onnx` | INT8 model, 1.9 MB |
| `resources.json` | vocabulary and candidate lists (must match the model) |
| `policy.json` | default threshold (high precision) |
| `policy.balanced.json` | lower threshold (higher recall) |
| `fp32/` | FP32 model with its own policies, 7.2 MB, slightly faster on desktop CPUs |
| `inference/` | dependency-light Python reference code (`numpy`, `onnxruntime`) |
| `eval/` | hand-written benchmarks A and B, the confusion list and every result in this card |
| `kinnow.png` | logo |
| `manifest.json` | SHA-256 of every file |

## Using the model

```bash
pip install numpy onnxruntime
python inference/fix.py --model model.onnx --resources resources.json --policy policy.json \
  --text "ہم اثر کی نماز کے بعد بازار گئے۔" --apply
```

```json
{"suggestions": [{"start": 3, "end": 6, "original": "اثر", "replacement": "عصر", "margin": 11.12, "curated": true}],
 "text": "ہم عصر کی نماز کے بعد بازار گئے۔"}
```

Offsets count Unicode code points. For a JavaScript editor, convert them to UTF-16 offsets. To process many lines, pipe one JSON object per line (`{"text": "..."}`) on standard input. `curated` says whether the suggestion came from the confusion list.

From Python:

```python
import sys; sys.path.insert(0, "inference")
from urdu_core import Fixer, suggestions, apply_edits, read_json
from evaluate import verify_policy

fx = Fixer("model.onnx", "resources.json")
cut = verify_policy(fx, read_json("policy.json"))   # checks the policy belongs to this model
text = "اس نے بینک سے فرض لے کر گھر خریدا۔"
print(apply_edits(text, suggestions(fx.decisions(text), cut)))   # اس نے بینک سے قرض لے کر گھر خریدا۔
```

**ONNX interface.** Inputs `left` (int64, `[N, 16]`), `right` (int64, `[N, 4]`), `candidates` (int64, `[N, 9]`; column 0 is the typed word, 0 = padding). Output `scores` (float32, `[N, 9]`). Higher means a better fit.

## Intended use

Suggestions in Urdu writing tools, and experiments on finished Urdu ASR transcripts. With the default policy it suits underlined suggestions the user accepts. It has not been tested on naturally occurring human typing errors, so silent automatic correction is not recommended yet.

## License and attribution

Model and code: MIT. Trained on [Urdu Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (CC BY-SA 3.0 / GFDL) by its contributors. Architecture and training recipe follow [paperweight/tinyfix](https://huggingface.co/paperweight/tinyfix) (MIT).

Training code, tests and the evaluation tooling: [github.com/mwzkhalil/kinnow](https://github.com/mwzkhalil/kinnow).

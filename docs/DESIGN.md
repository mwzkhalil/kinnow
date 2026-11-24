# Design notes

## The task

Two real Urdu words can look or sound alike while fitting different contexts. `اثر` usually means an effect; in `نماز عصر`, `عصر` names the afternoon prayer. A dictionary accepts both spellings. Kinnow asks which of the two fits this occurrence, using the words around it.

The approach follows [paperweight/tinyfix](https://huggingface.co/paperweight/tinyfix), an English model with about 1.9 million parameters: two GRUs read 16 words to the left and four to the right, score the typed word and up to eight alternatives, and a separate decision rule chooses when a proposed correction is worth showing. Kinnow keeps that design, with a new Urdu tokenizer, vocabulary, candidate builder and training corpus. [Upstream implementation](https://github.com/cfahlgren1/tinyfix)

Urdu already has published work on contextual real-word spelling: Aziz et al. (IEEE Access, 2023) reported a 125,562-sentence corpus with word n-grams and Damerau–Levenshtein candidates. Kinnow's contribution is a small downloadable ranker, explicit measurement of damage to clean text, and an ASR before/after comparison. Its metrics are not directly comparable with that paper's, which use another distribution, error density and decision rule. [Paper and abstract](https://repositorio.unib.org/id/eprint/8800/)

## Data

| Source | Role | Notes |
|---|---|---|
| Urdu Wikipedia, 2023-11-01 | Training, dev and synthetic test | 200,154 articles, one 167,627,869-byte Parquet shard pinned by SHA-256. Encyclopaedic vocabulary and some editorial errors. [Pinned release](https://huggingface.co/datasets/wikimedia/wikipedia/commit/126488064a94417b849f69e1223923108896203e) |
| Hand-written benchmarks A and B | Independent test | Authored for this project; see `docs/EVALUATION.md`. |
| FLEURS Urdu test references and Zipformer transcripts | Clean out-of-domain text, ASR before/after | Not redistributed. |
| Maḵẖzan | Possible future training text | 6.26 million words of edited Urdu; its terms restrict redistribution of raw text. [Corpus and terms](https://github.com/zeerakahmed/makhzan/blob/master/README.md) |
| UrduGEC-Gold | Related work | 1,613 expert-checked pairs focused on grammar; purely orthographic edits were filtered out, so it does not test this task directly. [Dataset card](https://huggingface.co/datasets/Kagura-Ahad-123/UrduGEC-Gold) |

Stable article groups define 90/5/5 train/dev/test buckets by hashing. Normalized sentences are deduplicated globally, and the vocabulary is built from training text only. After filtering: 1,563,750 training sentences (30.6 M tokens), 78,578 dev and 87,012 test sentences. Article IDs and URLs are kept in a local provenance database, which the evaluator uses to check test sentences against training text.

## Tokenization

Offsets are Unicode code points in the original text; untouched characters keep their source spelling. For lookup, Arabic `ي/ك` fold to Urdu `ی/ک` and tatweel is removed, but `ی / ے`, `ہ / ھ` and `ن / ں` stay distinct. Latin text, mixed-script identifiers, URLs, emails, hashtags, numbers, words with diacritics and names in `data/protect.txt` are never edited. [Unicode normalization](https://www.unicode.org/reports/tr15/)

## Candidates

Each of the 20,000 vocabulary words gets up to 8 alternatives:

1. members of its groups in `data/confusions.tsv`, by frequency: 18 hand-picked word pairs (`اثر/عصر`, `نظر/نذر`, `سات/ساتھ`, ...) and 54 grammatical paradigms (`گیا/گئی/گئے/گئیں`, `ہے/ہیں/ہوں/ہو`, `کا/کی/کے`, adjective and possessive endings, ...);
2. then words one substitution, insertion, deletion or adjacent transposition away, by frequency.

The paradigms matter because many agreement forms differ by two characters and would otherwise never be offered. The 12 lexical pairs in hand-written set A that are not in the hand-picked list were left out on purpose, so set A still measures pairs the model was not taught.

## Model and training

| Setting | Value |
|---|---|
| Embedding / GRU hidden size | 64 / 256 |
| Parameters | 1,807,424 |
| Context | 16 words left, 4 right |
| Candidates | typed word + up to 8 |
| Optimizer | AdamW, lr 0.002, cosine decay to 0.0002 |
| Steps × batch | 100,000 × 2,048 (about 205 M examples) |
| Selection | lowest dev loss, sampled every 500 steps (step 99,000, dev loss 0.280) |

Each training example picks a position and, half the time, replaces the word with one whose candidate list contains it, so the model learns both to correct and to leave correct words alone. A quarter of positions are centred on confusion-group words.

Scoring uses the best candidate's score minus the typed word's score. A policy sets the minimum margin, chosen on dev for a precision target and a ceiling on wrong changes per 1,000 clean words, and is bound to the model and vocabulary by SHA-256.

## Export

The GRU is written as plain ONNX operations so that weights can be quantized. INT8 stores embeddings and matrices as signed 8-bit integers with per-channel scales; computation runs in floating point after dequantization. On 512 dev positions both exports pick the same top candidate as the NumPy reference (FP32 max score difference < 0.00001, INT8 < 0.15). INT8 saves download size, not time: on a desktop CPU, FP32 is faster (2.2 vs 3.0 ms per sentence). [ONNX GRU](https://onnx.ai/onnx/operators/onnx__GRU.html) · [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

## Test fixture

`tests/fixtures/tiny_model` is a 39,096-parameter model trained with the NumPy-only trainer (`src/pilot_train.py`) on Inkscape Urdu strings and the seed sentences. The unit tests use it for span safety, gradient and ONNX-parity checks. It is not a usable spellchecker.

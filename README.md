<p align="center">
  <img src="assets/kinnow.png" width="200" alt="Kinnow logo: a cartoon orange with big eyes and two buck teeth">
</p>

<h1 align="center">Kinnow</h1>

**Offline Urdu autocorrect for real-word errors.** Kinnow is a 1.8M-parameter model that reads the words on both sides of a correctly spelled Urdu word and suggests a better-fitting real word: `اثر → عصر` in `نماز اثر`, but not in `دوا کا اثر`. It is named after the kinnow, the small citrus from Punjab.

| | |
|---|---|
| Parameters | 1,807,424 |
| Download | 1.9 MB INT8 ONNX + 1.1 MB vocabulary |
| Speed | ~3 ms per sentence, one CPU thread |
| Precision / recall (synthetic test) | 94.6% / 39.9% |
| Wrong changes on clean text | ~1 in 3,200 words (synthetic), 4 in 8,288 (news text) |
| Model on Hugging Face | [`mahwizzzz/kinnow`](https://huggingface.co/mahwizzzz/kinnow) |
| Version | 1.0.0 ([changelog](CHANGELOG.md)) |

```bash
python src/fix.py --model model.onnx --resources resources.json --policy policy.json \
  --text "ہم اثر کی نماز کے بعد بازار گئے۔" --apply
# → ہم عصر کی نماز کے بعد بازار گئے۔
```

Full results, including what it gets wrong: **[docs/EVALUATION.md](docs/EVALUATION.md)**. Model card: **[hf/README.md](hf/README.md)**. Design notes: **[docs/DESIGN.md](docs/DESIGN.md)**.

## Repository layout

```
src/                 training, export, evaluation and inference code
  urdu_core.py       tokenizer, candidate lists, NumPy reference model, Fixer
  prepare.py         download + clean Urdu Wikipedia, build vocabulary and splits
  train.py           PyTorch trainer (GPU)
  export_onnx.py     FP32 + INT8 ONNX export with parity check
  make_eval.py       synthetic dev/test sets
  evaluate.py        calibrate a threshold on dev, score it on test
  pipeline.py        runs all of the above, resumable
  fix.py             command-line suggestions
  gold.py            convert human-reviewed pairs into evaluation rows
  score_asr.py       WER/CER before and after correction for ASR transcripts
  bundle.py          package a model + policy into a deployable folder
  pilot_*.py         NumPy-only trainer and data builder for the test fixture
tests/               unit tests; tests/fixtures/tiny_model is a 39K-parameter fixture, not a release model
eval/                hand-written benchmarks A and B, evaluation script, policies, results
data/                confusion groups, protected words, seed sentences
tools/
  evaluate_release.sh     export the best checkpoint and run every test in docs/EVALUATION.md
  make_hf_release.py      build the Hugging Face folder
  train_resumable.sh      training with automatic resume, for shared GPUs
  asr_audit/              diagnose ASR corrections; tune thresholds on ASR dev data
hf/README.md         Hugging Face model card
docs/                evaluation report and design notes
notices/             data sources and third-party licenses
assets/kinnow.png    logo
```

## Setup

Python 3.10–3.13. Install a CUDA build of PyTorch first if you want to train.

```bash
conda activate mahwiz          # or any env with PyTorch
pip install -r requirements.txt
python -m unittest discover -s tests
```

## Train

```bash
python src/pipeline.py --work work/wiki --device cuda
```

This downloads and verifies the pinned Urdu Wikipedia snapshot (168 MB), prepares the corpus, trains, exports FP32 and INT8 ONNX, calibrates a threshold on the dev split and scores the test split. The defaults are the released settings: 100,000 steps, batch 2,048, cosine learning-rate decay. Training takes about 25 minutes on an RTX 4060 (8 GB) and about 1.3 GB of GPU memory. Re-run the same command to resume after an interruption, or use `tools/train_resumable.sh`, which does that automatically.

Outputs under `work/wiki/`:

| Path | Contents |
|---|---|
| `data/manifest.json` | corpus, vocabulary and split sizes, file hashes |
| `model/model.npz` | best checkpoint by dev loss (the released model) |
| `model/history.json`, `model/run.json` | loss curve, best step, settings |
| `onnx/model.{fp32,int8}.onnx`, `onnx/parity.json` | exported models and agreement with the reference |
| `reports/synthetic/{fp32,int8}/{policy,test}.json` | calibrated threshold and test results |

A 20-step smoke test with the bundled seed sentences:

```bash
python src/pilot_data.py --out work/smoke-docs.jsonl
python src/pipeline.py --work work/smoke --local-only --extra-jsonl work/smoke-docs.jsonl \
  --device cuda --steps 20 --batch 32 --eval-sentences 30
```

## Evaluate

Everything in [docs/EVALUATION.md](docs/EVALUATION.md) is produced by one script:

```bash
tools/evaluate_release.sh work/wiki work/fleurs299.jsonl
```

It exports the best checkpoint, calibrates the default and balanced policies on dev, then runs the synthetic test, both hand-written benchmarks, clean out-of-domain text, ASR before/after, CPU latency and the split-threshold experiment. Individual steps:

```bash
W=work/wiki
python eval/run_eval.py --model $W/onnx/model.int8.onnx --resources $W/model/resources.json \
  --policy $W/reports/synthetic/int8/policy.json --data $W/data \
  --bench eval/handwritten_b.tsv --clean-jsonl work/fleurs299.jsonl --out /tmp/kinnow-eval
python src/evaluate.py calibrate --model $W/onnx/model.int8.onnx --resources $W/model/resources.json \
  --data $W/data --rows $W/synthetic-5000/dev.jsonl --precision 0.90 --false-per-1000 1.0 \
  --policy /tmp/int8-balanced.json --report /tmp/dev.json
```

Add `--split-thresholds` to `calibrate` for separate cutoffs on confusion-list pairs and generic one-letter edits. It did not help for this model (see [docs/EVALUATION.md](docs/EVALUATION.md#6-separate-thresholds-for-confusion-list-pairs-tried-not-used)), so the released policies use one cutoff.

### Human-reviewed data

Put reviewed pairs in JSONL (`id`, `group`, `split` = `dev`/`test`, `domain`, `typed`, `corrected`; same text in both for a clean control; see `data/example_pairs.jsonl`), then:

```bash
python src/gold.py --pairs my_pairs.jsonl --out work/gold --reviewed
python src/pipeline.py --work work/wiki --device cuda \
  --human-dev work/gold/dev.jsonl --human-test work/gold/test.jsonl
```

### ASR transcripts

```bash
python src/score_asr.py --rows heldout_zipformer.jsonl --model release/hf/model.onnx \
  --resources release/hf/resources.json --policy release/hf/policy.json --out work/asr-before-after.json
```

Rows are `{"id", "group", "reference", "hypothesis"}`. See `tools/asr_audit/README.md` for per-position diagnostics.

## Release

```bash
python tools/make_hf_release.py --work work/wiki --out release/hf
hf upload mahwizzzz/kinnow release/hf . --repo-type model
```

Every policy file stores the SHA-256 of the model and vocabulary it was calibrated for, and loading a mismatched pair fails.

## How it works

Candidates come from 72 confusion groups (`data/confusions.tsv`: 18 word pairs and 54 gender/number and function-word paradigms) and from single-character edits, ranked by frequency, up to 8 per word. Two GRUs read 16 words left and 4 words right of the target; their joined state is compared with each candidate's embedding. A swap is suggested only if the best candidate beats the typed word by the policy margin. See [docs/DESIGN.md](docs/DESIGN.md).

## Acknowledgements and license

Code and model: [MIT](LICENSE). Kinnow's architecture and training recipe follow [paperweight/tinyfix](https://huggingface.co/paperweight/tinyfix) (MIT), an English model of the same kind. Training text: Urdu Wikipedia contributors (CC BY-SA 3.0 / GFDL). The test fixture was trained on the Inkscape Urdu translation catalogue (GPL-2.0-or-later). Details in [notices/SOURCES.md](notices/SOURCES.md).

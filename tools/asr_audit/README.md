# Kinnow ASR audit

Run these scripts from the Kinnow repository root in the training environment. The first script explains a trained model's decisions on saved transcripts. The second compares cutoffs on separate recordings. Both write diagnostic reports.

## 1. Inspect the 299 test transcripts

Build the release folder first (`python tools/make_hf_release.py --work work/wiki --out release/hf`), then run

```bash
python tools/asr_audit/rank.py --example برس برف
```

The command reads `work/fleurs299.jsonl` and `release/hf/{model.onnx,resources.json,policy.json}` (change with `--rows` and `--bundle`). It checks that the policy belongs to the model, runs inference on each original transcript and writes `work/wiki/fleurs299-rank-audit.json`. The console prints the counts and each occurrence of the requested pair.

Each position has the same unambiguous left and right neighbor pair in both texts, and its two normalized middle words differ by one character insertion, deletion or substitution. Word indices start at zero. The cohort includes spelling variants such as `لیے` and `لئے`. It excludes word splits and merges such as `کیا بن` versus `کیبن`. Report its percentages as coverage of this cohort.

| Outcome for a shortlisted reference | Interpretation |
| --- | --- |
| `typed_word_won_or_tied` | The recognizer's original word matched or exceeded every alternative's score. |
| `different_replacement_ranked_first` | Another alternative outranked the reference or won the tie between alternatives. |
| `reference_first_below_cutoff` | The reference won, with a positive margin smaller than the deployed cutoff. |
| `reference_applied` | Kinnow applied the reference under the deployed policy. |
| `reference_first_policy_abstains` | The reference won and the policy is configured to keep every original word. |
| `runtime_skipped_or_unmapped` | A resource entry exists, but this position had no reliably matched runtime score. Inspect the original UTF-8 span in context. |

`reference_rank_including_typed` starts at one and includes the original word. Both reported margins subtract the original word's score. The deployed alternative must win by a positive margin and meet the cutoff. For words outside the vocabulary, the report identifies which side is missing. For words in the vocabulary but off the shortlist, both IDs exist and the candidate graph lacks that directed edge.

## 2. Compare cutoffs on separate recordings

If the reference often wins but misses the cutoff, run this experiment in the same project

```bash
python tools/asr_audit/dev.py
```

The first run uses your local Urdu-Streaming-ASR checkout and released INT8 encoder, decoder, joiner and matching tokens. It fetches the pinned Urdu FLEURS validation file, verifies its SHA-256, and decodes its 267 recordings. Audio decoding uses PyArrow, SoundFile and the sherpa-onnx version from your Zipformer benchmark. The defaults reproduce the benchmark settings of 16 kHz audio, 320 ms chunks, modified beam search with four active paths, a blank penalty of 0.25 and one CPU thread. [Urdu FLEURS validation data](https://huggingface.co/datasets/google/fleurs/tree/d010211ed2d3d9281bcfed40dfdc5f57aceec472/ur_pk) · [Urdu-Streaming-ASR](https://github.com/mwzkhalil/Urdu-Streaming-ASR)

The command looks for a sibling `Urdu-Streaming-ASR` or `stream-asr` directory and for `~/PROXIMA/MAHWIZ/stream-asr`. With a different layout, substitute the actual paths in this example

```bash
python tools/asr_audit/dev.py \
  --asr-repo /path/to/Urdu-Streaming-ASR \
  --onnx-dir /path/to/zipformer/release/onnx
```

Use `--tokens /path/to/tokens.txt` if the Zipformer tokens are stored separately. Use `--parquet /path/to/validation-00000-of-00001.parquet` for an existing copy of the pinned shard. The generated hypotheses and decoder provenance are saved under `work/asr-dev/fleurs-validation/`. For a locally prepared, independent set of final transcripts, pass `--rows /path/to/development.jsonl`. Each line must have unique recording `id`, a prompt or speaker/session `group`, human `reference` and raw ASR `hypothesis`, all strings. An empty hypothesis counts as a recognition failure.

Development groups sharing an ID, prompt group, normalized reference or available audio digest with the frozen test are excluded. A deterministic split allocates about two thirds of the remaining groups to calibration and the remainder to confirmation. It also removes identified overlaps between those partitions. The report records the resulting counts. When supplying your own rows, use the same grouping convention and record IDs as the frozen evaluation wherever they identify the same material.

`work/asr-dev/threshold-experiment.json` reports full-transcript WER and CER using the project's existing normalizer. On calibration the script compares a fixed grid of margin cutoffs, the shipped cutoff and keeping the raw transcript. A cutoff qualifies if it reduces pooled word errors, does not increase character errors and makes at most 0.5 edits per 1,000 eligible words in reference-text controls. Ties favor fewer word errors, then fewer character errors, then fewer reference edits, then the higher cutoff. A selected threshold of `null` means that the original transcript is retained. This rule is for this experiment.

On confirmation, compare the selected cutoff with both raw Zipformer and the shipped policy. The report includes paired 95% bootstrap intervals by prompt group. Reference controls measure edits to human transcripts. Human review determines whether a changed spelling is harmful. A production release still needs evaluation on fresh target-domain speech and review of both proposed and missed edits.

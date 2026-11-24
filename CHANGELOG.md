# Changelog

## [1.0.0] - 2026-11-24

First release of Kinnow.

- 1.8M-parameter Urdu real-word ranker trained on Urdu Wikipedia (2023-11-01): 100,000 steps, batch 2,048, cosine learning-rate decay, best checkpoint at step 99,000.
- 72 confusion groups: 18 word pairs (`اثر/عصر`, `سات/ساتھ`, ...) and 54 gender/number and function-word paradigms (`گیا/گئی/گئے`, `ہے/ہیں`, `کا/کی/کے`, ...).
- INT8 (1.9 MB) and FP32 (7.2 MB) ONNX exports with parity checks.
- Two dev-calibrated policies: default (precision ≥ 95%) and balanced (precision ≥ 90%).
- Two hand-written Urdu benchmarks (97 errors, 76 correct controls) and `eval/run_eval.py`.
- Evaluation: synthetic test, hand-written sets, clean FLEURS references, Zipformer ASR before/after, CPU latency, split-threshold experiment. See `docs/EVALUATION.md`.
- Optional split thresholds (`evaluate.py calibrate --split-thresholds`), evaluated and not used by the released policies.

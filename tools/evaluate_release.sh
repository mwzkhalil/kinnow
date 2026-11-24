#!/bin/bash
# Export the best checkpoint, calibrate both policies on dev, and run every test reported in docs/EVALUATION.md.
#   tools/evaluate_release.sh [work/wiki] [work/fleurs299.jsonl]
# Expects a finished training run in $W (src/pipeline.py). Writes $W/onnx, $W/reports, eval/policies and eval/results.
set -euo pipefail
cd "$(dirname "$0")/.."
W=${1:-work/wiki}; F=${2:-work/fleurs299.jsonl}; R=$W/model/resources.json
DEV=$W/synthetic-5000/dev.jsonl; TEST=$W/synthetic-5000/test.jsonl

echo "== export best checkpoint ($W/model/model.npz)"
python src/export_onnx.py --data $W/data --model $W/model/model.npz --resources $R --out $W/onnx

rm -rf eval/results; mkdir -p eval/results eval/policies $W/split
for fmt in int8 fp32; do
  M=$W/onnx/model.$fmt.onnx; D=$W/reports/synthetic/$fmt; mkdir -p $D
  echo "== calibrate $fmt on dev"
  python src/evaluate.py calibrate --model $M --resources $R --data $W/data --rows $DEV --policy $D/policy.json --report $D/dev.json > /dev/null
  python src/evaluate.py test --model $M --resources $R --data $W/data --rows $TEST --policy $D/policy.json --report $D/test.json > /dev/null
  python src/evaluate.py calibrate --model $M --resources $R --data $W/data --rows $DEV --precision 0.90 --false-per-1000 1.0 \
    --policy eval/policies/$fmt-balanced.json --report $W/reports/synthetic/$fmt-balanced-dev.json > /dev/null
  for pol in "$D/policy.json $fmt" "eval/policies/$fmt-balanced.json $fmt-balanced"; do
    set -- $pol; O=eval/results/$2
    echo "== test $2"
    python eval/run_eval.py --model $M --resources $R --policy $1 --data $W/data --clean-jsonl $F --out $O > /dev/null
    python eval/run_eval.py --model $M --resources $R --policy $1 --data $W/data --bench eval/handwritten_b.tsv --out $O/handwritten_b > /dev/null
    python src/evaluate.py test --model $M --resources $R --data $W/data --rows $TEST --policy $1 --report $O/synthetic_test.json > /dev/null
    python src/score_asr.py --rows $F --model $M --resources $R --policy $1 --out $O/asr_fleurs_before_after.json > /dev/null
  done
  echo "== split-threshold experiment $fmt"
  mkdir -p eval/results/split-thresholds
  for P in "0.95 0.5 default" "0.90 1.0 balanced"; do
    set -- $P; S=$W/split/$fmt-$3
    python src/evaluate.py calibrate --split-thresholds --model $M --resources $R --data $W/data --rows $DEV --precision $1 --false-per-1000 $2 --policy $S-policy.json --report $S-dev.json > /dev/null
    python src/evaluate.py test --model $M --resources $R --data $W/data --rows $TEST --policy $S-policy.json --report eval/results/split-thresholds/$fmt-$3-test.json > /dev/null
    python -c "import json,sys; p=json.load(open(sys.argv[1])); json.dump({k: p.get(k) for k in ('threshold','thresholds','minimum_precision','max_false_per_1000','model_sha256')}, open(sys.argv[2],'w'), indent=1)" $S-policy.json eval/results/split-thresholds/$fmt-$3-policy.json
  done
done
echo "== done"

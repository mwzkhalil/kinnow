#!/bin/bash
# Run the training pipeline and resume it after transient failures (e.g. CUDA OOM on the shared GPU).
cd "$(dirname "$0")/.."
export PATH=~/miniconda3/envs/mahwiz/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for i in $(seq 1 20); do
  echo "=== attempt $i $(date)"
  python src/pipeline.py --work work/wiki --device cuda "$@" && { echo "=== DONE $(date)"; exit 0; }
  sleep 60
done
echo "=== GAVE UP $(date)"; exit 1

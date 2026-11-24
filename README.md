<p align="center">
  <img src="assets/kinnow.png" width="200" alt="Kinnow logo: a cartoon orange with big eyes and two buck teeth">
</p>

<h1 align="center">Kinnow</h1>

A 1.8M-parameter model that reads both sides of a correctly spelled word and swaps it only when a better real word clearly fits: `اثر → عصر` in `نماز اثر`, untouched in `دوا کا اثر`.

| | |
|---|---|
| Size | 1.9 MB INT8 ONNX + 1.1 MB vocab |
| Speed | ~3 ms/sentence, one CPU thread |
| Precision / recall | 94.6% / 39.9% (synthetic) |
| False edits | ~1 in 3,200 words |
| Model | [`mahwizzzz/kinnow`](https://huggingface.co/mahwizzzz/kinnow) · v1.0.0 |

```bash
pip install -r requirements.txt
python src/fix.py --model model.onnx --resources resources.json --policy policy.json \
  --text "ہم اثر کی نماز کے بعد بازار گئے۔" --apply
```

## Train and evaluate
```bash
python src/pipeline.py --work work/wiki --device cuda   # ~25 min on an RTX 4060
tools/evaluate_release.sh work/wiki work/fleurs299.jsonl
```

## How it works
72 confusion groups plus one-edit neighbors give up to 8 candidates. Two GRUs read 16 words left and 4 right, then a calibrated margin decides whether to swap.

[Evaluation](docs/EVALUATION.md) · [Design](docs/DESIGN.md) · [Model card](hf/README.md)

MIT · Follows [paperweight/tinyfix](https://huggingface.co/paperweight/tinyfix) · Trained on Urdu Wikipedia (CC BY-SA 3.0)

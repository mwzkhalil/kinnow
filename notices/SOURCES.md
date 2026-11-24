# Sources and attribution

Kinnow's architecture and training recipe follow the public description of [paperweight/tinyfix](https://huggingface.co/paperweight/tinyfix), an English real-word autocorrect model, and [its author's implementation](https://github.com/cfahlgren1/tinyfix). Both carry the MIT license. Kinnow's Urdu tokenizer, candidate builder, trainers, evaluation tools and ONNX graphs were written for this project, and its weights were trained from scratch.

The model is trained on the 2023-11-01 Urdu Wikipedia snapshot compiled by Wikimedia from articles by Urdu Wikipedia contributors. The dataset card lists CC BY-SA 3.0 and GFDL. The preparation script keeps source article IDs and URLs in the local provenance database. Keep that attribution and the relevant notices with any source text or derived corpus you redistribute. [Pinned snapshot, file hash, count and license labels](https://huggingface.co/datasets/wikimedia/wikipedia/commit/126488064a94417b849f69e1223923108896203e) · [Dataset card](https://huggingface.co/datasets/wikimedia/wikipedia)

The small test fixture in `tests/fixtures/tiny_model` was trained on the Inkscape Urdu translation catalogue installed with the Linux package (translation team `gist`, revision dated 2013-05-15, SHA-256 of the compiled input `237ef2b6920827e35f768ffd6b4655999d57b2125f2d47d1572adbf313ad5a6d`). Inkscape lists the translation license as GPL-2.0-or-later. [Urdu translation project and license](https://translate.inkscape.org/projects/inkscape/1-4-x/inkscape/ur/) · [GPL v2 text](Inkscape-GPL-2.txt)

The confusion groups, example annotations, seed sentences and both hand-written benchmarks were authored for this project. Evaluation policies and numerical reports apply to the model identified by their SHA-256.

Clean out-of-domain and ASR evaluation used the Urdu test split of [Google FLEURS](https://huggingface.co/datasets/google/fleurs) and transcripts from [Urdu-Streaming-ASR](https://github.com/mwzkhalil/Urdu-Streaming-ASR). Neither is redistributed here.

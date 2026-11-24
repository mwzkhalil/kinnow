"""Assemble the Hugging Face model repository folder from a finished training run.

    python tools/make_hf_release.py --work work/wiki --out release/hf

The output folder is self-contained: model, vocabulary, policies, inference code,
benchmark, results, model card and a SHA-256 manifest. Upload it with
    hf upload <user>/kinnow release/hf . --repo-type model
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from urdu_core import read_json, sha, write_json  # noqa: E402

POLICY_KEYS = ("model_sha256", "resources_sha256", "dev_sha256", "threshold", "thresholds", "minimum_precision", "max_false_per_1000")


def compact(policy, model_path, resources_path, name):
    p = read_json(policy)
    if p["model_sha256"] != sha(model_path) or p["resources_sha256"] != sha(resources_path):
        raise SystemExit(f"{policy} does not belong to {model_path}")
    return {**{k: p[k] for k in POLICY_KEYS if k in p}, "name": name}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True, help="training run directory, e.g. work/wiki")
    ap.add_argument("--out", type=Path, default=ROOT / "release/hf")
    ap.add_argument("--balanced-int8", type=Path, default=ROOT / "eval/policies/int8-balanced.json")
    ap.add_argument("--balanced-fp32", type=Path, default=ROOT / "eval/policies/fp32-balanced.json")
    ap.add_argument("--results", type=Path, default=ROOT / "eval/results", help="results folder shipped with the model")
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; remove it first")
    w, o = a.work, a.out
    res = w / "model/resources.json"
    (o / "fp32").mkdir(parents=True)
    (o / "inference").mkdir()
    (o / "eval").mkdir()

    int8, fp32 = w / "onnx/model.int8.onnx", w / "onnx/model.fp32.onnx"
    shutil.copy2(int8, o / "model.onnx")
    shutil.copy2(fp32, o / "fp32/model.onnx")
    shutil.copy2(res, o / "resources.json")
    write_json(o / "policy.json", compact(w / "reports/synthetic/int8/policy.json", int8, res, "default"))
    write_json(o / "policy.balanced.json", compact(a.balanced_int8, int8, res, "balanced"))
    write_json(o / "fp32/policy.json", compact(w / "reports/synthetic/fp32/policy.json", fp32, res, "default"))
    write_json(o / "fp32/policy.balanced.json", compact(a.balanced_fp32, fp32, res, "balanced"))

    for f in ("urdu_core.py", "fix.py", "evaluate.py"):
        shutil.copy2(ROOT / "src" / f, o / "inference" / f)
    for bench in ("handwritten_a.tsv", "handwritten_b.tsv"):
        shutil.copy2(ROOT / "eval" / bench, o / "eval" / bench)
    shutil.copy2(ROOT / "data/confusions.tsv", o / "eval/confusions.tsv")
    shutil.copytree(a.results, o / "eval/results")
    shutil.copy2(ROOT / "hf/README.md", o / "README.md")
    shutil.copy2(ROOT / "LICENSE", o / "LICENSE")
    shutil.copy2(ROOT / "assets/kinnow.png", o / "kinnow.png")

    files = sorted(p for p in o.rglob("*") if p.is_file())
    write_json(o / "manifest.json", {"project": "kinnow", "version": read_version(),
                                     "files": {str(p.relative_to(o)): sha(p) for p in files}})
    print(f"wrote {len(files) + 1} files to {o}")


def read_version():
    for line in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("## ["):
            return line[4:line.index("]")]
    return "unknown"


if __name__ == "__main__":
    main()

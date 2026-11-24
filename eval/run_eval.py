"""Independent evaluation of a Kinnow model and policy.

Runs three checks against a frozen model + policy (the threshold is never re-tuned here):

1. hand-written benchmark (eval/handwritten_a.tsv by default; --bench eval/handwritten_b.tsv for set B): authored sentences with one real-word
   error each, plus clean controls that use the easily-confused words correctly.
2. out-of-domain clean text: human FLEURS reference transcripts (read news-style speech,
   not Wikipedia). Every suggestion on these is counted as a false edit.
3. CPU latency of the full suggestion path (tokenize + shortlist + ONNX scoring), single thread.

Example:
    python eval/run_eval.py --model work/wiki/onnx/model.int8.onnx \
        --resources work/wiki/model/resources.json \
        --policy work/wiki/reports/synthetic/int8/policy.json \
        --data work/wiki/data --clean-jsonl work/fleurs299.jsonl --out eval/results/int8
"""
import argparse
import json
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluate import inspect, measure, verify_policy  # noqa: E402
from gold import edits_for  # noqa: E402
from urdu_core import Fixer, lookup, read_groups, read_json, sha, suggestions, write_json  # noqa: E402


def load_bench(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        ident, category, typed, corrected = line.split("\t")
        rows.append({"id": ident, "category": category, "typed": typed, "corrected": corrected})
    return rows


def to_eval_rows(bench, domain_of):
    out = []
    for r in bench:
        edits = [] if r["typed"] == r["corrected"] else edits_for(r["typed"], r["corrected"])
        out.append({"id": r["id"], "group": "handwritten-" + r["id"], "split": "test", "kind": "example",
                    "domain": domain_of(r), "text": r["typed"],
                    "edits": [{"start": e["start"], "end": e["end"], "replacement": e["replacement"]} for e in edits]})
    return out


def run_rows(fx, rows, cut, corpus):
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", encoding="utf-8", delete=False) as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        path = f.name
    docs, meta = inspect(fx, path, "test", corpus)
    Path(path).unlink()
    by = defaultdict(list)
    for d in docs:
        by[d[0]].append(d)
    return {"overall": measure(docs, cut), "by_domain": {k: measure(v, cut) for k, v in sorted(by.items())},
            "training_overlap_rows": meta["train_overlap_rows"], "training_overlap_checked": meta["training_provenance_checked"]}


def explain(fx, r, cut):
    """Per-sentence trace: what the model proposed, its margin, and the outcome."""
    gold = {}
    if r["typed"] != r["corrected"]:
        for e in edits_for(r["typed"], r["corrected"]):
            gold[(e["start"], e["end"])] = e["replacement"]
    decisions = fx.decisions(r["typed"])
    applied = suggestions(decisions, cut)
    out = {"id": r["id"], "category": r["category"], "typed": r["typed"], "expected": r["corrected"], "suggestions": [
        {"original": e["original"], "replacement": e["replacement"], "margin": round(e["margin"], 3)} for e in applied]}
    for (s, t), want in gold.items():
        word = r["typed"][s:t]
        observed = fx.ids.get(lookup(word), 0)
        target = fx.ids.get(lookup(want), 0)
        out["error_word"], out["expected_word"] = word, want
        out["on_shortlist"] = bool(observed and target and target in fx.graph[observed, 1:])
        best = next((d for d in decisions if d["start"] == s), None)
        out["best_alternative"] = best and {"word": best["replacement"], "margin": round(best["margin"], 3)}
    hit = {(e["original"], e["replacement"]) for e in applied}
    if gold:
        ok = any(e["replacement"] == gold.get((e["start"], e["end"])) for e in applied)
        out["outcome"] = "fixed" if ok else ("wrong_fix" if applied else ("missed_below_threshold" if out.get("on_shortlist") else "missed_not_on_shortlist"))
    else:
        out["outcome"] = "false_edit" if hit else "left_alone"
    return out


def latency(fx, texts, cut, warmup=30):
    for t in texts[:warmup]:
        suggestions(fx.decisions(t), cut)
    per_sentence, words = [], 0
    for t in texts:
        a = time.perf_counter()
        suggestions(fx.decisions(t), cut)
        per_sentence.append((time.perf_counter() - a) * 1000)
        words += len(t.split())
    q = statistics.quantiles(per_sentence, n=20)
    return {"sentences": len(texts), "words": words, "median_ms_per_sentence": statistics.median(per_sentence),
            "p95_ms_per_sentence": q[18], "mean_ms_per_word": sum(per_sentence) / words, "threads": 1}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--data", type=Path, help="training corpus dir, enables the train-overlap check")
    p.add_argument("--bench", type=Path, default=ROOT / "eval/handwritten_a.tsv")
    p.add_argument("--clean-jsonl", type=Path, help="JSONL with a 'reference' field: clean out-of-domain text")
    p.add_argument("--groups", type=Path, default=ROOT / "data/confusions.tsv", help="confusion list the model was trained with")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    fx = Fixer(a.model, a.resources)
    cut = verify_policy(fx, read_json(a.policy))
    a.out.mkdir(parents=True, exist_ok=True)
    report = {"model_sha256": fx.fingerprint, "resources_sha256": fx.asset, "threshold": cut}

    bench = load_bench(a.bench)
    hw = run_rows(fx, to_eval_rows(bench, lambda r: r["category"]), cut, a.data)
    # Same rows split by whether the error pair is in the training confusion list (taught) or not.
    pairs = set()
    for group in read_groups(a.groups):
        pairs.update((x, y) for x in group for y in group if x != y)

    def listed(r):
        if r["typed"] == r["corrected"]:
            return "clean"
        e = edits_for(r["typed"], r["corrected"])[0]
        return "in-list" if (lookup(r["typed"][e["start"]:e["end"]]), lookup(e["replacement"])) in pairs else "not-in-list"
    hw["by_confusion_list"] = run_rows(fx, to_eval_rows(bench, listed), cut, None)["by_domain"]
    hw["confusion_list_sha256"] = sha(a.groups)
    trace = [explain(fx, r, cut) for r in bench]
    hw["outcomes"] = {}
    for t in trace:
        hw["outcomes"].setdefault(t["category"], {}).setdefault(t["outcome"], 0)
        hw["outcomes"][t["category"]][t["outcome"]] += 1
    report["handwritten"] = hw
    write_json(a.out / "handwritten_trace.json", trace)

    if a.clean_jsonl:
        refs = [json.loads(l) for l in a.clean_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
        clean = [{"id": r["id"], "category": "fleurs-reference", "typed": r["reference"], "corrected": r["reference"]} for r in refs]
        report["clean_out_of_domain"] = run_rows(fx, to_eval_rows(clean, lambda r: "fleurs-reference"), cut, a.data)
        report["clean_out_of_domain"]["false_edits"] = [
            {"id": r["id"], **{k: e[k] for k in ("original", "replacement")}, "margin": round(e["margin"], 3)}
            for r in clean for e in suggestions(fx.decisions(r["typed"]), cut)]
        report["latency"] = latency(fx, [r["typed"] for r in clean], cut)

    write_json(a.out / "summary.json", report)
    o = hw["overall"]
    print(json.dumps({"threshold": cut, "handwritten": {k: o[k] for k in ("gold_errors", "suggestions", "fixed", "precision", "recall", "candidate_recall", "clean_control_false_edits")},
                      "handwritten_by_category": {d: {k: m[k] for k in ("gold_errors", "fixed", "suggestions", "recall", "precision")} for d, m in hw["by_domain"].items()},
                      "by_confusion_list": {d: {k: m[k] for k in ("gold_errors", "fixed", "suggestions", "edits_to_correct_words")} for d, m in hw["by_confusion_list"].items()},
                      "outcomes": hw["outcomes"], "overlap": hw["training_overlap_rows"],
                      "clean_ood": report.get("clean_out_of_domain", {}).get("overall", {}).get("false_edits_per_1000_clean"),
                      "clean_ood_words": report.get("clean_out_of_domain", {}).get("overall", {}).get("clean_control_words"),
                      "latency": report.get("latency")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

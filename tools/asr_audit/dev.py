"""Compare frozen Kinnow thresholds using separate ASR validation prompts."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

from common import ROOT, load_bundle, read_rows
from zipformer_assets import validation_rows
from score_asr import distance, words
from urdu_core import apply_edits, lookup, sentences, sha, span_key, suggestions, write_json

WORD_REF, WORD_ERR, CHAR_REF, CHAR_ERR, CONTROL_WORDS, CONTROL_EDITS, HYP_EDITS, CHANGED = range(8)
GRID = (0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0)


def signatures(row):
    reference = " ".join(words(row["reference"]))
    if not reference:
        raise ValueError(f"No normalized reference words in {row['id']}")
    keys = {("id", row["id"]), ("group", row["group"]), ("ref", reference)}
    if row.get("audio_sha256"):
        keys.add(("audio", row["audio_sha256"]))
    return keys


def partition(dev, frozen):
    """Split by prompt, excluding groups sharing identifiable evidence across sets."""
    forbidden = set().union(*(signatures(r) for r in frozen))
    groups = defaultdict(list)
    for row in dev:
        groups[row["group"]].append(row)
    fingerprints = {
        group: set().union(*(signatures(row) for row in rows))
        for group, rows in groups.items()
    }
    available = [g for g in groups if not (forbidden & fingerprints[g])]
    # Historical salt; changing it reshuffles the calibration/confirmation split.
    available.sort(key=lambda g: hashlib.sha256(("tinyfix-asr-dev-v1\0" + g).encode()).digest())
    if len(available) < 2:
        raise ValueError("Need independent development recordings with at least two groups after test exclusions.")
    n = 2 * len(available) // 3
    tune_groups = available[:n]
    used = set().union(*(fingerprints[g] for g in tune_groups))
    confirm_groups = [g for g in available[n:] if not (used & fingerprints[g])]
    if not confirm_groups:
        raise ValueError("No independent confirmation groups after overlap removal.")
    tune = [r for g in tune_groups for r in groups[g]]
    confirm = [r for g in confirm_groups for r in groups[g]]
    details = {
        "input_recordings": len(dev),
        "excluded_for_test_overlap": sum(len(groups[g]) for g in groups.keys() - set(available)),
        "excluded_for_calibration_overlap": sum(len(groups[g]) for g in set(available[n:]) - set(confirm_groups)),
        "calibration_recordings": len(tune),
        "calibration_groups": len(tune_groups),
        "confirmation_recordings": len(confirm),
        "confirmation_groups": len(confirm_groups),
    }
    return tune, confirm, details


def compute(fixer, rows, cutoffs):
    """Score the original hypothesis and an untouched reference control per recording."""
    by_cut = {t: defaultdict(lambda: np.zeros(8, dtype=np.int64)) for t in cutoffs}
    for row in rows:
        ref = row["reference"]
        hyp = row["hypothesis"]
        rwords = words(ref)
        rchars = "".join(rwords)
        eligible = sum(span_key(lookup(t.raw)) for s in sentences(ref) for t in s)
        decisions = fixer.decisions(hyp)
        controls = fixer.decisions(ref) if hyp != ref else decisions
        distances = {}
        for cut in cutoffs:
            edits = suggestions(decisions, cut)
            corrected = apply_edits(hyp, edits)
            if corrected not in distances:
                hwords = words(corrected)
                distances[corrected] = distance(rwords, hwords), distance(rchars, "".join(hwords))
            wd, cd = distances[corrected]
            by_cut[cut][row["group"]] += [
                len(rwords), wd, len(rchars), cd, eligible, len(suggestions(controls, cut)),
                len(edits), int(bool(edits)),
            ]
    return by_cut


def summary(group_counts):
    totals = np.sum(list(group_counts.values()), axis=0)
    if min(totals[WORD_REF], totals[CHAR_REF], totals[CONTROL_WORDS]) <= 0:
        raise ValueError("Evaluation needs nonempty Urdu reference controls.")
    return {
        "word_errors": int(totals[WORD_ERR]),
        "reference_words": int(totals[WORD_REF]),
        "wer_pct": float(100 * totals[WORD_ERR] / totals[WORD_REF]),
        "character_errors": int(totals[CHAR_ERR]),
        "reference_characters": int(totals[CHAR_REF]),
        "cer_pct": float(100 * totals[CHAR_ERR] / totals[CHAR_REF]),
        "reference_control_edits": int(totals[CONTROL_EDITS]),
        "control_words": int(totals[CONTROL_WORDS]),
        "reference_edits_per_1000": float(1000 * totals[CONTROL_EDITS] / totals[CONTROL_WORDS]),
        "hypothesis_edits": int(totals[HYP_EDITS]),
        "changed_utterances": int(totals[CHANGED]),
    }


def select(stats):
    raw = stats[None]
    eligible = [
        cutoff for cutoff, s in stats.items()
        if cutoff is not None and s["word_errors"] < raw["word_errors"]
        and s["character_errors"] <= raw["character_errors"] and s["reference_edits_per_1000"] <= 0.5
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda t: (stats[t]["word_errors"], stats[t]["character_errors"],
                                         stats[t]["reference_control_edits"], -t))


def paired_change(before, after):
    keys = sorted(before)
    if set(keys) != set(after):
        raise ValueError("Unmatched groups for paired comparison.")
    a = np.stack([before[k] for k in keys])
    b = np.stack([after[k] for k in keys])
    rng = np.random.default_rng(27)
    metrics = {}
    for name, error_col, ref_col in (("WER",WORD_ERR,WORD_REF), ("CER",CHAR_ERR,CHAR_REF)):
        delta = b[:,error_col] - a[:,error_col]
        denom = a[:,ref_col]
        change = float(100 * delta.sum()/denom.sum())
        interval = None
        if len(keys) > 1:
            estimates = []
            for _ in range(2000):
                idx = rng.integers(0,len(keys),size=len(keys))
                total = denom[idx].sum()
                if total:
                    estimates.append(float(100*delta[idx].sum()/total))
            if estimates:
                interval = [float(v) for v in np.percentile(estimates,[2.5,97.5])]
        metrics[name] = {"delta_percentage_points":change, "group_bootstrap_95":interval}
    return metrics


def experiment(fixer, current, dev, frozen):
    tune, confirm, splits = partition(dev, frozen)
    grid = sorted(set(GRID) | ({current} if current is not None else set()))
    counts = compute(fixer, tune, [None, *grid])
    stats = {t:summary(cs) for t,cs in counts.items()}
    chosen = select(stats)
    candidates = dict.fromkeys([None, current, chosen])
    confirmation_counts = compute(fixer, confirm, candidates)
    arms = {}
    for name,cut in (("raw_zipformer",None), ("shipped",current), ("selected_on_calibration",chosen)):
        arm = confirmation_counts[cut]
        arms[name] = {"threshold":cut, **summary(arm), "vs_raw":paired_change(confirmation_counts[None],arm)}
    return {"split":splits, "shipped_threshold":current, "selected_threshold":chosen,
            "selection_rule":"Require fewer word errors, no more character errors than raw Zipformer, and at most "
                             "0.5 reference-control edits per 1000 on calibration. Minimize word errors, then "
                             "character errors, then reference-control edits, then prefer the higher threshold. "
                             "A null threshold keeps the original transcript.",
            "calibration":[{"threshold":t, **s} for t,s in stats.items()],
            "confirmation":arms,
            "selected_vs_shipped_on_confirmation":paired_change(confirmation_counts[current],confirmation_counts[chosen])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=ROOT / "release/hf")
    p.add_argument("--work", type=Path, default=ROOT / "work/asr-dev")
    p.add_argument("--rows", type=Path, help="Existing independent dev JSONL with id, group, reference and hypothesis.")
    p.add_argument("--frozen-test", type=Path, default=ROOT / "work/fleurs299.jsonl")
    p.add_argument("--asr-repo", type=Path, help="Local Urdu-Streaming-ASR checkout for creating validation predictions.")
    p.add_argument("--onnx-dir", type=Path, help="Local directory with the three released INT8 Zipformer components.")
    p.add_argument("--tokens", type=Path, help="Matching Zipformer tokens.txt if stored separately.")
    p.add_argument("--parquet", type=Path, help="Existing copy of the pinned ur_pk validation shard.")
    a = p.parse_args()
    if a.rows and any((a.asr_repo,a.onnx_dir,a.tokens,a.parquet)):
        p.error("Use --rows for existing predictions, or supply decoder options to create the predictions.")
    a.work.mkdir(parents=True, exist_ok=True)
    out = a.work / "threshold-experiment.json"
    inputs = [f for f in (a.rows,a.frozen_test,a.parquet,a.tokens) if f is not None]
    if out.resolve() in {f.resolve() for f in inputs}:
        p.error("The report must have a distinct path from input data.")
    fixed = read_rows(a.frozen_test)
    fixer, current, policy_sha = load_bundle(a.bundle)
    path = a.rows or validation_rows(a.work, a.asr_repo, a.onnx_dir, a.tokens, a.parquet)
    report = experiment(fixer, current, read_rows(path), fixed)
    report.update(model_sha256=fixer.fingerprint, resources_sha256=fixer.asset, policy_sha256=policy_sha,
                  development_rows_sha256=sha(path), frozen_test_rows_sha256=sha(a.frozen_test),
                  grouping="source-defined group; FLEURS prompts for generated predictions")
    write_json(out, report)
    print("Development split:",json.dumps(report["split"]))
    print("Selected threshold (None means keep raw):",report["selected_threshold"])
    for name, s in report["confirmation"].items():
        print(f"{name}: WER {s['wer_pct']:.4f}%, CER {s['cer_pct']:.4f}%, changed {s['changed_utterances']} recordings, "
              f"{s['reference_edits_per_1000']:.3f} edits per 1000 reference-control words")
    print("Paired changes and full report:",out)


if __name__ == "__main__":
    main()

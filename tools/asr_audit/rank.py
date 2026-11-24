"""Explain reference-word misses at the existing, frozen Kinnow operating point."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np

from common import ROOT, load_bundle, read_rows, unique_middles
from score_asr import distance, words
from urdu_core import Fixer, cutoff_for, sha, span_key, suggestions, write_json


class TracedFixer(Fixer):
    """Capture the backend outputs used by the shipped decisions() function."""

    def score(self, left, right, candidates):
        scores = super().score(left, right, candidates)
        if not np.isfinite(scores[candidates != 0]).all():
            raise ValueError("The backend returned a non-finite score.")
        batches = getattr(self, "_scored_batches", None)
        if batches is not None:
            batches.append((candidates, scores))
        return scores

    def traced_decisions(self, text):
        self._scored_batches = []
        try:
            decisions = self.decisions(text)
            scored = [(ids, logits) for candidates, scores in self._scored_batches
                      for ids, logits in zip(candidates, scores)]
            if len(scored) != len(decisions):
                raise RuntimeError("The installed Kinnow scorer has a different batch layout.")
            return list(zip(decisions, scored))
        finally:
            del self._scored_batches


def reference_differences(row):
    hyp = words(row["hypothesis"])
    ref = words(row["reference"])
    reference_by_context = unique_middles(ref)
    positions = {}
    for context, index in unique_middles(hyp).items():
        other = reference_by_context.get(context)
        if other is None:
            continue
        observed, intended = hyp[index], ref[other]
        if span_key(observed) and span_key(intended) and distance(observed, intended) == 1:
            positions[index] = (observed, intended)
    return hyp, positions


def map_scored_positions(text, tokens, positions, traces):
    # Evaluation folds spelling and removes punctuation. Model inference uses original
    # Unicode spans and sentence boundaries. Match a whole token and verify its offset.
    wanted_forms = {observed for observed, _ in positions.values()}
    by_index = {}
    for decision, scored in traces:
        normalized = words(decision["original"])
        if len(normalized) != 1 or normalized[0] not in wanted_forms:
            continue
        before = words(text[:decision["start"]])
        index = len(before)
        if index not in positions or tokens[index] != normalized[0]:
            continue
        if before + normalized + words(text[decision["end"]:]) != tokens:
            continue
        if index in by_index:
            raise ValueError("Ambiguous model span for a normalized word position.")
        by_index[index] = decision, scored
    return by_index


def coverage(fixer, observed, reference):
    obs = fixer.ids.get(observed)
    gold = fixer.ids.get(reference)
    if obs is None or gold is None:
        return "outside_vocabulary"
    if gold not in fixer.res["candidates"][obs][1:]:
        return "in_vocabulary_off_shortlist"
    return "on_shortlist"


def rank_detail(fixer, cutoff, traced, intended):
    if traced is None:
        return {"reason": "runtime_skipped_or_unmapped"}
    decision, (candidate_ids, logits) = traced
    n = int(np.count_nonzero(candidate_ids))
    ids = candidate_ids[:n]
    scores = logits[:n]
    matches = np.flatnonzero(ids == fixer.ids[intended])
    if len(matches) != 1:
        raise RuntimeError("Runtime shortlist differs from the resources at this position.")
    slot = int(matches[0])
    reference_score = scores[slot]
    # At equal score, the typed word comes first and alternative ties use slot order.
    rank = 1 + int(np.sum(scores > reference_score)) + int(np.sum(scores[:slot] == reference_score))
    best = decision["replacement"]
    margin = decision["margin"]
    if margin <= 0:
        reason = "typed_word_won_or_tied"
    elif best != intended:
        reason = "different_replacement_ranked_first"
    elif cutoff_for(decision, cutoff) is None:
        reason = "reference_first_policy_abstains"
    elif margin < cutoff_for(decision, cutoff):
        reason = "reference_first_below_cutoff"
    else:
        reason = "reference_applied"
    return {
        "reason": reason,
        "reference_rank_including_typed": rank,
        "reference_margin_vs_typed": float(reference_score - scores[0]),
        "best_alternative": best,
        "best_alternative_margin_vs_typed": margin,
        "start": decision["start"], "end": decision["end"],
        "applied_replacement": best if suggestions([decision], cutoff) else None,
    }


def audit(fixer, cutoff, rows):
    coverage_counts = Counter()
    reasons = Counter()
    details = []
    changed = 0
    total_suggestions = 0
    for row in rows:
        tokens, positions = reference_differences(row)
        traces = fixer.traced_decisions(row["hypothesis"])
        changes = suggestions([e for e, _ in traces], cutoff)
        changed += int(bool(changes))
        total_suggestions += len(changes)
        scored = map_scored_positions(row["hypothesis"], tokens, positions, traces)
        for index, (observed, intended) in positions.items():
            category = coverage(fixer, observed, intended)
            coverage_counts[category] += 1
            item = {"id": row["id"], "group": row["group"], "token_index": index,
                    "left": tokens[index-1], "observed": observed, "right": tokens[index+1],
                    "reference": intended, "coverage": category}
            if category == "on_shortlist":
                item.update(rank_detail(fixer, cutoff, scored.get(index), intended))
                reasons[item["reason"]] += 1
            elif category == "outside_vocabulary":
                item["observed_in_vocabulary"] = observed in fixer.ids
                item["reference_in_vocabulary"] = intended in fixer.ids
            details.append(item)
    for key in ("outside_vocabulary", "in_vocabulary_off_shortlist", "on_shortlist"):
        coverage_counts.setdefault(key, 0)
    for key in ("typed_word_won_or_tied", "different_replacement_ranked_first",
                "reference_first_below_cutoff", "reference_applied", "runtime_skipped_or_unmapped",
                "reference_first_policy_abstains"):
        reasons.setdefault(key, 0)
    assert sum(reasons.values()) == coverage_counts["on_shortlist"]
    return {"utterances": len(rows), "groups": len({r["group"] for r in rows}),
            "positions_checked": len(details), "coverage": dict(coverage_counts),
            "shortlisted_outcomes": dict(reasons), "changed_utterances_all_positions": changed,
            "applied_suggestions_all_positions": total_suggestions, "threshold": cutoff,
            "model_sha256": fixer.fingerprint, "resources_sha256": fixer.asset, "positions": details}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=Path, default=ROOT / "work/fleurs299.jsonl")
    p.add_argument("--bundle", type=Path, default=ROOT / "release/hf")
    p.add_argument("--out", type=Path)
    p.add_argument("--example", nargs=2, metavar=("OBSERVED", "REFERENCE"))
    a = p.parse_args()
    out = a.out or ROOT / "work/wiki" / (a.rows.stem + "-rank-audit.json")
    protected = [a.rows, *(a.bundle / n for n in ("model.onnx","policy.json","resources.json"))]
    if out.resolve() in {f.resolve() for f in protected}:
        p.error("Choose an output path distinct from the input rows and model bundle.")
    model, cutoff, policy_sha = load_bundle(a.bundle, TracedFixer)
    report = audit(model, cutoff, read_rows(a.rows))
    report.update(policy_sha256=policy_sha, rows_sha256=sha(a.rows))
    write_json(out, report)
    for k in ("model_sha256", "policy_sha256", "utterances", "positions_checked", "coverage",
              "shortlisted_outcomes", "changed_utterances_all_positions", "threshold"):
        print(f"{k}: {json.dumps(report[k], ensure_ascii=False)}")
    if a.example:
        for item in report["positions"]:
            if [item["observed"], item["reference"]] == a.example:
                print(json.dumps(item, ensure_ascii=False))
    print("Detailed report:", out)


if __name__ == "__main__":
    main()

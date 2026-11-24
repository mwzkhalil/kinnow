"""Shared inputs for diagnostics run inside the Kinnow repository."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]  # repository root
if not (ROOT / "src" / "urdu_core.py").is_file():
    raise SystemExit("Run from the kinnow repository (tools/asr_audit/).")
sys.path.insert(0, str(ROOT / "src"))
from evaluate import verify_policy
from urdu_core import Fixer, read_json, sha


def load_bundle(directory, fixer_class=Fixer):
    model = fixer_class(directory / "model.onnx", directory / "resources.json")
    policy = directory / "policy.json"
    return model, verify_policy(model, read_json(policy)), sha(policy)


def read_rows(path):
    rows = []
    ids = set()
    with Path(path).open(encoding="utf-8-sig") as src:
        for n, line in enumerate(src, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("id", "group", "reference", "hypothesis"):
                if not isinstance(row.get(key), str) or (key != "hypothesis" and not row[key].strip()):
                    raise ValueError(f"{path}, line {n}: expected a string in {key!r}.")
            if row["id"] in ids:
                raise ValueError(f"Duplicate recording ID in {path}: {row['id']}")
            ids.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError(f"No recordings in {path}")
    return rows


def unique_middles(tokens):
    triples = list(zip(tokens, tokens[1:], tokens[2:]))
    frequency = Counter((left, right) for left, _, right in triples)
    return {(left, right): i for i, (left, _, right) in enumerate(triples, 1)
            if frequency[left, right] == 1}

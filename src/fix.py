"""Offline suggestion CLI. Input offsets are Unicode code points."""
import argparse, json, sys
from pathlib import Path
from urdu_core import Fixer, apply_edits, read_json, suggestions
from evaluate import verify_policy
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--resources', type=Path, required=True)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--text')
    p.add_argument('--apply', action='store_true')
    a = p.parse_args()
    fx = Fixer(a.model, a.resources)
    cut = verify_policy(fx, read_json(a.policy))
    def run(s):
        edits = suggestions(fx.decisions(s), cut)
        result = {'suggestions': edits}
        if a.apply:
            result['text'] = apply_edits(s, edits)
        return result
    if a.text is not None:
        print(json.dumps(run(a.text), ensure_ascii=False))
    else:
        for line in sys.stdin:
            if line.strip():
                print(json.dumps(run(json.loads(line)['text']), ensure_ascii=False), flush=True)
if __name__=="__main__":
    main()

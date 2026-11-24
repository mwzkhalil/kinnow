import argparse
from collections import defaultdict
import json
from pathlib import Path
import unicodedata
import numpy as np
from evaluate import verify_policy
from urdu_core import Fixer, apply_edits, lookup, read_json, sha, suggestions, write_json

def words(text):
    text = lookup(text).casefold()
    return ''.join((' ' if unicodedata.category(c)[0] in ('P', 'S') else c for c in text)).split()

def distance(a,b):
    if len(a)<len(b):
        a, b = (b, a)
    row = list(range(len(b) + 1))
    for i,x in enumerate(a,1):
        new = [i]
        for j,y in enumerate(b,1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = new
    return row[-1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model","resources","policy","rows","out"):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    fx = Fixer(a.model, a.resources)
    cut = verify_policy(fx, read_json(a.policy))
    groups = defaultdict(lambda: np.zeros(6, dtype=np.int64))
    seen = set()
    changed = 0
    with a.rows.open(encoding="utf-8-sig") as f:
        for n,line in enumerate(f,1):
            if not line.strip():
                continue
            r = json.loads(line)
            if not all(isinstance(r.get(k),str) for k in ("id","group","reference","hypothesis")) or not r["group"].strip() or not r["id"].strip():
                raise ValueError(f'Invalid row {n}')
            if r["id"] in seen:
                raise ValueError('Repeated utterance ID ' + r['id'])
            seen.add(r['id'])
            hyp = r['hypothesis']
            ref = r['reference']
            edits = suggestions(fx.decisions(hyp), cut)
            changed += bool(edits)
            w, b, t = map(words, (ref, hyp, apply_edits(hyp, edits)))
            # Duty of the corpus assembler is to retain authorized exact references and group by speaker/session.
            cw, cb, ct = map(''.join, (w, b, t))
            groups[r["group"]]+=[len(w),distance(w,b),distance(w,t),len(cw),distance(cw,cb),distance(cw,ct)]
    arr = np.array(list(groups.values()))
    if not len(arr) or not arr[:,0].sum() or not arr[:,3].sum():
        raise ValueError('References need some words.')
    total = arr.sum(0)
    rng = np.random.default_rng(27)
    report={"utterances":len(seen),"groups":len(groups),"changed_utterances":changed,
            "model_sha256":fx.fingerprint,"policy_sha256":sha(a.policy)}
    for j,metric in ((0,"WER"),(3,"CER")):
        stat={"before_pct":100*total[j+1]/total[j],"after_pct":100*total[j+2]/total[j],
              "delta_percentage_points":100*(total[j+2]-total[j+1])/total[j],"delta_group_bootstrap_95":None}
        if len(arr)>1:
            boot = []
            for _ in range(2000):
                x = arr[rng.integers(0, len(arr), len(arr))].sum(0)
                if x[j]:
                    boot.append(100 * (x[j + 2] - x[j + 1]) / x[j])
            if boot:
                stat['delta_group_bootstrap_95'] = list(np.percentile(boot, [2.5, 97.5]))
        report[metric] = stat
    write_json(a.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__=="__main__":
    main()

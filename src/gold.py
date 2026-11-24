"""Convert independently reviewed sentence pairs to exact Unicode-offset gold JSONL."""
import argparse
import json
from pathlib import Path
import os
import shutil
import tempfile
from urdu_core import apply_edits, lookup, sentences, span_key

def edits_for(typed,corrected):
    if typed==corrected:
        return []
    def spans(text):
        return [t for sent in sentences(text) for t in sent]
    before, after = (spans(typed), spans(corrected))
    if len(before)!=len(after):
        raise ValueError('Different word counts; review for task scope.')
    edits = []
    for old,new in zip(before,after):
        if old.raw!=new.raw:
            if not (span_key(lookup(old.raw)) and span_key(lookup(new.raw))):
                raise ValueError('Expected a single Urdu real word in each editable position.')
            edits.append({'start': old.start, 'end': old.end, 'original': old.raw, 'replacement': new.raw})
    if apply_edits(typed,edits)!=corrected:
        raise ValueError('Whitespace, punctuation or another unsupported edit differs.')
    return [{k: v for k, v in e.items() if k != 'original'} for e in edits]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument("--reviewed",action="store_true",help="Mark genuine, editor-reviewed real-world pairs as human.")
    a = p.parse_args()
    if a.out.exists():
        p.error('Choose a new output directory.')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=a.out.parent, prefix='gold-'))
    groups = {}
    identities = set()
    counts = {'dev': 0, 'test': 0}
    from contextlib import ExitStack
    try:
        with ExitStack() as stack:
            handles={k:stack.enter_context((tmp/(k+".jsonl")).open("w",encoding="utf-8")) for k in counts}
            with a.pairs.open(encoding="utf-8-sig") as src:
                for num,line in enumerate(src,1):
                    if not line.strip():
                        continue
                    try:
                        x = json.loads(line)
                        d = {k: x[k] for k in ('id', 'group', 'split', 'domain')}
                        if not all(isinstance(v,str) and v.strip() for v in d.values()) or d["split"] not in counts:
                            raise ValueError('Invalid identity or split')
                        if not all(isinstance(x.get(k),str) and x[k].strip() for k in ("typed","corrected")):
                            raise ValueError('Empty typed or corrected string')
                        if d["id"] in identities or groups.setdefault(d["group"],d["split"])!=d["split"]:
                            raise ValueError('Duplicate ID or group overlap')
                        identities.add(d['id'])
                        row=dict(d,kind="human" if a.reviewed else "example",text=x["typed"],edits=edits_for(x["typed"],x["corrected"]))
                        handles[d['split']].write(json.dumps(row, ensure_ascii=False) + '\n')
                        counts[d['split']] += 1
                    except (KeyError,ValueError,TypeError) as e:
                        raise ValueError(f'Input line {num}: {e}') from e
        if not all(counts.values()):
            raise ValueError('Both dev and test sets need at least one document.')
        tmp.rename(a.out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(counts)
if __name__=="__main__":
    main()

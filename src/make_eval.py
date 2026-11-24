"""Create repeatable synthetic calibration and test data from held-out corpus documents."""
import argparse, hashlib, json, os, sqlite3, tempfile
from pathlib import Path
import random
from urdu_core import read_json, sentences

def make(data,split,out,limit=5000,seed=27,other_noisy=None):
    res = read_json(data / 'resources.json')
    words = res['words']
    ids = {w: i for i, w in enumerate(words)}
    db = sqlite3.connect(f"{(data / 'provenance.sqlite').resolve().as_uri()}?mode=ro", uri=True)
    rng = random.Random(seed + (split == 'test'))
    total = 0
    noisy = 0
    used_noisy = other_noisy if other_noisy is not None else set()
    if out.exists():
        raise FileExistsError(out)
    # Signatures give a reproducible sample across the entire shard, regardless of article order.
    sql="SELECT lines.docid,docs.grp,hex(lines.sig),lines.words FROM lines JOIN docs ON docs.id=lines.docid WHERE lines.part=? ORDER BY lines.sig"
    with tempfile.NamedTemporaryFile("w",dir=out.parent,delete=False,encoding="utf-8") as f:
        tmp = Path(f.name)
        try:
            for doc,grp,sig,text in db.execute(sql,(split,)):
                # Use an inert Latin placeholder for protected/non-Urdu tokens in the diagnostic.
                text = text.replace('<other>', 'XYZ').replace('<num>', '123')
                base = {'group': grp, 'domain': 'synthetic-heldout', 'kind': 'synthetic', 'split': split}
                f.write(json.dumps({**base,"id":sig+"-keep","text":text,"edits":[]},ensure_ascii=False)+"\n")
                eligible=[t for sent in sentences(text) for t in sent if t.editable and len(res["candidates"][ids.get(t.key,0)])>1]
                if eligible:
                    t = rng.choice(eligible)
                    alternate = words[rng.choice(res['candidates'][ids[t.key]][1:])]
                    observed = text[:t.start] + alternate + text[t.end:]
                    noise_line = ' '.join((tok.key for segment in sentences(observed) for tok in segment))
                    digest = hashlib.sha256(noise_line.encode()).digest()
                    # An injected error can accidentally recreate a genuine corpus sentence.
                    if digest not in used_noisy and not db.execute("SELECT 1 FROM lines WHERE sig=?", (digest,)).fetchone():
                        used_noisy.add(digest)
                        e = {'start': t.start, 'end': t.start + len(alternate), 'replacement': t.raw}
                        f.write(json.dumps({**base,"id":sig+"-noisy","text":observed,"edits":[e]},ensure_ascii=False)+"\n")
                        noisy += 1
                total += 1
                if total>=limit:
                    break
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    db.close()
    os.replace(tmp, out)
    return {'clean_rows': total, 'injected_rows': noisy}
if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit', type=int, default=5000)
    a = p.parse_args()
    if a.limit<1:
        p.error('limit must be positive')
    a.out.mkdir(parents=True, exist_ok=True)
    used_noisy = set()
    for split in ["dev","test"]:
        path = a.out / (split + '.jsonl')
        print(split, make(a.data, split, path, a.limit, other_noisy=used_noisy))

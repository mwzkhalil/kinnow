import argparse
from array import array
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import urllib.request
import numpy as np
from urdu_core import SPECIAL, UNK, build_resources, lookup, read_groups, sentences, sha, span_key, write_json

BASE = Path(__file__).resolve().parent.parent  # repository root
REV = '126488064a94417b849f69e1223923108896203e'
FILE = '20231101.ur/train-00000-of-00001.parquet'
CHECKSUM = 'f711c56af77a88ea8036282c9c49243d0319fe0ddf08ea4e88800d0482d6f8d0'
URL = f'https://huggingface.co/datasets/wikimedia/wikipedia/resolve/{REV}/{FILE}'

def wiki(cache):
    import pyarrow.parquet as pq
    cache.mkdir(parents=True, exist_ok=True)
    final = cache / 'urwiki-20231101.parquet'
    if not final.exists() or sha(final)!=CHECKSUM:
        with tempfile.NamedTemporaryFile(dir=cache,delete=False) as f:
            temp = Path(f.name)
            try:
                with urllib.request.urlopen(URL,timeout=120) as src:
                    shutil.copyfileobj(src, f, 1 << 20)
            except BaseException:
                temp.unlink(missing_ok=True)
                raise
        if sha(temp)!=CHECKSUM:
            temp.unlink()
            raise ValueError('Downloaded Urdu Wikipedia shard failed its SHA-256 check.')
        os.replace(temp, final)
    pf = pq.ParquetFile(final)
    if pf.metadata.num_rows!=200154:
        raise ValueError('Unexpected wiki snapshot size.')
    for batch in pf.iter_batches(batch_size=256,columns=["id","text","url"]):
        for row in batch.to_pylist():
            yield {'id': str(row['id']), 'text': row['text'], 'source': 'urwiki-20231101', 'url': row['url']}

def docs_from_file(path):
    with path.open(encoding="utf-8-sig") as f:
        for n,line in enumerate(f,1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not all(isinstance(row.get(k),str) and row[k].strip() for k in ("id","source","text")):
                raise ValueError(f'{path}:{n}: expected nonempty string id, source, text')
            yield row

def split_for(group):
    # The salt string is historical. Changing it reshuffles the train/dev/test split.
    # The salt string is historical. Changing it reshuffles the train/dev/test split.
    val = int(hashlib.sha256(('urdu-tinyfix-v1\x00' + group).encode()).hexdigest()[:8], 16) % 100
    return 'test' if val < 5 else 'dev' if val < 10 else 'train'

def prepare(target, streams,meta,cap,min_count,groups,protected):
    target = Path(target)
    if target.exists():
        raise FileExistsError(f'{target} already exists; select a fresh directory for a new corpus.')
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=target.name + '.partial-', dir=target.parent))
    try:
        db = sqlite3.connect(stage / 'provenance.sqlite')
        with db:
            db.executescript("CREATE TABLE docs (id TEXT PRIMARY KEY, grp TEXT, part TEXT, url TEXT);"
                             "CREATE TABLE lines (sig BLOB PRIMARY KEY, docid TEXT, part TEXT, words TEXT);"
                             "CREATE INDEX docs_grp_part ON docs(grp,part);")
            totals = Counter()
            vocab = Counter()
            seen_sources = Counter()
            for stream in streams:
                for row in stream:
                    key = row['source'] + '\x00' + row['id']
                    group = str(row.get('group') or key)
                    part = split_for(group)
                    db.execute('INSERT INTO docs VALUES(?,?,?,?)', (key, group, part, row.get('url', '')))
                    seen_sources[row['source']] += 1
                    for sentence in sentences(row["text"]):
                        terms = [t.key for t in sentence]
                        ordinary = [w for w in terms if span_key(w)]
                        if not (4<=len(terms)<=120) or len(ordinary)<max(3,len(terms)//2):
                            continue
                        line = ' '.join(terms)
                        sig = hashlib.sha256(line.encode()).digest()
                        cur=db.execute("INSERT OR IGNORE INTO lines VALUES(?,?,?,?)",(sig,key,part,line))
                        if cur.rowcount != 1:
                            totals['duplicate_sentences'] += 1
                            continue
                        if part=="train":
                            vocab.update(ordinary)
                        totals[part + '_sentences'] += 1
            res = build_resources(vocab, groups, protected, cap, min_count)
            write_json(stage / 'resources.json', res)
            ids = {w: i for i, w in enumerate(res['words'])}
            for part in ["train","dev","test"]:
                ends = array('Q')
                offset = 0
                with (stage/f"{part}.ids").open("wb") as out:
                    for (line,) in db.execute("SELECT words FROM lines WHERE part=? ORDER BY rowid",(part,)):
                        seq = np.array([ids.get(w, UNK) for w in line.split()], dtype='<u4')
                        out.write(seq.tobytes())
                        offset += len(seq)
                        ends.append(offset)
                np.asarray(ends, dtype='<u8').tofile(stage / f'{part}.ends')
                totals[part + '_tokens'] = offset
                if not offset:
                    raise ValueError(f'Empty {part} split; add more independent documents.')
        db.close()
        info={"format":1,"sources":meta,"document_counts":dict(seen_sources),"counts":dict(totals),
              "vocab":len(ids),"split_policy":"SHA256 group buckets 90/5/5; global exact lexical-sentence dedup, first occurrence retained",
              "files":{p.name:sha(p) for p in sorted(stage.iterdir())}}
        write_json(stage / 'manifest.json', info)
        stage.rename(target)
        return info
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wiki', action='store_true')
    p.add_argument('--jsonl', type=Path, action='append', default=[])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cache', type=Path, default=Path('cache'))
    p.add_argument('--vocab', type=int, default=20000)
    p.add_argument('--min-count', type=int, default=3)
    p.add_argument('--groups', type=Path, default=BASE / 'data/confusions.tsv')
    p.add_argument('--protect', type=Path, default=BASE / 'data/protect.txt')
    a = p.parse_args()
    if not(a.wiki or a.jsonl) or a.vocab<8 or a.min_count<1:
        p.error('Supply --wiki and/or --jsonl, vocab >=8 and min-count >=1.')
    streams = []
    source = []
    if a.wiki:
        streams.append(wiki(a.cache))
        source.append({"dataset":"wikimedia/wikipedia","revision":REV,"file":FILE,"sha256":CHECKSUM,"card_license":"CC BY-SA 3.0 / GFDL"})
    for path in a.jsonl:
        streams.append(docs_from_file(path))
        source.append({'local_file': path.name, 'sha256': sha(path)})
    protect={lookup(w.strip()) for w in a.protect.read_text(encoding="utf-8").splitlines() if w.strip() and not w.startswith("#")}
    info = prepare(a.out, streams, source, a.vocab, a.min_count, read_groups(a.groups), protect)
    print(json.dumps(info['counts'], indent=2))
    print(f"Vocab: {info['vocab']}")
if __name__=="__main__":
    main()

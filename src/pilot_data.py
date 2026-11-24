"""Make the reproducible pilot corpus from authored examples and an optional Inkscape catalogue."""
import argparse, gettext, hashlib, json, re
from pathlib import Path
from prepare import BASE

def records(catalog=None):
    prefixes = ['', 'اس بار ', 'ایک مرتبہ ', 'پھر ', 'بتایا گیا کہ ', 'ہم نے سنا کہ ']
    for i,line in enumerate((BASE/"data/seed_sentences.txt").read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        for j,prefix in enumerate(prefixes):
            yield {"source":"authored-pilot","id":f"{i}-{j}","group":f"pilot-pattern-{i}","text":prefix+line}
    if catalog:
        with Path(catalog).open("rb") as f:
            translations = gettext.GNUTranslations(f)
        for k,v in sorted(translations._catalog.items(),key=lambda x:str(x[0])):
            if not k or not isinstance(v,str) or len(v)>650:
                continue
            if "<" in v or "%" in v:
                continue
            clean = v.replace('_', '')
            if len(re.findall(r"[\u0600-\u06ff]+",clean))<4:
                continue
            ident = hashlib.sha256(str(k).encode()).hexdigest()
            yield {'source': 'inkscape-ur', 'id': ident, 'text': clean}

if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    with a.out.open("x",encoding="utf-8") as f:
        for d in records(a.catalog):
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

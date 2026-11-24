import hashlib
import json
import math
import os
import re
import tempfile
import unicodedata as ud
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional
import numpy as np

PAD, UNK, BOS, NUM, OTHER = range(5)
SPECIAL = ['<pad>', '<unk>', '<s>', '<num>', '<other>']
LEFT, RIGHT, WIDTH = (16, 4, 9)
PROTECTED = re.compile(r"```[\s\S]*?(?:```|\Z)|`[^`\n]*(?:`|(?=\n)|\Z)|(?:https?://|www\.)[^\s<>]+|[\w.+-]+@[\w.-]+\.[\w-]+|[@#][\w\u200c\u200d]+", re.IGNORECASE)
BOUND = set('۔؟!?\r\n.;؛')
JOIN = {'\u200c', '\u200d'}
FOLD = str.maketrans({'ي': 'ی', 'ك': 'ک', 'ـ': ''})

def is_arabic(c):
    n = ord(c)
    return (0x600 <= n <= 0x6ff or 0x750 <= n <= 0x77f or 0x8a0 <= n <= 0x8ff
            or 0xfb50 <= n <= 0xfdff or 0xfe70 <= n <= 0xfeff)

def lookup(s):
    return ud.normalize('NFC', ud.normalize('NFKC', s).translate(FOLD))

@dataclass(frozen=True)
class Token:
    start: int
    end: int
    raw: str
    key: str
    editable: bool

def sentences(text):
    """Yield lists of lexical tokens; offsets count Unicode code points in the unchanged input."""
    sent, i = ([], 0)
    while i < len(text):
        c = text[i]
        # Check protected runs at the start of a token; do not match a suffix of an identifier.
        if c.isalnum() or c in "_@`#":
            protected = PROTECTED.match(text, i)
            if protected:
                j = protected.end()
                sent.append(Token(i, j, text[i:j], SPECIAL[OTHER], False))
                if "\n" in text[i:j]:
                    yield sent
                    sent = []
                i = j
                continue
        if c.isalnum() or c == "_":
            j = i + 1
            while j < len(text) and (text[j].isalnum() or text[j] in JOIN or text[j]=="_" or ud.category(text[j]).startswith("M")):
                j += 1
            s = text[i:j]
            key = lookup(s)
            arabic = all((is_arabic(ch) and ud.category(ch).startswith('L') for ch in key))
            editable = arabic and len(key) >= 2 and not any(ud.category(ch).startswith("M") or ch in JOIN or ch=="ـ" for ch in s)
            if not arabic:
                key = SPECIAL[NUM] if s.isdecimal() else SPECIAL[OTHER]
            sent.append(Token(i, j, s, key, editable))
            i = j
            continue
        if c in BOUND and sent:
            yield sent
            sent = []
        i += 1
    if sent:
        yield sent

def span_key(word):
    return len(word) >= 2 and all((is_arabic(c) and ud.category(c).startswith('L') for c in word))

def one_edit(a,b):
    if a == b or abs(len(a)-len(b)) > 1:
        return False
    if len(a) == len(b):
        ix = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        return len(ix)==1 or (len(ix)==2 and ix[1]==ix[0]+1 and a[ix[0]]==b[ix[1]] and a[ix[1]]==b[ix[0]])
    if len(a)>len(b):
        a, b = (b, a)
    p = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), len(a))
    return a[p:] == b[p + 1:]

def read_groups(path):
    lines = Path(path).read_text(encoding='utf-8-sig').splitlines()
    return [list(dict.fromkeys(lookup(x.strip()) for x in line.split("\t") if x.strip()))
            for line in lines if line.strip() and not line.lstrip().startswith("#")]

def build_resources(counts, groups, protected, vocab_cap, min_count):
    words = sorted((w for w,f in counts.items() if f>=min_count and span_key(w)), key=lambda w:(-counts[w],w))
    words = SPECIAL + words[:vocab_cap - len(SPECIAL)]
    ids = {w: i for i, w in enumerate(words)}
    banned = set(protected)
    allowed = [w for w in words[len(SPECIAL):] if w not in banned and len(w) <= 32]
    allowed_set = set(allowed)
    stems = defaultdict(list)
    for w in allowed:
        for k in {w[:i]+w[i+1:] for i in range(len(w))}:
            stems[k].append(w)
    curated = defaultdict(set)
    for group in groups:
        for w in group:
            curated[w].update((x for x in group if x != w and x in allowed_set))
    rows, priority = ([], [])
    for i,w in enumerate(words):
        if w not in allowed_set:
            rows.append([i] if i >= len(SPECIAL) else [])
            priority.append([])
            continue
        sortkey = lambda x: (-counts[x], x)
        manual = sorted(curated[w], key=sortkey)
        options = set(stems.get(w, ()))
        for stem in {w[:j]+w[j+1:] for j in range(len(w))}:
            if stem in allowed_set:
                options.add(stem)
            options.update(stems.get(stem, ()))
        for j in range(len(w)-1):
            swap = w[:j] + w[j + 1] + w[j] + w[j + 2:]
            if swap in allowed_set:
                options.add(swap)
        options.difference_update(manual)
        close = sorted((other for other in options if one_edit(w, other)), key=sortkey)
        rows.append([i] + [ids[x] for x in (manual + close)[:WIDTH - 1]])
        priority.append([ids[x] for x in manual[:WIDTH - 1]])
    return {"format":1,"words":words,"counts":[int(counts.get(w,0)) for w in words],
            "candidates":rows,"curated":priority,"protected":sorted(banned),"left":LEFT,"right":RIGHT,"width":WIDTH}

def write_json(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w",dir=path.parent,encoding="utf-8",delete=False) as f:
        tmp = Path(f.name)
        try:
            json.dump(value, f, ensure_ascii=False, sort_keys=True, allow_nan=False)
            f.write('\n')
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, path)

def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(path):
    with open(path,"rb") as f:
        digest = hashlib.sha256()
        for block in iter(lambda: f.read(1024*1024), b""):
            digest.update(block)
        return digest.hexdigest()

def save_npz(path,weights):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
        tmp = Path(f.name)
        try:
            np.savez_compressed(f, **weights)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, path)

def table(rows):
    ary = np.zeros((len(rows), WIDTH), dtype=np.int64)
    for i,row in enumerate(rows):
        ary[i, :len(row)] = row
    return ary

def contexts(flat, ends, positions):
    doc = np.searchsorted(ends, positions, side='right')
    lo = np.where(doc > 0, ends[np.maximum(doc - 1, 0)], 0)
    hi = ends[doc]
    left = positions[:, None] + np.arange(-LEFT, 0)
    right = positions[:, None] + np.arange(RIGHT, 0, -1)
    l=np.where(left>=lo[:,None],flat[left.clip(0,len(flat)-1)],np.where(left==lo[:,None]-1,BOS,PAD))
    r = np.where(right < hi[:, None], flat[right.clip(0, len(flat) - 1)], PAD)
    return (l.astype(np.int64), r.astype(np.int64))

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))

def gru_pass(x,w,prefix,cache=False):
    wi, wh, bi, bh = [w[prefix + k] for k in ('_wi', '_wh', '_bi', '_bh')]
    h = np.zeros((len(x), wh.shape[0]), np.float32)
    frames = []
    for j in range(x.shape[1]):
        a, b, c = np.split(x[:, j] @ wi + bi, 3, axis=1)
        d, e, f = np.split(h @ wh + bh, 3, axis=1)
        r, z = (sigmoid(a + d), sigmoid(b + e))
        n = np.tanh(c + r * f)
        if cache:
            frames.append((x[:, j], h, r, z, n, f))
        h = z * h + (1 - z) * n
    return (h, frames)

def forward(w,l,r,c,cache=False):
    E = w['emb']
    hl, cl = gru_pass(E[l], w, 'l', cache)
    hr, cr = gru_pass(E[r], w, 'r', cache)
    h = np.concatenate((hl, hr), axis=1)
    p = np.tanh(h @ w['proj'] + w['bias'])
    scores = (E[c] * p[:, None, :]).sum(-1)
    scores = np.where(c != PAD, scores, -10000.0)
    return (scores, (cl, cr, h, p) if cache else None)

class Fixer:
    def __init__(self,model,resources):
        self.res = read_json(resources)
        self.ids = {x: i for i, x in enumerate(self.res['words'])}
        self.graph = table(self.res['candidates'])
        self.curated = [frozenset(x) for x in self.res.get('curated', [])]
        self.fingerprint = sha(model)
        self.asset = sha(resources)
        if str(model).endswith(".onnx"):
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            self.session=ort.InferenceSession(str(model),sess_options=options,providers=["CPUExecutionProvider"])
            meta = self.session.get_modelmeta().custom_metadata_map
            if meta.get("resources_sha256")!=self.asset:
                raise ValueError('ONNX vocabulary mismatch')
            self.w = None
        else:
            with np.load(model,allow_pickle=False) as a:
                if str(a["resources_sha256"]) != self.asset:
                    raise ValueError('Vocabulary mismatch')
                self.w = {k: a[k] for k in a.files if k != 'resources_sha256'}
            self.session = None
    def score(self,l,r,c):
        if self.session is not None:
            return self.session.run(['scores'], {'left': l, 'right': r, 'candidates': c})[0]
        return forward(self.w, l, r, c)[0]
    def decisions(self,text):
        edits = []
        for sent in sentences(text):
            seq = np.array([self.ids.get(t.key, UNK) for t in sent], dtype=np.int64)
            pos=np.array([j for j,t in enumerate(sent) if t.editable and self.graph[seq[j],1]!=0],dtype=np.int64)
            for at in range(0,len(pos),256):
                pp = pos[at:at + 256]
                l, r = contexts(seq, np.array([len(seq)]), pp)
                c = self.graph[seq[pp]]
                s = self.score(l, r, c)
                for z,j in enumerate(pp):
                    count = np.count_nonzero(c[z])
                    k = 1 + int(np.argmax(s[z, 1:count]))
                    t = sent[j]
                    edits.append({"start":t.start,"end":t.end,"original":t.raw,
                                  "replacement":self.res["words"][c[z,k]],"margin":float(s[z,k]-s[z,0]),
                                  "curated":bool(self.curated) and int(c[z,k]) in self.curated[seq[j]]})
        return edits

class SplitCut(NamedTuple):
    """Separate margins for curated confusion pairs and for generic one-edit candidates.

    The English tinyfix uses a lower margin for likely slips; here a pair listed in
    data/confusions.tsv is the likely slip. None for a class means never suggest it.
    """
    curated: Optional[float]
    edit: Optional[float]

def cutoff_for(decision,threshold):
    """The margin a decision must reach; None means it is never suggested."""
    if isinstance(threshold, SplitCut):
        return threshold.curated if decision.get("curated") else threshold.edit
    return threshold

def suggestions(decisions,threshold):
    out = []
    for e in decisions:
        cut = cutoff_for(e, threshold)
        if cut is not None and e['margin'] > 0 and e['margin'] >= cut:
            out.append(e)
    return out

def apply_edits(text,edits):
    parts = []
    last = 0
    for e in sorted(edits,key=lambda x:x["start"]):
        i, j = (e['start'], e['end'])
        if i<last or i>=j or j>len(text) or text[i:j]!=e["original"]:
            raise ValueError('Stale or overlapping edit')
        parts.extend((text[last:i], e['replacement']))
        last = j
    return ''.join(parts) + text[last:]

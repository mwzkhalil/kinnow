from pathlib import Path
import numpy as np
from urdu_core import WIDTH, contexts, read_json, sha, table

class Sampler:
    def __init__(self, data, split, seed, verify=True):
        self.data = Path(data)
        m = read_json(self.data / 'manifest.json')
        if verify:
            for name in ("resources.json",split+".ids",split+".ends"):
                if sha(self.data/name)!=m["files"][name]:
                    raise ValueError('Corpus checksum mismatch: ' + name)
        self.words = np.memmap(self.data / (split + '.ids'), mode='r', dtype='<u4')
        self.ends = np.memmap(self.data / (split + '.ends'), mode='r', dtype='<u8').astype(np.int64)
        res = read_json(self.data / 'resources.json')
        self.graph = table(res['candidates'])
        inv = [[] for _ in self.graph]
        manual = [[] for _ in self.graph]
        for typed, opts in enumerate(res["candidates"]):
            for gold in opts[1:]:
                inv[gold].append(typed)
                if gold in res["curated"][typed]:
                    manual[gold].append(typed)
        def dense(rows):
            arr = np.zeros((len(rows), WIDTH - 1), dtype=np.int64)
            lens = np.zeros(len(rows), dtype=np.int64)
            for j,row in enumerate(rows):
                vals = list(dict.fromkeys(row))[:WIDTH - 1]
                arr[j, :len(vals)] = vals
                lens[j] = len(vals)
            return (arr, lens)
        self.inv, self.ninv = dense([manual[i] + lst for i, lst in enumerate(inv)])
        self.manual, self.nman = dense(manual)
        positions = []
        focused = []
        eligible = (self.ninv > 0) & (self.graph[:, 1] != 0)
        for start in range(0,len(self.words),1_000_000):
            block = self.words[start:start + 1000000]
            local = np.flatnonzero(eligible[block])
            positions.append(local + start)
            focused.append(local[self.nman[block[local]] > 0] + start)
        self.pos = np.concatenate(positions)
        self.focus = np.concatenate(focused)
        if not len(self.pos):
            raise ValueError(f'No trainable real-word positions in {split}.')
        self.rng = np.random.default_rng(seed)

    def batch(self,n):
        g = self.rng
        pp = g.choice(self.pos, n, replace=True)
        focus = g.random(n) < 0.25 if len(self.focus) else np.zeros(n, bool)
        if focus.any():
            pp[focus] = g.choice(self.focus, focus.sum(), replace=True)
        gold = self.words[pp].astype(np.int64)
        col = g.integers(np.maximum(self.ninv[gold], 1))
        wrong = self.inv[gold, col]
        if focus.any():
            which = gold[focus]
            wrong[focus] = self.manual[which, g.integers(self.nman[which])]
        typed = np.where(g.random(n) < 0.5, wrong, gold)
        cand = self.graph[typed]
        label = (cand == gold[:, None]).argmax(1)
        if not np.all(cand[np.arange(n),label]==gold):
            raise RuntimeError('Unreachable training label')
        l, r = contexts(self.words, self.ends, pp)
        return (l, r, cand, label)

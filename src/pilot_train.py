"""Small CPU reference trainer. Production GPU training uses train.py."""
import argparse, json
from pathlib import Path
import numpy as np
from urdu_core import forward, read_json, save_npz, sha, write_json
from sampling import Sampler

def init_weights(n,e,h,seed):
    g = np.random.default_rng(seed)
    def matrix(a,b):
        return g.normal(0, (2 / (a + b)) ** 0.5, (a, b)).astype(np.float32)
    w = {'emb': g.normal(0, 0.14, (n, e)).astype('f'), 'proj': matrix(2 * h, e), 'bias': np.zeros(e, 'f')}
    for p in ("l","r"):
        w[p + '_wi'] = matrix(e, 3 * h)
        w[p + '_wh'] = matrix(h, 3 * h)
        w[p + '_bi'] = np.zeros(3 * h, 'f')
        w[p + '_bh'] = np.zeros(3 * h, 'f')
    w['emb'][0] = 0
    return w

def backward_gru(w,g,p,frames,dh):
    dx = []
    for x,h,r,z,n,f in reversed(frames):
        dn = dh * (1 - z) * (1 - n * n)
        dz = dh * (h - n) * z * (1 - z)
        dr = dn * f * r * (1 - r)
        a = np.concatenate((dr, dz, dn), axis=1)
        b = np.concatenate((dr, dz, dn * r), axis=1)
        g[p + '_wi'] += x.T @ a
        g[p + '_bi'] += a.sum(0)
        g[p + '_wh'] += h.T @ b
        g[p + '_bh'] += b.sum(0)
        dx.append(a @ w[p + '_wi'].T)
        dh = dh * z + b @ w[p + '_wh'].T
    return np.stack(dx[::-1], axis=1)

def loss_grad(w,batch,grads=True):
    l, r, c, y = batch
    s, cache = forward(w, l, r, c, grads)
    s -= s.max(1, keepdims=True)
    prob = np.exp(s)
    prob /= prob.sum(1, keepdims=True)
    loss = float(-np.log(np.maximum(prob[np.arange(len(y)), y], 1e-30)).mean())
    if not grads:
        return loss
    cl, cr, h, p = cache
    prob[np.arange(len(y)), y] -= 1
    prob /= len(y)
    g = {k: np.zeros_like(v) for k, v in w.items()}
    np.add.at(g['emb'], c, prob[:, :, None] * p[:, None, :])
    dp = (prob[:, :, None] * w['emb'][c]).sum(1) * (1 - p * p)
    g['proj'] += h.T @ dp
    g['bias'] += dp.sum(0)
    dl, dr = np.split(dp @ w['proj'].T, 2, axis=1)
    for ctx,frames,dh,prefix in ((l,cl,dl,"l"),(r,cr,dr,"r")):
        np.add.at(g['emb'], ctx, backward_gru(w, g, prefix, frames, dh))
    g['emb'][0] = 0
    return (loss, g)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=650)
    p.add_argument('--batch', type=int, default=48)
    p.add_argument('--emb', type=int, default=24)
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out/"model.npz").exists():
        p.error('Use a fresh output directory.')
    tr = Sampler(a.data, 'train', a.seed)
    va = Sampler(a.data, 'dev', a.seed + 1)
    held = [va.batch(a.batch) for _ in range(8)]
    w = init_weights(len(tr.graph), a.emb, a.hidden, a.seed)
    m = {k: np.zeros_like(v) for k, v in w.items()}
    v = {k: x.copy() for k, x in m.items()}
    best = float('inf')
    history = []
    for step in range(1,a.steps+1):
        loss, g = loss_grad(w, tr.batch(a.batch))
        for k in w:
            m[k] = 0.9 * m[k] + 0.1 * g[k]
            v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
            w[k] -= 0.003 * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v[k] / (1 - 0.999 ** step)) + 1e-08)
        if step==1 or step%50==0 or step==a.steps:
            dev = float(np.mean([loss_grad(w, b, False) for b in held]))
            history.append({'step': step, 'train_minibatch_loss': loss, 'dev_loss': dev})
            if dev<best:
                best = dev
                save_npz(a.out/"model.npz",{**w,"resources_sha256":np.asarray(sha(a.data/"resources.json"))})
            print(f'step {step} dev {dev:.4f}', flush=True)
    write_json(a.out/"history.json",{"seed":a.seed,"batch":a.batch,"steps":a.steps,"emb":a.emb,"hidden":a.hidden,
                                     "parameters":sum(x.size for x in w.values()),"history":history})
if __name__=="__main__":
    main()

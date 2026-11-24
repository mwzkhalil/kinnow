import argparse
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sampling import Sampler
from urdu_core import save_npz, sha, write_json

class Ranker(nn.Module):
    def __init__(self,n,e,h):
        super().__init__()
        self.emb = nn.Embedding(n, e, padding_idx=0)
        nn.init.normal_(self.emb.weight, std=0.12)
        with torch.no_grad():
            self.emb.weight[0].zero_()
        self.l = nn.GRU(e, h, batch_first=True)
        self.r = nn.GRU(e, h, batch_first=True)
        self.proj = nn.Linear(h * 2, e)
    def forward(self,l,r,c):
        _, hl = self.l(self.emb(l))
        _, hr = self.r(self.emb(r))
        p = self.proj(torch.cat((hl[-1], hr[-1]), dim=1)).tanh()
        return (self.emb(c) * p[:, None, :]).sum(-1).masked_fill(c == 0, -10000.0)
    def arrays(self):
        arr = lambda a: a.detach().cpu().numpy().copy()
        w = {'emb': arr(self.emb.weight), 'proj': arr(self.proj.weight.T), 'bias': arr(self.proj.bias)}
        for k in ("l","r"):
            g = getattr(self, k)
            for name,t in (("wi",g.weight_ih_l0.T),("wh",g.weight_hh_l0.T),("bi",g.bias_ih_l0),("bh",g.bias_hh_l0)):
                w[k + '_' + name] = arr(t)
        return w

def atomic_torch(path,value):
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as fp:
        tmp = Path(fp.name)
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--steps', type=int, default=100000)
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--emb', type=int, default=64)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--seed', type=int, default=27)
    p.add_argument('--lr', type=float, default=0.002)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--schedule', choices=('constant', 'cosine'), default='cosine',
                   help='cosine decays the learning rate to 10%% of --lr by the final step')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default='auto')
    a = p.parse_args()
    if min(a.steps,a.batch,a.emb,a.hidden,a.eval_every)<=0 or a.lr<=0:
        p.error('Positive training parameters required.')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device
    if device=="cuda" and not torch.cuda.is_available():
        p.error('CUDA unavailable in this Python; verify the existing PyTorch installation.')
    a.out.mkdir(parents=True, exist_ok=True)
    # A held OS lock is released automatically after process termination.
    lock = (a.out / '.train.lock').open('a+b')
    if os.name=="nt":
        import msvcrt
        lock.seek(0)
        lock.write(b'0')
        lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cp = a.out / 'last.pt'
    if a.resume and not cp.exists():
        p.error('Resume needs last.pt.')
    if not a.resume and any((a.out/n).exists() for n in ["last.pt","model.npz"]):
        p.error('Output already contains a run; use --resume or choose a new directory.')
    manifest_hash = sha(a.data / 'manifest.json')
    res_hash = sha(a.data / 'resources.json')
    settings={"manifest_sha256":manifest_hash,"asset_sha256":res_hash,"batch":a.batch,"embedding":a.emb,"hidden":a.hidden,
              "seed":a.seed,"lr":a.lr,"eval_every":a.eval_every}
    if a.schedule!="constant":  # constant LR is not recorded, so older checkpoints still resume
        settings["schedule"] = a.schedule
    torch.manual_seed(a.seed)
    if device=="cuda":
        torch.cuda.manual_seed_all(a.seed)
    tr = Sampler(a.data, 'train', a.seed)
    va = Sampler(a.data, 'dev', a.seed + 1)
    valid = [va.batch(a.batch) for _ in range(32)]
    model = Ranker(len(tr.graph), a.emb, a.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    step = 0
    best = math.inf
    best_step = 0
    history = []
    if a.resume:
        ck = torch.load(cp, map_location='cpu', weights_only=True)
        if ck["settings"]!=settings or ck["device"]!=device:
            raise ValueError('Resume configuration or dataset changed.')
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        tr.rng.bit_generator.state = json.loads(ck['sampler'])
        torch.set_rng_state(ck['torch_rng'])
        if device=="cuda":
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
        step, best_step, best, history = (ck['step'], ck['best_step'], ck['best'], ck['history'])
        # The checkpoint is authoritative even after an interruption between file writes.
        best_weights = ck['best_weights']
        save_npz(a.out/"model.npz",{**{k:v.numpy() for k,v in best_weights.items()},"resources_sha256":np.asarray(res_hash)})
    if a.steps<step:
        p.error('Requested total steps are below the last checkpoint.')
    shutil.copyfile(a.data / 'resources.json', a.out / 'resources.json')
    env={"torch":str(torch.__version__),"numpy":str(np.__version__),"device":device,
         "cuda":torch.version.cuda,"hardware":torch.cuda.get_device_name() if device=="cuda" else "CPU"}
    write_json(a.out/"run.json",{"config":settings,"runtime":env,"target_steps":a.steps,"parameters":sum(v.numel() for v in model.parameters())})
    def loss(batch):
        l, r, c, y = [torch.from_numpy(v).to(device) for v in batch]
        return F.cross_entropy(model(l, r, c), y)
    for step in range(step+1,a.steps+1):
        model.train()
        if a.schedule=="cosine":
            for g in opt.param_groups:
                g["lr"] = a.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * step / a.steps)))
        opt.zero_grad(set_to_none=True)
        z = loss(tr.batch(a.batch))
        if not torch.isfinite(z):
            raise FloatingPointError('Nonfinite training loss')
        z.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 3.0, error_if_nonfinite=True)
        opt.step()
        if step%a.eval_every==0 or step==a.steps or step==1:
            model.eval()
            with torch.inference_mode():
                v = float(np.mean([loss(b).item() for b in valid]))
            if not math.isfinite(v):
                raise FloatingPointError('Nonfinite validation loss')
            history.append({'step': step, 'train_loss': z.item(), 'dev_loss': v})
            if v<best:
                best, best_step = (v, step)
                best_weights = {k: torch.from_numpy(val) for k, val in model.arrays().items()}
            ck={"settings":settings,"device":device,"step":step,"best":best,"best_step":best_step,"history":history,
                "model":model.state_dict(),"opt":opt.state_dict(),"best_weights":best_weights,"sampler":json.dumps(tr.rng.bit_generator.state),
                "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if device=="cuda" else []}
            atomic_torch(cp, ck)
            save_npz(a.out/"model.npz",{**{k:v.numpy() for k,v in best_weights.items()},"resources_sha256":np.asarray(res_hash)})
            write_json(a.out / 'history.json', {'history': history, 'best_step': best_step})
            print(f'step {step} train {z.item():.4f} dev {v:.4f} best {best_step}', flush=True)
if __name__=="__main__":
    main()

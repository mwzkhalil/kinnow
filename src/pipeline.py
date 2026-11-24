"""Train, export and evaluate an Urdu ranker in a resumable run directory."""
import argparse
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urdu_core import read_json, sha, write_json
from prepare import BASE, CHECKSUM

def call(script, *args):
    cmd = [sys.executable, str(BASE / "src" / script), *map(str, args)]
    print('+', subprocess.list2cmdline(cmd), flush=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '1'))
    subprocess.run(cmd, check=True, env=env)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, default=Path('work/wiki'))
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--steps', type=int, default=100000)
    p.add_argument('--schedule', choices=('constant', 'cosine'), default='cosine', help='learning-rate schedule passed to train.py')
    p.add_argument('--extra-jsonl', type=Path, action='append', default=[])
    p.add_argument('--local-only', action='store_true', help='Use only documents passed with --extra-jsonl.')
    p.add_argument('--human-dev', type=Path)
    p.add_argument('--human-test', type=Path)
    p.add_argument('--groups', type=Path, default=BASE / 'data/confusions.tsv')
    p.add_argument('--protect', type=Path, default=BASE / 'data/protect.txt')
    p.add_argument("--eval-sentences",type=int,default=5000,help="Maximum held-out clean sentences per synthetic split.")
    a = p.parse_args()
    if bool(a.human_dev)!=bool(a.human_test):
        p.error('Pass both human development and test files.')
    if a.local_only and not a.extra_jsonl:
        p.error('--local-only needs --extra-jsonl.')
    if min(a.batch,a.steps,a.eval_sentences)<1:
        p.error('Steps, batch and evaluation size must be positive.')
    if sys.version_info<(3,10):
        p.error('Python 3.10 or newer is required.')
    for lib in (["torch","onnx","onnxruntime"]+([] if a.local_only else ["pyarrow"])):
        try:
            importlib.import_module(lib)
        except ImportError as exc:
            p.error(f'Missing {lib} ({exc}); see README setup.')
    import torch
    import platform
    from importlib.metadata import version
    if a.device=="cuda" and not torch.cuda.is_available():
        p.error('CUDA is unavailable in the active PyTorch installation.')
    root = a.work.resolve()
    root.mkdir(parents=True, exist_ok=True)
    # A lock covers preparation, export, calibration and evaluation as well as training.
    guard = (root / '.pipeline.lock').open('a+b')
    if os.name=="nt":
        import msvcrt
        guard.seek(0)
        guard.write(b'0')
        guard.flush()
        guard.seek(0)
        msvcrt.locking(guard.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    packages = ['torch', 'onnx', 'onnxruntime', 'numpy'] + ([] if a.local_only else ['pyarrow'])
    write_json(root/"versions.json",{"python":platform.python_version(),"platform":platform.platform(),
                                   "packages":{name:version(name) for name in packages}})
    data = root / 'data'
    model = root / 'model'
    graphs = root / 'onnx'
    configuration = root / 'inputs.json'
    config={"wiki_sha256":None if a.local_only else CHECKSUM,"extra_jsonl":[[x.name,sha(x)] for x in a.extra_jsonl],
            "groups":sha(a.groups),"protected":sha(a.protect)}
    if configuration.exists() and read_json(configuration)!=config:
        p.error('Corpus inputs changed. Choose a new --work directory.')
    if not configuration.exists() and data.exists():
        p.error('A corpus without a matching run manifest exists. Use a new --work.')
    write_json(configuration, config)
    if not data.exists():
        args = ['--cache', root / 'cache', '--out', data, '--groups', a.groups, '--protect', a.protect]
        if not a.local_only:
            args.append('--wiki')
        for item in a.extra_jsonl:
            args += ['--jsonl', item]
        call('prepare.py', *args)
    if not (data/"manifest.json").is_file():
        p.error('Incomplete corpus. Choose a new --work directory.')
    res = model / 'resources.json'
    native = model / 'model.npz'
    train_args=["--data",data,"--out",model,"--batch",a.batch,"--steps",a.steps,"--device",a.device,"--schedule",a.schedule]
    if (model/"last.pt").is_file():
        train_args.append('--resume')
    call('train.py', *train_args)
    call('export_onnx.py', '--data', data, '--model', native, '--resources', res, '--out', graphs)
    if a.human_dev:
        rows = {'dev': a.human_dev, 'test': a.human_test}
        label = 'human'
    else:
        diag = root / ('synthetic-' + str(a.eval_sentences))
        if not diag.exists():
            tmp = Path(tempfile.mkdtemp(prefix='diagnostic-', dir=root))
            try:
                call('make_eval.py', '--data', data, '--out', tmp, '--limit', a.eval_sentences)
                tmp.rename(diag)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        rows = {k: diag / (k + '.jsonl') for k in ('dev', 'test')}
        label = 'synthetic'
    results = {}
    for tag,path in (("fp32",graphs/"model.fp32.onnx"),("int8",graphs/"model.int8.onnx")):
        out = root / 'reports' / label / tag
        shared = ['--model', path, '--resources', res, '--data', data, '--policy', out / 'policy.json']
        call('evaluate.py', 'calibrate', *shared, '--rows', rows['dev'], '--report', out / 'dev.json')
        call('evaluate.py', 'test', *shared, '--rows', rows['test'], '--report', out / 'test.json')
        results[tag] = read_json(out / 'test.json')['meets_proposed_release_checks']
    print('Reports:', root / 'reports' / label, flush=True)
    for tag,passed in results.items():
        print(f'{tag}: proposed human release checks passed = {passed}')
if __name__=="__main__":
    main()

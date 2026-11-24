"""Create a portable inference directory tied to one model, policy and test report."""
import argparse
from pathlib import Path
import shutil
import tempfile
from urdu_core import read_json, sha, write_json

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--format', choices=['fp32', 'int8'], default='int8')
    p.add_argument('--evaluation', choices=['synthetic', 'human'], default='synthetic')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument("--require-pass",action="store_true",help="Only package a model that meets the proposed human release checks.")
    a = p.parse_args()
    source = a.work / 'onnx' / ('model.' + a.format + '.onnx')
    res = a.work / 'model/resources.json'
    reports = a.work / 'reports' / a.evaluation / a.format
    policy = read_json(reports / 'policy.json')
    test = read_json(reports / 'test.json')
    dev = read_json(reports / 'dev.json')
    if not (sha(source)==test["model_sha256"]==policy["model_sha256"]==dev["model_sha256"] and
            test["threshold"]==policy.get("thresholds", policy["threshold"])==dev["threshold"] and
            policy["dev_sha256"]==dev["rows_sha256"] and sha(res)==policy["resources_sha256"]):
        raise ValueError('Model, policy or reports do not match. Re-run evaluation.')
    passed = test['meets_proposed_release_checks'] is True
    if a.require_pass and not passed:
        p.error('This candidate did not pass the proposed human release checks. See test.json.')
    if a.out.exists():
        p.error('Select a new output directory.')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(dir=a.out.parent, prefix='bundle-'))
    try:
        shutil.copyfile(source, stage / 'model.onnx')
        shutil.copyfile(res, stage / 'resources.json')
        shutil.copyfile(reports / 'test.json', stage / 'test.json')
        write_json(stage/"policy.json",{k:policy[k] for k in ("model_sha256","resources_sha256","threshold","thresholds") if k in policy})
        manifest={"project":"kinnow","format":a.format,"evaluation":a.evaluation,"human_release_checks_passed":passed,
                  "files":{f.name:sha(f) for f in sorted(stage.iterdir())}}
        write_json(stage / 'manifest.json', manifest)
        stage.rename(a.out)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    print(a.out, 'human release checks passed =', passed, flush=True)
if __name__=="__main__":
    main()

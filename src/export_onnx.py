"""Export the trained ranker and compare both CPU graphs with held-out NumPy logits."""
import argparse
import os
from pathlib import Path
import tempfile
import numpy as np
from sampling import Sampler
from urdu_core import Fixer, LEFT, RIGHT, WIDTH, write_json

def graph(weights, quantized):
    nodes, tensors = ([], {})
    def op(kind, inputs, outputs, **attributes):
        nodes.append((kind, inputs, outputs, attributes))
    def parameter(name, value, axis):
        value = np.ascontiguousarray(value, dtype=np.float32)
        if not quantized:
            tensors[name] = value
            return
        dims = tuple((i for i in range(value.ndim) if i != axis))
        maximum = np.max(np.abs(value), axis=dims, keepdims=True)
        scale = np.where(maximum > 0, maximum / 127, np.float32(1)).astype(np.float32)
        tensors[name + '_q'] = np.rint(value / scale).clip(-127, 127).astype(np.int8)
        tensors[name + '_scale'] = scale.reshape(-1)
        tensors[name + '_zero'] = np.zeros(scale.size, dtype=np.int8)
        op('DequantizeLinear', [name + '_q', name + '_scale', name + '_zero'], [name], axis=axis)

    parameter('embedding', weights['emb'], 0)
    tensors['axis0'] = np.array([0], np.int64)
    tensors['axis1'] = np.array([1], np.int64)
    tensors['last_axis'] = np.array([-1], np.int64)
    tensors['pad'] = np.array(0, np.int64)
    tensors['masked_logit'] = np.array(-10000.0, np.float32)
    hidden = weights['l_wh'].shape[0]
    gru_order = np.r_[hidden:2 * hidden, 0:hidden, 2 * hidden:3 * hidden]
    for side, tag in (("left", "l"), ("right", "r")):
        for short, source in (("W", "wi"), ("R", "wh")):
            matrix = weights[tag + '_' + source][:, gru_order].T
            parameter(tag + short, matrix[None, :, :], 1)
        tensors[tag + "B"] = np.concatenate([weights[tag + "_" + n][gru_order]
                                             for n in ("bi","bh")])[None, :]
        op('Gather', ['embedding', side], [tag + 'vec'], axis=0)
        op('Transpose', [tag + 'vec'], [tag + 'seq'], perm=[1, 0, 2])
        op("GRU", [tag + "seq", tag+"W", tag+"R", tag+"B"], ["",tag+"last"],
           hidden_size=hidden, linear_before_reset=1)
        op('Squeeze', [tag + 'last', 'axis0'], [tag + 'state'])
    parameter('project', weights['proj'], 1)
    tensors['bias'] = weights['bias']
    op('Concat', ['lstate', 'rstate'], ['context'], axis=1)
    op('MatMul', ['context', 'project'], ['projected'])
    op('Add', ['projected', 'bias'], ['biased'])
    op('Tanh', ['biased'], ['query'])
    op('Unsqueeze', ['query', 'axis1'], ['query_axis'])
    op('Gather', ['embedding', 'candidates'], ['candidate_vectors'], axis=0)
    op('Mul', ['candidate_vectors', 'query_axis'], ['products'])
    op('ReduceSum', ['products', 'last_axis'], ['raw_scores'], keepdims=0)
    op('Equal', ['candidates', 'pad'], ['mask'])
    op('Where', ['mask', 'masked_logit', 'raw_scores'], ['scores'])
    return (nodes, tensors)

def export(path, source, quantized):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    specs, tensors = graph(source.w, quantized)
    nodes = [helper.make_node(kind, ins, outs, **attrs) for kind, ins, outs, attrs in specs]
    def info(name, dtype, width):
        return helper.make_tensor_value_info(name, dtype, ['batch', width])
    inputs = [info(k,TensorProto.INT64,n) for k,n in (("left",LEFT),("right",RIGHT),("candidates",WIDTH))]
    outputs = [info('scores', TensorProto.FLOAT, WIDTH)]
    initializer = [numpy_helper.from_array(a, name=k) for k, a in tensors.items()]
    model = helper.make_model(helper.make_graph(nodes,"Kinnow",inputs,outputs,initializer),
                              producer_name="kinnow",opset_imports=[helper.make_opsetid("",13)])
    model.ir_version = 9
    helper.set_model_props(model, {"resources_sha256":source.asset, "source_sha256":source.fingerprint,
                                   "weight_format":"int8-per-channel" if quantized else "float32"})
    onnx.checker.check_model(model, full_check=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
        temp = Path(f.name)
    try:
        onnx.save(model, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)

def parity(reference, runner, sampler, batches=8, batch_size=64):
    max_error = 0.0
    same = total = 0
    for _ in range(batches):
        left, right, candidates, _ = sampler.batch(batch_size)
        truth = reference.score(left, right, candidates)
        actual = runner.score(left, right, candidates)
        if not np.isfinite(actual).all():
            raise ValueError('Nonfinite exported output')
        max_error = max(max_error, float(np.abs(actual - truth).max()))
        same += int((actual.argmax(1) == truth.argmax(1)).sum())
        total += len(left)
    return {'examples': total, 'max_abs_logit_error': max_error, 'top1_agreement': same / total}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("model","resources","data","out"):
        p.add_argument('--' + flag, type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    source = Fixer(a.model, a.resources)
    results = {}
    for key in ("fp32","int8"):
        path = a.out / ('model.' + key + '.onnx')
        export(path, source, key == 'int8')
        runner = Fixer(path, a.resources)
        report = parity(source, runner, Sampler(a.data, 'dev', 106))
        report.update({'bytes': path.stat().st_size, 'sha256': runner.fingerprint})
        results[key] = report
    results['fp32_within_tolerance'] = results['fp32']['max_abs_logit_error'] <= 0.0001
    write_json(a.out / 'parity.json', results)
    print(results, flush=True)
    if not results["fp32_within_tolerance"]:
        raise SystemExit('FP32 parity failed; inspect parity.json.')
if __name__=="__main__":
    main()

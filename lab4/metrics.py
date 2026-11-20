import math
import time
import numpy as np
import torch
from sklearn.metrics import f1_score

def run_pytorch_inference(model, loader):
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            logits = model(inputs)
            all_logits.append(logits.cpu().numpy().astype(np.float32))
            all_targets.append(targets.cpu().numpy().astype(np.int64))

    return np.concatenate(all_logits, axis = 0), np.concatenate(all_targets, axis = 0)

def run_onnx_inference(session, loader):
    all_logits = []

    for inputs, _ in loader:
        batch_np = inputs.numpy().astype(np.float32)
        logits = session.run(['logits'], {'input': batch_np})[0]
        all_logits.append(logits.astype(np.float32))

    return np.concatenate(all_logits, axis = 0)

def compute_export_parity(pt_logits, onnx_logits, targets):
    max_abs_diff = float(np.max(np.abs(onnx_logits - pt_logits)))
    pt_preds = np.argmax(pt_logits, axis = 1)
    onnx_preds = np.argmax(onnx_logits, axis = 1)
    argmax_agreement = float(np.mean(pt_preds == onnx_preds))
    pt_macro_f1 = float(f1_score(targets, pt_preds, average = 'macro'))
    onnx_macro_f1 = float(f1_score(targets, onnx_preds, average = 'macro'))

    return {
        'max_abs_logit_diff': max_abs_diff,
        'argmax_agreement': argmax_agreement,
        'pytorch_macro_f1': pt_macro_f1,
        'onnx_fp32_macro_f1': onnx_macro_f1
    }

def compute_quality_metrics(variant_logits, fp32_logits, targets):
    n = int(len(targets))
    fp32_preds = np.argmax(fp32_logits, axis = 1)
    var_preds = np.argmax(variant_logits, axis = 1)

    macro_f1 = float(f1_score(targets, var_preds, average = 'macro'))
    accuracy = float(np.mean(var_preds == targets))
    agreement_with_fp32 = float(np.mean(var_preds == fp32_preds))

    fp32_correct = fp32_preds == targets
    var_correct = var_preds == targets
    b = int(np.sum(var_correct & (~fp32_correct)))
    c = int(np.sum((~var_correct) & fp32_correct))

    delta_acc = float((b - c) / n)
    se_delta = float(math.sqrt(b + c) / n)
    noise_bound = float(2.0 * math.sqrt(b + c))
    distinguishable = bool(abs(b - c) > noise_bound)

    if distinguishable:
        direction = 'higher' if b > c else 'lower'
    else:
        direction = None

    return {
        'macro_f1': macro_f1,
        'accuracy': accuracy,
        'agreement_with_fp32': agreement_with_fp32,
        'b': b,
        'c': c,
        'delta_acc': delta_acc,
        'se_delta': se_delta,
        'distinguishable_from_noise': distinguishable,
        'direction': direction
    }

def compute_single_model_se(accuracy, n):
    return float(math.sqrt(accuracy * (1.0 - accuracy) / n))

def measure_one_round(session, feed, warmup, runs):
    for _ in range(warmup):
        session.run(None, feed)

    timings_s = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, feed)
        t1 = time.perf_counter()
        timings_s.append(t1 - t0)

    return [t * 1000.0 for t in timings_s]

def summarize_latency(rounds_ms):
    all_ms = [val for r in rounds_ms for val in r]
    round_p50 = [float(np.percentile(r, 50)) for r in rounds_ms]

    return {
        'p50_ms': float(np.percentile(all_ms, 50)),
        'p95_ms': float(np.percentile(all_ms, 95)),
        'round_p50_ms': round_p50,
        'measurements_ms': all_ms
    }

def benchmark_latency(fp32_session, int8_session, sample_input, rounds = 3, warmup = 20, runs = 100):
    feed = {'input': sample_input}
    fp32_rounds = []
    int8_rounds = []

    for _ in range(rounds):
        fp32_rounds.append(measure_one_round(fp32_session, feed, warmup, runs))
        int8_rounds.append(measure_one_round(int8_session, feed, warmup, runs))

    fp32_summary = summarize_latency(fp32_rounds)
    int8_summary = summarize_latency(int8_rounds)

    fp32_r = fp32_summary['round_p50_ms']
    int8_r = int8_summary['round_p50_ms']
    intervals_disjoint = bool(max(fp32_r) < min(int8_r) or max(int8_r) < min(fp32_r))

    speedup_p50 = float(fp32_summary['p50_ms'] / int8_summary['p50_ms'])
    speedup_p95 = float(fp32_summary['p95_ms'] / int8_summary['p95_ms'])

    return {
        'fp32': fp32_summary,
        'int8': int8_summary,
        'intervals_disjoint': intervals_disjoint,
        'speedup_p50': speedup_p50,
        'speedup_p95': speedup_p95
    }

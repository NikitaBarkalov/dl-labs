import argparse
import json
import os
import platform
import subprocess
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader

from data import (
    SplitImageDataset,
    get_eval_transform,
    load_and_verify_split,
    load_image_tensor,
    resolve_image_path,
    select_calibration_rows
)
from metrics import (
    benchmark_latency,
    compute_export_parity,
    compute_quality_metrics,
    compute_single_model_se,
    run_onnx_inference,
    run_pytorch_inference
)
from onnx_utils import (
    create_cpu_session,
    export_to_onnx,
    inspect_onnx_graph,
    load_pytorch_model,
    preprocess_onnx_for_quant,
    quantize_onnx_static
)

def get_cpu_model_name():
    system = platform.system()
    if system == 'Windows':
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r'HARDWARE\DESCRIPTION\System\CentralProcessor\0'
            )
            name, _ = winreg.QueryValueEx(key, 'ProcessorNameString')
            winreg.CloseKey(key)
            return name.strip()
        except Exception:
            return platform.processor()

    if system == 'Darwin':
        try:
            out = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text = True)
            return out.strip()
        except Exception:
            return platform.processor()

    if system == 'Linux':
        try:
            with open('/proc/cpuinfo', 'r', encoding = 'utf-8') as f:
                for line in f:
                    if line.startswith('model name'):
                        return line.split(':', 1)[1].strip()
        except Exception:
            return platform.processor()

    return platform.processor()

def get_environment_info(session, sample_input):
    sess_options = session.get_session_options()
    return {
        'cpu_model': get_cpu_model_name(),
        'os': platform.platform(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'onnx_version': onnx.__version__,
        'onnxruntime_version': ort.__version__,
        'intra_op_num_threads': int(sess_options.intra_op_num_threads),
        'inter_op_num_threads': int(sess_options.inter_op_num_threads),
        'batch_size': int(sample_input.shape[0])
    }

def get_file_size_mb(path):
    return float(os.path.getsize(path) / 1_000_000.0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, required = True)
    parser.add_argument('--split', type = str, required = True)
    parser.add_argument('--data-dir', type = str, required = True)
    parser.add_argument('--out', type = str, required = True)
    parser.add_argument('--onnx-dir', type = str, required = True)
    args = parser.parse_args()

    model_path = Path(args.model)
    split_path = Path(args.split)
    data_dir = Path(args.data_dir)
    out_json_path = Path(args.out)
    onnx_dir = Path(args.onnx_dir)

    if not model_path.exists():
        raise FileNotFoundError(f'Model file not found: {model_path}')
    if not split_path.exists():
        raise FileNotFoundError(f'Split file not found: {split_path}')
    if not data_dir.exists():
        raise FileNotFoundError(f'Data directory not found: {data_dir}')
    if out_json_path.exists():
        raise FileExistsError(f'Output file already exists: {args.out}')
    if onnx_dir.exists():
        raise FileExistsError(f'Output directory already exists: {args.onnx_dir}')

    onnx_dir.mkdir(parents = True)
    out_json_path.parent.mkdir(parents = True, exist_ok = True)

    pt_model, img_size = load_pytorch_model(model_path)
    eval_tf = get_eval_transform(img_size)

    train_rows, test_rows, split_control = load_and_verify_split(split_path)
    calib_rows, calib_control = select_calibration_rows(train_rows)

    test_ds = SplitImageDataset(test_rows, data_dir, eval_tf)
    test_loader = DataLoader(
        test_ds,
        batch_size = 64,
        shuffle = False,
        drop_last = False,
        num_workers = 0
    )

    fp32_path = onnx_dir / 'lab4_fp32.onnx'
    prep_path = onnx_dir / 'lab4_fp32_prep.onnx'
    int8_path = onnx_dir / 'lab4_int8.onnx'
    int8_pt_path = onnx_dir / 'lab4_int8_per_tensor.onnx'

    export_time_s = export_to_onnx(pt_model, img_size, fp32_path)
    graph_info = inspect_onnx_graph(fp32_path)

    fp32_session = create_cpu_session(fp32_path)
    pt_logits, targets = run_pytorch_inference(pt_model, test_loader)
    fp32_logits = run_onnx_inference(fp32_session, test_loader)

    export_parity = compute_export_parity(pt_logits, fp32_logits, targets)

    preprocess_time_s = preprocess_onnx_for_quant(fp32_path, prep_path)
    int8_quant_time_s = quantize_onnx_static(
        prep_path = prep_path,
        int8_path = int8_path,
        calib_rows = calib_rows,
        data_dir = data_dir,
        transform = eval_tf,
        per_channel = True
    )

    int8_session = create_cpu_session(int8_path)
    int8_logits = run_onnx_inference(int8_session, test_loader)

    fp32_quality = compute_quality_metrics(fp32_logits, fp32_logits, targets)
    int8_quality = compute_quality_metrics(int8_logits, fp32_logits, targets)
    fp32_single_model_se = compute_single_model_se(fp32_quality['accuracy'], len(targets))

    quantize_onnx_static(
        prep_path = prep_path,
        int8_path = int8_pt_path,
        calib_rows = calib_rows,
        data_dir = data_dir,
        transform = eval_tf,
        per_channel = False
    )

    int8_pt_session = create_cpu_session(int8_pt_path)
    int8_pt_logits = run_onnx_inference(int8_pt_session, test_loader)
    int8_pt_quality = compute_quality_metrics(int8_pt_logits, fp32_logits, targets)

    first_test_row = test_rows[0]
    first_test_path = resolve_image_path(
        data_dir = data_dir,
        part = first_test_row['part'],
        label = first_test_row['label'],
        filename = first_test_row['file']
    )
    first_tensor = load_image_tensor(first_test_path, eval_tf)
    sample_input = first_tensor.unsqueeze(0).numpy()
    env_info = get_environment_info(fp32_session, sample_input)

    latency_results = benchmark_latency(
        fp32_session = fp32_session,
        int8_session = int8_session,
        sample_input = sample_input,
        rounds = 3,
        warmup = 20,
        runs = 100
    )

    fp32_size_mb = get_file_size_mb(fp32_path)
    int8_size_mb = get_file_size_mb(int8_path)
    size_ratio = float(fp32_size_mb / int8_size_mb)

    results = {
        'environment': env_info,
        'split_control': split_control,
        'calibration_control': calib_control,
        'export_and_parity': {
            'export_time_s': export_time_s,
            'max_abs_logit_diff': export_parity['max_abs_logit_diff'],
            'argmax_agreement': export_parity['argmax_agreement'],
            'pytorch_macro_f1': export_parity['pytorch_macro_f1'],
            'onnx_fp32_macro_f1': export_parity['onnx_fp32_macro_f1'],
            'conv_nodes': graph_info['conv_count'],
            'batchnorm_nodes': graph_info['batchnorm_count']
        },
        'quantization': {
            'preprocess_time_s': preprocess_time_s,
            'int8_quant_time_s': int8_quant_time_s
        },
        'quality': {
            'fp32': fp32_quality,
            'int8': int8_quality,
            'int8_per_tensor': int8_pt_quality,
            'fp32_single_model_se': fp32_single_model_se
        },
        'per_tensor_experiment': {
            'top3_convs': graph_info['top3_convs'],
            'median_r': graph_info['median_r'],
            'median_bits_cost': graph_info['median_bits_cost']
        },
        'file_sizes': {
            'fp32_mb': fp32_size_mb,
            'int8_mb': int8_size_mb,
            'fp32_to_int8_ratio': size_ratio
        },
        'latency': latency_results
    }

    with open(out_json_path, 'w', encoding = 'utf-8') as f:
        json.dump(results, f, indent = 2, ensure_ascii = False)

    print('split control:')
    print(f'train: {split_control['train']['count']} / {split_control['train']['name_sum']}')
    print(f'test:  {split_control['test']['count']} / {split_control['test']['name_sum']}')
    print()

    print('calibration control:')
    print(f'count: {calib_control['count']}, sum: {calib_control['name_sum']}')
    print(f'classes: {calib_control['class_counts']}')
    print()

    print('export and parity:')
    print(f'export_time_s: {export_time_s:.4f}')
    print(f'max_abs_logit_diff: {export_parity['max_abs_logit_diff']:.6e}')
    print(f'argmax_agreement: {export_parity['argmax_agreement']:.4f}')
    print(f'pytorch_macro_f1: {export_parity['pytorch_macro_f1']:.4f}')
    print(f'onnx_fp32_macro_f1: {export_parity['onnx_fp32_macro_f1']:.4f}')
    print(f'conv_nodes: {graph_info['conv_count']}, batchnorm_nodes: {graph_info['batchnorm_count']}')
    print()

    print('quantization times:')
    print(f'preprocess_time_s: {preprocess_time_s:.4f}')
    print(f'int8_quant_time_s: {int8_quant_time_s:.4f}')
    print()

    print('quality:')
    for label_name, q in [
        ('FP32', fp32_quality),
        ('INT8', int8_quality),
        ('INT8_per_tensor', int8_pt_quality)
    ]:
        print(f'{label_name}:')
        print(f'  macro_f1: {q['macro_f1']:.4f}')
        print(f'  accuracy: {q['accuracy']:.4f}')
        print(f'  agreement_with_fp32: {q['agreement_with_fp32']:.4f}')
        print(f'  b: {q['b']}, c: {q['c']}')
        print(f'  delta_acc: {q['delta_acc']:+.4f} +/- {q['se_delta']:.4f}')
        print(f'  distinguishable_from_noise: {q['distinguishable_from_noise']}')
        if q['direction'] is not None:
            print(f'  direction: {q['direction']}')
    print(f'fp32_single_model_se: {fp32_single_model_se:.4f}')
    print()

    print('channel ranges:')
    for item in graph_info['top3_convs']:
        print(f'{item['node_name']}: r = {item['r']:.4f}, bits = {item['bits_cost']:.4f}')
    print(f'median_r: {graph_info['median_r']:.4f}')
    print(f'median_bits_cost: {graph_info['median_bits_cost']:.4f}')
    print()

    print('latency and file size:')
    fp32_lat = latency_results['fp32']
    int8_lat = latency_results['int8']
    for label_name, size_mb, lat in [
        ('FP32', fp32_size_mb, fp32_lat),
        ('INT8', int8_size_mb, int8_lat)
    ]:
        print(f'{label_name}:')
        print(f'  file_size_mb: {size_mb:.2f}')
        print(f'  p50_ms: {lat['p50_ms']:.2f}')
        print(f'  p95_ms: {lat['p95_ms']:.2f}')
        print(f'  rounds_p50_ms: {[round(x, 2) for x in lat['round_p50_ms']]}')
    print(f'size_ratio: {size_ratio:.2f}x')
    print(f'speedup_p50: {latency_results['speedup_p50']:.2f}x')
    print(f'speedup_p95: {latency_results['speedup_p95']:.2f}x')
    print(f'intervals_disjoint: {latency_results['intervals_disjoint']}')

if __name__ == '__main__':
    main()

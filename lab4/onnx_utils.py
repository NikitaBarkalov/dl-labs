import math
import time
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision
from onnxruntime.quantization import (
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quant_pre_process,
    quantize_static
)
from data import CLASSES, ImageCalibrationDataReader

def load_pytorch_model(model_path):
    ckpt = torch.load(model_path, map_location = 'cpu', weights_only = True)
    classes = list(ckpt['classes'])
    if classes != CLASSES:
        raise ValueError(f'Classes mismatch: {classes}')

    model = torchvision.models.resnet18(weights = None)
    model.fc = torch.nn.Linear(512, 6)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    return model, int(ckpt['img_size'])

def export_to_onnx(model, img_size, out_path):
    dummy_input = torch.randn(1, 3, img_size, img_size, dtype = torch.float32)

    t0 = time.perf_counter()
    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        dynamo = False,
        opset_version = 17,
        input_names = ['input'],
        output_names = ['logits'],
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )
    export_time_s = time.perf_counter() - t0

    return float(export_time_s)

def inspect_onnx_graph(onnx_path):
    onnx_model = onnx.load(str(onnx_path))
    graph = onnx_model.graph

    conv_nodes = [node for node in graph.node if node.op_type == 'Conv']
    bn_nodes = [node for node in graph.node if node.op_type == 'BatchNormalization']
    initializers = {init.name: init for init in graph.initializer}

    conv_stats = []
    for node in conv_nodes:
        weight_name = node.input[1]
        weights = onnx.numpy_helper.to_array(initializers[weight_name])
        alpha_c = np.max(np.abs(weights), axis = (1, 2, 3))
        max_alpha = float(np.max(alpha_c))
        median_alpha = float(np.median(alpha_c))
        ratio_r = float(max_alpha / median_alpha)
        bits_cost = float(math.log2(ratio_r))

        conv_stats.append({
            'node_name': node.name,
            'r': ratio_r,
            'bits_cost': bits_cost
        })

    sorted_convs = sorted(conv_stats, key = lambda item: item['r'], reverse = True)
    median_r = float(np.median([item['r'] for item in conv_stats]))
    median_bits_cost = float(math.log2(median_r))

    return {
        'conv_count': len(conv_nodes),
        'batchnorm_count': len(bn_nodes),
        'top3_convs': sorted_convs[:3],
        'median_r': median_r,
        'median_bits_cost': median_bits_cost
    }

def create_cpu_session(onnx_path):
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1

    return ort.InferenceSession(
        str(onnx_path),
        sess_options = sess_options,
        providers = ['CPUExecutionProvider']
    )

def preprocess_onnx_for_quant(fp32_path, prep_path):
    t0 = time.perf_counter()
    quant_pre_process(
        str(fp32_path),
        str(prep_path),
        skip_symbolic_shape = True
    )
    return float(time.perf_counter() - t0)

def quantize_onnx_static(prep_path, int8_path, calib_rows, data_dir, transform, per_channel = True):
    reader = ImageCalibrationDataReader(
        calib_rows = calib_rows,
        data_dir = data_dir,
        transform = transform,
        input_name = 'input'
    )

    t0 = time.perf_counter()
    quantize_static(
        model_input = str(prep_path),
        model_output = str(int8_path),
        calibration_data_reader = reader,
        quant_format = QuantFormat.QDQ,
        activation_type = QuantType.QUInt8,
        weight_type = QuantType.QInt8,
        per_channel = per_channel,
        calibrate_method = CalibrationMethod.MinMax
    )
    return float(time.perf_counter() - t0)

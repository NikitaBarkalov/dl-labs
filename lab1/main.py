import numpy as np
import torch
import torch.nn as nn
import argparse
from data.preprocess import load_data, split_data, standardize_data
from model.weights import generate_weights
from model.custom import CustomCrossEntropyLoss, CustomSequential, CustomLinear, CustomReLU
from grads.grads import run_backward, compare, compare_numbers

torch.set_default_dtype(torch.float64)

SPLIT_PROPORTION = 0.7
GENERATOR_SEED = 0
HIDDEN_LAYER_SIZE = 8
EPSILON = 1e-6
MAX_DELTA_GRAD = 1e-12
MAX_DELTA_NUMERIC = 1e-7

def numeric_grad(model, loss, step, train_data, train_targets, component, idx):
    original_value = component.params[idx].copy()

    component.params[idx] = original_value + step
    right_logits = model(train_data)
    right_loss = loss(right_logits, train_targets).item()

    component.params[idx] = original_value - step
    left_logits = model(train_data)
    left_loss = loss(left_logits, train_targets).item()

    component.params[idx] = original_value

    return (right_loss - left_loss) / (2 * step)

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--with-error', action = 'store_true')
    args = argparser.parse_args()
    with_error = args.with_error
    if with_error:
        print(f'Running with a special error in gradient')

    split_generator = np.random.default_rng(GENERATOR_SEED)
    data, target = load_data()

    train_data, train_target, test_data, test_target = split_data(data, target, SPLIT_PROPORTION, split_generator)
    standardized_train_data, standardized_test_data = standardize_data(train_data, test_data)
    torched_standardized_train_data = torch.from_numpy(standardized_train_data)
    torched_train_target = torch.from_numpy(train_target)

    classes_count = len(np.unique(target))
    weights_generator = np.random.default_rng(GENERATOR_SEED)

    w1 = generate_weights(train_data.shape[1], HIDDEN_LAYER_SIZE, 'He', weights_generator)
    b1 = np.zeros(HIDDEN_LAYER_SIZE)
    w2 = generate_weights(HIDDEN_LAYER_SIZE, classes_count, 'Xavier', weights_generator)
    b2 = np.zeros(classes_count)

    custom_model = CustomSequential(
        CustomLinear(w1, b1),
        CustomReLU(),
        CustomLinear(w2, b2)
    )

    custom_criterion = CustomCrossEntropyLoss() if not with_error else CustomCrossEntropyLoss(False)

    torch_model = nn.Sequential(
        nn.Linear(train_data.shape[1], HIDDEN_LAYER_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_LAYER_SIZE, classes_count)
    )

    with torch.no_grad():
        torch_model[0].weight.copy_(torch.from_numpy(w1.T))
        torch_model[0].bias.copy_(torch.from_numpy(b1))

        torch_model[2].weight.copy_(torch.from_numpy(w2.T))
        torch_model[2].bias.copy_(torch.from_numpy(b2))

    torch_criterion = nn.CrossEntropyLoss()

    loss_custom, w1_grad_custom, \
        b1_grad_custom, w2_grad_custom, b2_grad_custom = run_backward(custom_model, custom_criterion, standardized_train_data, train_target)
    loss_torch, w1_grad_torch, \
        b1_grad_torch, w2_grad_torch, b2_grad_torch = run_backward(torch_model, torch_criterion, torched_standardized_train_data, torched_train_target)

    w1_grad_torch_np = w1_grad_torch.numpy().T
    b1_grad_torch_np = b1_grad_torch.numpy()
    w2_grad_torch_np = w2_grad_torch.numpy().T
    b2_grad_torch_np = b2_grad_torch.numpy()

    loss_abs_diff = compare_numbers(loss_custom, loss_torch)
    print(f'Custom NumPy loss value is {loss_custom}')
    print(f'PyTorch loss value is {loss_torch}\n')
    print(f'Absolute difference for losses is {loss_abs_diff}')

    differences = [
        (w1_grad_custom, w1_grad_torch_np, "W1"),
        (b1_grad_custom, b1_grad_torch_np, "b1"),
        (w2_grad_custom, w2_grad_torch_np, "W2"),
        (b2_grad_custom, b2_grad_torch_np, "b2"),
    ]

    for grad1, grad2, component_name in differences:
        max_diff = compare(grad1, grad2)
        print(f'Maximum absolute difference for {component_name} is {max_diff}')
        if max_diff > MAX_DELTA_GRAD:
            print(f'Warning: the absolute difference for custom gradient for {component_name} is too big\n')

    print()
    experiments = [
        (custom_model.components[0].weight, w1_grad_custom, (0, 0), "W1[0, 0]"),
        (custom_model.components[0].bias, b1_grad_custom, (0,), "b1[0]"),
        (custom_model.components[2].weight, w2_grad_custom, (0, 0), "W2[0, 0]"),
        (custom_model.components[2].bias, b2_grad_custom, (0,), "b2[0]")
    ]

    for component, grad_backward, idx, param_name in experiments:
        numeric_grad_component = numeric_grad(custom_model, custom_criterion, EPSILON, standardized_train_data, train_target, component, idx)
        backward_grad_component = grad_backward[idx]
        compared = compare_numbers(numeric_grad_component, backward_grad_component)

        print(f'({param_name}): backward-grad = {backward_grad_component}, numeric-grad = {numeric_grad_component}, absolute-difference = {compared}')
        if compared > MAX_DELTA_NUMERIC:
            print(f'Warning: the absolute difference for numeric gradient for {param_name} is too big\n')


if __name__ == '__main__':
    main()




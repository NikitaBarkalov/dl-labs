import numpy as np

def run_backward(model, criterion, train_data, train_targets):
    grads = []
    logits = model(train_data)
    loss = criterion(logits, train_targets)
    loss_value = loss.item()
    loss.backward()

    for component in model:
        if hasattr(component, 'weight'):
            grads.append(component.weight.grad)

        if hasattr(component, 'bias'):
            grads.append(component.bias.grad)

    return loss_value, *grads

def compare(array1, array2):
    abs_diff = np.abs(array1 - array2)
    return np.max(abs_diff)

def compare_numbers(n1, n2):
    abs_diff = np.abs(n1 - n2)
    return abs_diff
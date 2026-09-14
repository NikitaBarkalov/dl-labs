import numpy as np

class Parameter():
    def __init__(self, params):
        self.params = params
        self.grad = None

class CustomLinear():
    def __init__(self, weight, bias):
        self.weight = Parameter(weight)
        self.bias = Parameter(bias)
        self.input = None

    def __call__(self, input):
        self.input = input
        return input @ self.weight.params + self.bias.params

    def backward(self, output_grad):
        self.weight.grad = self.input.T @ output_grad
        self.bias.grad = np.sum(output_grad, axis = 0)
        return output_grad @ self.weight.params.T

class CustomReLU():
    def __init__(self):
        self.input = None

    def __call__(self, input):
        self.input = input
        return np.maximum(0.0, input)

    def backward(self, output_grad):
        return output_grad * (self.input > 0.0)

class CustomSequential():
    _last_model = None

    def __init__(self, *components):
        self.components = [c for c in components]

    def __iter__(self):
        return iter(self.components)

    def __call__(self, input):
        CustomSequential._last_model = self
        current_outputs = None
        for component in self.components:
            current_outputs = component(input) if current_outputs is None else component(current_outputs)

        return current_outputs

    def backward(self, output_grad):
        for component in reversed(self.components):
            output_grad = component.backward(output_grad)
        return output_grad

class CustomCrossEntropyLoss():
    def __init__(self, is_correct = True):
        self.is_correct = is_correct
        self.probas = None
        self.targets = None
        self.loss_value = None

    def __call__(self, logits, targets):
        self.targets = targets
        count = logits.shape[0]

        max_logits = np.max(logits, axis = 1, keepdims = True)
        shifted_logits = logits - max_logits
        log_sum_exp = np.log(np.sum(np.exp(shifted_logits), axis = 1, keepdims = True))
        logprobs = shifted_logits - log_sum_exp

        self.probas = np.exp(logprobs)
        self.loss_value = -np.mean(logprobs[np.arange(count), targets])
        return self

    def item(self):
        return self.loss_value

    def backward(self):
        count = self.probas.shape[0]
        dL_dZ = self.probas.copy()
        dL_dZ[np.arange(count), self.targets] -= 1.0

        if self.is_correct:
            CustomSequential._last_model.backward(dL_dZ / count)
        else:
            CustomSequential._last_model.backward(dL_dZ)
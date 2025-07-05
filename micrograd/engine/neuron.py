import random

from micrograd.engine.value import Value


class Neuron:

    def __init__(self, n_inputs):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, input):
        activation = (
            sum(
                weight_ith * input_ith
                for weight_ith, input_ith in zip(self.weights, input)
            )
            + self.bias
        )
        out = activation.tanh()
        return out

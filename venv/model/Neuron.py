# OOP Neuron GitHub Link from an Java Example.
# https://github.com/lanconi/OOP-Neuron-Modeling
import numpy as np
from Functions import Functions


class Neuron:
    # Initializer / Instance Attributes
    def __init__(self, nr, weights, layer, bias):
        self.nr = nr
        self.layer = layer
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        result = np.dot(self.weights, inputs) + self.bias
        return Functions.sigmoid(result)

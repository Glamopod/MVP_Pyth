# OOP Neuron GitHub Link from an Java Example.
# https://github.com/lanconi/OOP-Neuron-Modeling
import numpy as np
from function.Functions import Functions as fkt


class Neuron:
    # Initializer / Instance Attributes
    def __init__(self, id, layer, weights, bias):
        self.id = id
        self.layer = layer
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        result = np.dot(self.weights, inputs) + self.bias
        return Functions.sigmoid(result)

# OOP Neuron GitHub Link from an Java Example.
# https://github.com/lanconi/OOP-Neuron-Modeling


class Neuron:
    # Initializer / Instance Attributes
    def __init__(self, id, weight, layer, bias):
        self.name = id
        self.age = weight
        self.layer = layer
        self.bias = bias

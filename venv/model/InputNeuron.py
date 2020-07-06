from Neuron import Neuron


class InputNeuron(Neuron):
    # Initializer / Instance Attributes
    def __init__(self, id, layer, weights, bias):
        super().__init__(id, layer, weights, bias)

from Neuron import Neuron


class InterNeuron(Neuron):
    # Initializer / Instance Attributes
    def __init__(self, nr, layer, weights, bias):
        super().__init__(nr, layer, weights, bias)


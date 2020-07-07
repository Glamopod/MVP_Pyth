import numpy as np
from Neuron import Neuron
from InputNeuron import InputNeuron
from InterNeuron import InterNeuron
from OutputNeuron import OutputNeuron


class NeuralNetwork:
    def __init__(self):
        # nr, weights, layer, bias
        weights = np.array([0, 1])
        bias = 0
        self.inputNeuron1 = InputNeuron(1, weights, 1, bias)
        self.inputNeuron2 = InputNeuron(2, weights, 1, bias)
        self.interNeuron1 = InterNeuron(3, weights, 2, bias)
        self.interNeuron2 = InterNeuron(4, weights, 2, bias)
        self.interNeuron3 = InterNeuron(5, weights, 2, bias)
        self.outputNeuron = OutputNeuron(1, weights, 3, bias)

        self.h1 = Neuron(1, weights, 0, bias)
        self.h2 = Neuron(2, weights, 0, bias)
        self.o1 = Neuron(3, weights, 0, bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        # Вводы для о1 являются выводами h1 и h2
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1


network = NeuralNetwork()
x = np.array([2, 3])
print(network.feedforward(x))  # 0.7216325609518421

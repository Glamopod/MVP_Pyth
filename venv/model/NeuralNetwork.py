import numpy as np
from Neuron import Neuron
from InputNeuron import InputNeuron
from InterNeuron import InterNeuron
from OutputNeuron import OutputNeuron


class NeuralNetwork:
    def __init__(self):
        # nr(id), weights, layer, bias
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

    def feed_forward(self, x):
        output_of_h1 = self.h1.feed_forward_neuron(x)
        output_of_h2 = self.h2.feed_forward_neuron(x)

        # Вводы для о1 являются выводами h1 и h2
        output_of_o1 = self.o1.feed_forward_neuron(np.array([output_of_h1, output_of_h2]))

        return output_of_o1


network = NeuralNetwork()
x = np.array([2, 3])
print(network.feed_forward(x))  # 0.7216325609518421

import numpy as np


class Functions:
    def sigmoid(self, x):
        # Aktivierungsfunktion: f(x) = 1 / (1 + e^(-x))
        return 1 / (1 + np.exp(-x))

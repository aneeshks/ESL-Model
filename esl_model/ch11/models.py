from ..base import BaseStatModel
from ..math_utils import sigmoid
import numpy as np


class ClassificationMixin:
    def __init__(self, *args, **kwargs):
        K = kwargs.pop('K')
        super().__init__(*args, **kwargs)
        self.K = K

    def _pre_processing_y(self, y: np.ndarray):
        N = y.shape[0]
        iy = y.flatten()
        mask = iy.copy() - 1
        classification = np.zeros((N, self.K))
        for i in range(self.K):
            classification[mask==i, i] = 1
        return classification


class BaseNeuralNetwork(BaseStatModel):


    def _pre_processing_x(self, X: np.ndarray):
        # Manual add bias "1" in `train`
        X = self.standardize(X)
        return X

    @property
    def y_hat(self):
        return self._y_hat


class NeuralNetworkN1(ClassificationMixin, BaseNeuralNetwork):

    def train(self):
        X = np.insert(self.train_x, 0, [1], axis=1)
        y = self.train_y

        weights = np.random.uniform(-0.7, 0.7, (self.p + 1, self.K))

        # forward propagation
        T = sigmoid(X@weights)

        # back propagation
        for i in range(self.N):
            x = X[i]
            delta = y[i] - T[i]


        self._y_hat = T





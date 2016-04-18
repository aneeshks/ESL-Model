from ..base import BaseStatModel, ClassificationMixin
from ..math_utils import sigmoid
import numpy as np


class BaseNeuralNetwork(ClassificationMixin, BaseStatModel):
    def __init__(self, *args, **kwargs):
        self.alpha = kwargs.pop('alpha')
        # self.K = kwargs.pop('K')
        super().__init__(*args, **kwargs)

    def _pre_processing_x(self, X: np.ndarray):
        # Manual add bias "1" in `train`
        X = self.standardize(X)
        return X

    # def _pre_processing_y(self, y: np.ndarray):
    #     N = y.shape[0]
    #     iy = y.flatten()
    #     mask = iy.copy()
    #     classification = np.zeros((N, self.K))
    #     for i in range(self.K):
    #         classification[mask == i, i] = 1
    #     return classification

    @property
    def y_hat(self):
        return self._y_hat

    @property
    def rss(self):
        return 1 - np.sum(self.y_hat == self._raw_train_y) / self.N


class NeuralNetworkN1(BaseNeuralNetwork):

    def train(self):
        X = self.train_x
        y = self.train_y
        N = self.N

        _weights = np.random.uniform(-0.7, 0.7, (self.p + 1, self.n_class))
        weights = _weights[1:]
        weights0 = _weights[0]
        nw = np.zeros_like(weights)
        nw0 = np.zeros_like(weights0)
        print('y', y.shape)
        for i in range(N):
            x = X[i]
            t = sigmoid(x @ weights + weights0)
            print(t)
            # break
            # print(t)
            delta = (t- y[i])#*(1-x @ weights)*(x @ weights)
            # print(delta)
            # uw = np.repeat(x.reshape((-1,1)), 10, axis=1) * delta
            nw = nw + (x[:,None]@delta[None,:])

            nw0 += delta

        nw = (self.alpha*weights + nw)/N
        nw0 = nw0/N
        NT = sigmoid(X@(nw) + nw0)
        self.nw = nw
        self.nw0 = nw0

        # self._y_hat = NT.argmax(axis=1).reshape((-1,1))
        self._y_hat = self._inverse_matrix_to_class(NT)

    def predict(self, X: np.ndarray):
        X = self._pre_processing_x(X)
        y = sigmoid(X@self.nw + self.nw0)
        # return y.argmax(axis=1).reshape((-1,1))
        return self._inverse_matrix_to_class(y)






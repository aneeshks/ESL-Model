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
        for r in range(300):
            nw = np.zeros_like(weights)
            nw0 = np.zeros_like(weights0)
            for i in range(N):
                x = X[i]
                z = x @ weights + weights0
                t = sigmoid(z)
                # print(t)
                # break
                # print(t)
                delta = -(y[i]-t)#*((1-t)*t)*2
                # print('delta',delta)
                # break
                # print(delta)
                # uw = np.repeat(x.reshape((-1,1)), 10, axis=1) * delta
                nw = nw + (x[:,None]@delta[None,:])

                nw0 = nw0 + delta

            weights = weights  - self.alpha*nw
            # weights0 = self.alpha*nw0

        print(weights)
        NT = sigmoid(X@(weights) + weights0)
        self.nw = weights
        self.nw0 = weights0


        self._y_hat = NT.argmax(axis=1).reshape((-1,1))
        # self._y_hat = self._inverse_matrix_to_class(NT)

    def predict(self, X: np.ndarray):
        X = self._pre_processing_x(X)
        y = sigmoid(X@self.nw + self.nw0)
        # return y.argmax(axis=1).reshape((-1,1))
        return self._inverse_matrix_to_class(y)



class NN1(BaseNeuralNetwork):
    def train(self):
        X = self.train_x
        y = self.train_y
        N = self.N

        _weights = np.random.uniform(-0.7, 0.7, (self.p + 1, self.n_class))
        weights = _weights[1:]
        weights0 = _weights[0]
        nw = np.zeros_like(weights)
        nw0 = np.zeros_like(weights0)

        for i in range(N):
            x = X[i]
            z = x@weights + weights0
            t = sigmoid(z)
            delta = -2*(y[i]-t)*sigmoid(1-z)*sigmoid(z)
            weights = weights - self.alpha*(x[:,None]@delta[None,:])
            weights0 = weights0 - self.alpha*delta

        nw = weights
        nw0=weights0
        NT = sigmoid(X @ (nw) + nw0)
        self._y_hat = NT.argmax(axis=1).reshape((-1, 1))
        self.nw = nw
        self.nw0 = nw0


    def predict(self, X: np.ndarray):
        X = self._pre_processing_x(X)
        y = sigmoid(X @ self.nw + self.nw0)
        # return y.argmax(axis=1).reshape((-1,1))
        return self._inverse_matrix_to_class(y)


class NN2(BaseNeuralNetwork):
    def __init__(self, *args, hidden=12, iter_time=3,**kwargs):
        self.hidden = hidden
        self.iter_time = iter_time
        super().__init__(*args, **kwargs)

    def train(self):
        X = self.train_x
        y = self.train_y
        N = self.N

        _weights = np.random.uniform(-0.7, 0.7, (self.p + 1, self.hidden))
        weights = _weights[1:]
        weights0 = _weights[0]
        _weights = np.random.uniform(-0.7, 0.7, (self.hidden + 1, self.n_class))
        weights1 = _weights[1:]
        weights10 = _weights[0]

        # w1 = np.zeros_like(weights1)
        # w10 = np.zeros_like(weights10)
        # w = np.zeros_like(weights)
        # w0 = np.zeros_like(weights0)
        for r in range(self.iter_time):
            w1 = np.zeros_like(weights1)
            w10 = np.zeros_like(weights10)
            w = np.zeros_like(weights)
            w0 = np.zeros_like(weights0)
            for i in range(N):
                x = X[i]
                a1 = x
                z2 = x @ weights + weights0
                a2 = sigmoid(z2)
                z3 = a2 @ weights1 + weights10
                a3 = sigmoid(z3)

                delta3 = -2*(y[i] - a3) #* (1-a3)*a3
                delta2 = weights1@delta3*a2*(1-a2)

                w1 += self.alpha * (a2[:, None] @ delta3[None, :])
                w10 += self.alpha * delta3

                w +=self.alpha * (a1[:, None] @ delta2[None, :])
                w0 += self.alpha * delta2
            weights0 -= w0/N
            weights -= w/N
            weights10 -= w10/N
            weights1 -= w1/N
        # nw = weights
        # nw0 = weights0
        # NT = sigmoid(X @ (nw) + nw0)
        # self._y_hat = NT.argmax(axis=1).reshape((-1, 1))
        # self.nw = nw
        # self.nw0 = nw0
        self.weights = weights
        self.weights0 =weights0
        self.weights1 = weights1
        self.weights10 = weights10


    def predict(self, X: np.ndarray):
        X = self._pre_processing_x(X)
        a2 = sigmoid(X @ self.weights + self.weights0)
        y = sigmoid(a2 @ self.weights1 + self.weights10)
        # return y.argmax(axis=1).reshape((-1,1))
        return self._inverse_matrix_to_class(y)

    @property
    def y_hat(self):
        return self.predict(self._raw_train_x)
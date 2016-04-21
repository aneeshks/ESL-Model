from ..base import BaseStatModel, ClassificationMixin
from ..math_utils import sigmoid
import numpy as np
from itertools import chain



class BaseNeuralNetwork(ClassificationMixin, BaseStatModel):
    def __init__(self, *args, n_iter=10, **kwargs):
        self.alpha = kwargs.pop('alpha')
        self.n_iter = n_iter
        super().__init__(*args, **kwargs)

    def _pre_processing_x(self, X: np.ndarray):
        # Manual add bias "1" in `train`
        X = self.standardize(X)
        return X

    @property
    def y_hat(self):
        return self.predict(self._raw_train_x)

    @property
    def rss(self):
        return 1 - np.sum(self.y_hat == self._raw_train_y) / self.N

    def _forward_propagation(self, x):
        raise NotImplementedError

    def _back_propagation(self, target, layer_output):
        raise NotImplementedError


class BaseMiniBatchNeuralNetwork(BaseNeuralNetwork):
    def __init__(self, *args, mini_batch=10, hidden_layer=None, **kwargs):
        self.mini_batch = mini_batch
        self.hidden_layer=hidden_layer or list()
        super().__init__(*args, **kwargs)


    def _forward_propagation(self, x):
        a = x.copy()
        layer_output = []
        layer_output.append(a)
        for theta, intercept in self.thetas:
            a = sigmoid(a@theta + intercept)
            layer_output.append(a)
        return layer_output

    def _back_propagation(self, target, layer_output):
        delta = -(target - layer_output[-1])
        theta_grad = []

        for (theta, intercept), a in zip(reversed(self.thetas), reversed(layer_output[:-1])):
            grad = a.T @ delta
            intercept_grad = np.sum(delta, axis=0)
            # TODO: verify this right
            delta =  ((1-a)*a) * (delta @ theta.T)
            theta -= grad * self.alpha / self.mini_batch
            intercept -= intercept_grad * self.alpha / self.mini_batch


        return theta_grad[::-1]

    def _one_iter_train(self):
        X = self.train_x
        y = self.train_y
        mini_batch = self.mini_batch
        for j in range(0, self.N, mini_batch):
            x = X[j: j + mini_batch]
            target = y[j: j + mini_batch]
            layer_output = self._forward_propagation(x)
            self._back_propagation(target=target, layer_output=layer_output)

    def _init_theta(self):
        """
        theta is weights
        init all theta, depend on hidden layer
        :return: No return, store the result in self.thetas which is a list
        """
        thetas = []
        input_dimension = self.train_x.shape[1]
        for target_dimension in chain(self.hidden_layer, [self.n_class]):
            _theta = np.random.uniform(-0.7, 0.7, (input_dimension + 1, target_dimension))
            theta = _theta[1:]
            intercept = _theta[0]
            thetas.append((theta, intercept))
            input_dimension = target_dimension
        self.thetas = thetas

    def train(self):
        self._init_theta()
        for r in range(self.n_iter):
            self._one_iter_train()

    def predict(self, X: np.ndarray):
        X = self._pre_processing_x(X)
        y = self._forward_propagation(X)[-1]
        return self._inverse_matrix_to_class(y)



class MiniBatchNN(BaseMiniBatchNeuralNetwork):
    pass

class NeuralNetworkN1(BaseNeuralNetwork):
    """
    depend on book, use batch update.
    """

    def train(self):
        X = self.train_x
        y = self.train_y
        N = self.N
        _weights = np.random.uniform(-0.7, 0.7, (self.p + 1, self.n_class))
        weights = _weights[1:]
        weights0 = _weights[0]
        for r in range(self.n_iter):
            nw = np.zeros_like(weights)
            nw0 = np.zeros_like(weights0)
            for i in range(N):
                x = X[i]
                z = x @ weights + weights0
                t = sigmoid(z)
                # reference ng cousera course.
                delta = -(y[i]-t)
                nw = nw + (x[:,None]@delta[None,:])
                nw0 = nw0 + delta

            weights = weights  - self.alpha*nw
            weights0 = weights0 - self.alpha*nw0

        self.nw = weights
        self.nw0 = weights0

    def predict(self, X: np.ndarray):
        X = self._pre_processing_x(X)
        y = sigmoid(X@self.nw + self.nw0)
        return self._inverse_matrix_to_class(y)



class MiniBatchNN1(BaseNeuralNetwork):
    """
    use mini batch
    """
    def __init__(self, *args, mini_batch=10, **kwargs):
        self.mini_batch=mini_batch
        super().__init__(*args, **kwargs)

    def train(self):
        X = self.train_x
        y = self.train_y
        N = self.N

        _weights = np.random.uniform(-0.7, 0.7, (self.p + 1, self.n_class))
        weights = _weights[1:]
        weights0 = _weights[0]

        for r in range(self.n_iter):
            for j in range(0, N, self.mini_batch):
                nw = np.zeros_like(weights)
                nw0 = np.zeros_like(weights0)
                for i in range(j, j+self.mini_batch):
                    if (j+self.mini_batch) >= N:
                        break

                    x = X[i]
                    z = x@weights + weights0
                    t = sigmoid(z)
                    delta = -(y[i]-t)
                    nw += (x[:,None]@delta[None,:])
                    nw0 += delta

                weights -= self.alpha*nw
                weights0 -= self.alpha*nw0

        self.weights = weights
        self.weights0 = weights0


    def predict(self, X: np.ndarray):
        X = self._pre_processing_x(X)
        y = sigmoid(X @ self.weights + self.weights0)
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
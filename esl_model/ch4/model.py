from math import log

from ..ch3.model import LeastSquareModel, LinearModel
import numpy as np
from numpy import linalg as LA
from ..utils import lazy_method
from functools import partial



class LinearRegression(LinearModel):
    def __init__(self, *args, **kwargs):
        self.K = kwargs.pop('K', 1)
        super().__init__(*args, **kwargs)

    @property
    @lazy_method
    def error_rate(self):
        return (1 - np.sum((self._raw_train_y == self.y_hat))/ self.N)


class LinearRegressionIndicatorMatrix(LeastSquareModel):
    def __init__(self, *args, **kwargs):
        self.K = kwargs.pop('K', 1)
        super().__init__(*args, **kwargs)

    def _pre_processing_y(self, y):
        iy = y.flatten()
        N = y.shape[0]
        if self.K > 1:
            Y = np.zeros((N, self.K))
            for i in range(N):
                k = iy[i]
                # k starts from 1
                Y[i, k-1] = 1
        else:
            return super()._pre_processing_y(y)

        return Y

    def predict(self, x):
        Y_hat = super().predict(x)
        y = (Y_hat.argmax(axis=1)).reshape((-1,1)) + 1

        return y

    @property
    @lazy_method
    def error_rate(self):
        return (1 - np.sum((self._raw_train_y == self.y_hat))/ self.N)


class LDAModel(LinearRegression):
    """
    Linear Discriminant Analysis
    from page 106
    """

    def _pre_processing_x(self, x):
        x = self.standardize(x)
        return x

    def train(self):
        X = self.train_x
        y = self.train_y
        K = self.K
        p = self.p

        self.Mu = np.zeros((K, p))
        self.Pi = np.zeros((K, 1))
        self.Sigma_hat = np.zeros((p, p))

        for k in range(K):
            mask = (y == k+1)
            N_k = sum(mask)

            X_k = X[mask.flatten(), :]

            self.Pi[k] = N_k / self.N
            self.Mu[k] = np.sum(X_k, axis=0).reshape((1, -1)) / N_k
            self.Sigma_hat = self.Sigma_hat + ((X_k - self.Mu[k]).T @ (X_k - self.Mu[k])) / (self.N - K)




    def linear_discriminant_func(self, x, k):
        """
        linear discriminant function.
        Define by (4.10)
        :return: delta_k(x)
        """
        mu_k = self.Mu[k]
        pi_k = self.Pi[k]
        sigma_inv = self.math.pinv(self.Sigma_hat)
        result = mu_k @ sigma_inv @ x.T - (mu_k @ sigma_inv @ mu_k.T)/2 + log(pi_k)

        return result

    def predict(self, x):
        X = self._pre_processing_x(x)
        N = X.shape[0]
        Y = np.zeros((N, self.K))

        for k in range(self.K):
            # delta_k is (N x 1)
            delta_k = self.linear_discriminant_func(X, k)
            Y[:, k] = delta_k

        # make the k start from 1
        y_hat = Y.argmax(axis=1).reshape((-1, 1)) + 1

        return y_hat


class QDAModel(LinearRegression):
    """
    Quadratic Discriminant Analysis
    pp. 110

    Ref
    ---
    http://www.wikicoursenote.com/wiki/Stat841#In_practice
    """

    def train(self):
        X = self.train_x
        y = self.train_y
        K = self.K
        p = self.p

        self.Mu = np.zeros((K, p))
        self.Pi = np.zeros((K, 1))
        self.Sigma_hat = []

        for k in range(K):
            mask = (y == k+1)
            N_k = sum(mask)

            X_k = X[mask.flatten(), :]

            self.Pi[k] = N_k / self.N
            self.Mu[k] = np.sum(X_k, axis=0).reshape((1, -1)) / N_k
            # We div by N_k instead of (N-K)
            self.Sigma_hat.append(((X_k - self.Mu[k]).T @ (X_k - self.Mu[k])) / (N_k))



    def quadratic_discriminant_func(self, x, k):
        mu_k = self.Mu[k]
        pi_k = self.Pi[k]
        sigma_k = self.Sigma_hat[k]
        pinv = self.math.pinv

        # assume that each row of x contain observation

        result = -(np.log(np.linalg.det(sigma_k)))/2 - \
                 ((x - mu_k) @ pinv(sigma_k, rcond=0) @ (x - mu_k).T)/2 + \
                 log(pi_k)
        return result



    def predict(self, x):
        X = self._pre_processing_x(x)
        N = X.shape[0]
        Y = np.zeros((N, self.K))

        for k in range(self.K):
            # the intuitive solution is use np.apply_along_axis, but is too slow
            # delta_k is (N x 1)
            # delta_k_func = partial(self.linear_discriminant_func, k=k)
            # delta_k = np.apply_along_axis(delta_k_func, 1, X)


            # d is NxN,
            # Let B = A@A.T, the diagonal of B is [A[i] @ A[i].T for i in range(A.shape(0)]
            d = self.quadratic_discriminant_func(X, k)
            Y[:, k] = d.diagonal()

        # make the k start from 1
        y_hat = Y.argmax(axis=1).reshape((-1, 1)) + 1

        return y_hat


class RDAModel(QDAModel):
    def __init__(self, *args, alpha=1, **kwargs):
        self.alpha = alpha
        super().__init__(*args, **kwargs)


    def train(self):

        X = self.train_x
        y = self.train_y
        K = self.K
        p = self.p

        self.Mu = np.zeros((K, p))
        self.Pi = np.zeros((K, 1))
        # list of sigma_k
        self.Sigma_hat = []
        # the sum of sigma_k
        self.Sigma_tot = np.zeros((1, p))

        for k in range(K):
            mask = (y == k+1)
            N_k = sum(mask)

            X_k = X[mask.flatten(), :]

            self.Pi[k] = N_k / self.N
            self.Mu[k] = np.sum(X_k, axis=0).reshape((1, -1)) / N_k
            # We div by N_k instead of (N-K)
            self.Sigma_hat.append(((X_k - self.Mu[k]).T @ (X_k - self.Mu[k])) / (N_k))
            self.Sigma_tot = self.Sigma_tot + (X_k - self.Mu[k]).T @ (X_k - self.Mu[k])

        self.Sigma_tot = self.Sigma_tot / (self.N - K)

        for k in range(K):
            self.Sigma_hat[k] = (self.Sigma_hat[k] * self.alpha) + self.Sigma_tot * (1 - self.alpha)




class LDAForComputation(LDAModel):

    def train(self):
        super().train()
        sigma = self.Sigma_hat
        D_, U = LA.eigh(sigma)
        D = np.diagflat(D_)

        self.A = np.power(LA.pinv(D),0.5) @ U.T

    def predict(self, x):
        X = self._pre_processing_x(x)
        Y = np.zeros((X.shape[0], self.K))
        A = self.A

        # because X is (N x p), A is (K x p), we can to get the X_star (NxK)
        X_star = X @ A.T

        for k in range(self.K):
            # mu_s_star shape is (p,)
            mu_k_star = A @ self.Mu[k]

            # Ref: http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html
            # Ref: http://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
            Y[:, k] = LA.norm(X_star - mu_k_star, axis=1) * 0.5 - log(self.Pi[k])

        y_hat = Y.argmin(axis=1).reshape((-1, 1)) + 1
        return y_hat



class ReducedRankLDAModel(LDAForComputation):
    def __init__(self, *args, L, **kwargs):
        self.L = L
        super().__init__(*args, **kwargs)

    def train(self):
        super().train()
        W = self.Sigma_hat
        mu = np.sum(self.Pi * self.Mu, axis=0)
        B = np.zeros((self.p, self.p))
        for k in range(self.K):
            B = B + self.Pi[k]*((self.Mu[k] - mu)[:, None] @ ((self.Mu[k] - mu)[None, :]))
        # print('B',B)
        # k=1
        # print('shape', self.Pi[k]*((self.Mu[k] - mu)[:, None] @ (self.Mu[k] - mu)[:, None]))

        # get W**0.5
        Dw_, Uw = LA.eigh(W)
        Dw_ = Dw_[::-1]
        Uw = np.fliplr(Uw)
        Dw = np.diagflat(np.power(Dw_, -0.5))
        W_half = self.math.pinv(np.diagflat(Dw_**0.5) @ Uw.T)
        B_star = Dw @ Uw.T @ B @ Uw @ Dw
        # print(Dw_)
        D_, V = LA.eigh(B_star)
        V = np.fliplr(V)
        self.A = np.zeros((self.L, self.p))
        for l in range(self.L):
            self.A[l, :] = (W_half) @ V[:, l]

        # self.Mu_star = self.Mu @ self.A.T


    def predict(self, x):
        X = self._pre_processing_x(x)
        Y = np.zeros((X.shape[0], self.K))
        A = self.A

        # because X is (N x p), A is (K x p), we can to get the X_star (NxK)
        X_star = X @ A.T

        for k in range(self.K):
            # mu_s_star shape is (p,)
            mu_k_star = A @ self.Mu[k]

            # Ref: http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html
            # Ref: http://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
            Y[:, k] = LA.norm(X_star - mu_k_star, axis=1) * 0.5 + log(self.Pi[k])

        y_hat = Y.argmin(axis=1).reshape((-1, 1)) + 1
        return y_hat

# import sklearn.lda
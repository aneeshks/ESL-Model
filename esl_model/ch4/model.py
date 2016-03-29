from math import log

from ..ch3.model import LeastSquareModel, LinearModel
import numpy as np
from numpy import linalg as LA
from ..utils import lazy_method


class LinearRegression(LinearModel):
    def __init__(self, *args, **kwargs):
        self.K = kwargs.pop('K', 1)
        super().__init__(*args, **kwargs)

    @property
    @lazy_method
    def error_rate(self):
        return 1 - np.sum((self._raw_train_y == self.y_hat))/ self.N


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

    def predict(self, X):
        Y_hat = super().predict(X)
        y = (Y_hat.argmax(axis=1)).reshape((-1, 1)) + 1
        return y

    @property
    @lazy_method
    def error_rate(self):
        return (1 - np.sum((self._raw_train_y == self.y_hat)) / self.N)


class LDAModel(LinearRegression):
    """
    Linear Discriminant Analysis
    from page 106
    """

    def _pre_processing_x(self, X):
        X = self.standardize(X)
        return X

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

    def predict(self, X):
        X = self._pre_processing_x(X)
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
            self.Sigma_hat.append(((X_k - self.Mu[k]).T @ (X_k - self.Mu[k])) / N_k)

    def quadratic_discriminant_func(self, x, k):
        mu_k = self.Mu[k]
        pi_k = self.Pi[k]
        sigma_k = self.Sigma_hat[k]
        pinv = self.math.pinv

        # assume that each row of x contain observation
        result = -(np.log(np.linalg.det(sigma_k)))/2 - \
                 ((x - mu_k) @ pinv(sigma_k, rcond=0) @ (x - mu_k).T)/2 + log(pi_k)
        return result

    def predict(self, X):
        X = self._pre_processing_x(X)
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
            self.Sigma_hat.append(((X_k - self.Mu[k]).T @ (X_k - self.Mu[k])) / N_k)
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
        self.A = np.power(LA.pinv(D), 0.5) @ U.T

    def predict(self, X):
        X = self._pre_processing_x(X)
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

        # Python index start from 0, transform to start with 1
        y_hat = Y.argmin(axis=1).reshape((-1, 1)) + 1
        return y_hat


class ReducedRankLDAModel(LDAForComputation):
    """
    page 113, 4.3.3

    I also write a blog describe how to write RRLDA:
    http://littlezz.github.io/how-to-write-reduced-rank-linear-discriminant-analysis-with-python.html

    ref: http://sites.stat.psu.edu/~jiali/course/stat597e/notes2/lda2.pdf
    """

    def __init__(self, *args, L, **kwargs):
        self.L = L
        super().__init__(*args, **kwargs)

    def train(self):
        super().train()
        W = self.Sigma_hat
        # prior probabilities (K, 1)
        Pi = self.Pi
        # class centroids (K, p)
        Mu = self.Mu
        p = self.p
        # the number of class
        K = self.K
        # the dimension you want
        L = self.L

        # Mu is (K, p) matrix, Pi is (K, 1)
        mu = np.sum(Pi * Mu, axis=0)
        B = np.zeros((p, p))

        for k in range(K):
            # vector @ vector equal scalar, use vector[:, None] to transform to matrix
            # vec[:, None] equal to vec.reshape((1, vec.shape[0]))
            B = B + Pi[k]*((Mu[k] - mu)[:, None] @ ((Mu[k] - mu)[None, :]))

        # Be careful, the `eigh` method get the eigenvalues in ascending , which is opposite to R.
        Dw, Uw = LA.eigh(W)
        # reverse the Dw_ and Uw
        Dw = Dw[::-1]
        Uw = np.fliplr(Uw)

        W_half = self.math.pinv(np.diagflat(Dw**0.5) @ Uw.T)
        B_star = W_half.T @ B @ W_half
        D_, V = LA.eigh(B_star)

        # reverse V
        V = np.fliplr(V)

        # overwrite `self.A` so that we can reuse `predict` method define in parent class
        self.A = np.zeros((L, p))
        for l in range(L):
            self.A[l, :] = W_half @ V[:, l]


class BinaryLogisticRegression(LinearRegression):
    """
    page 119.
    two class case

    note that self.W is the second partial derivative.

    ref: http://sites.stat.psu.edu/~jiali/course/stat597e/notes2/logit.pdf
    """
    def __init__(self, *args, max_iter=500, **kwargs):
        self.max_iter = max_iter
        super().__init__(*args, **kwargs)

    def _pre_processing_x(self, X):
        # TODO: refactor that allow choose whether use standardizing
        # X = super()._pre_processing_x(X)
        X = np.insert(X, 0, [1], axis=1)
        return X


    def train(self):
        X = self.train_x
        y = self.train_y
        # include intercept
        beta = np.zeros((self.p+1, 1))

        iter_times = 0
        while True:
            e_X = np.exp(X @ beta)

            # N x 1
            self.P = e_X / (1 + e_X)

            # W is a vector
            self.W = (self.P * (1 - self.P)).flatten()

            beta = beta + self.math.pinv((X.T * self.W) @ X) @ X.T @ (y - self.P)

            iter_times += 1
            if iter_times > self.max_iter:
                break

        self.beta_hat = beta


    def predict(self, X):
        X = self._pre_processing_x(X)
        y = X @ self.beta_hat
        y[y>=0] = 1
        y[y<0] = 0
        return y


    @property
    def std_err(self):
        """
        ref: https://groups.google.com/d/msg/comp.soft-sys.stat.spss/Fv7Goxs_Bwk/ff0jCesG8REJ
        """
        return self.math.pinv(self.train_x.T * self.W @ self.train_x).diagonal() ** 0.5

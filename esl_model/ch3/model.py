from ..base import BaseStatModel
from ..base import np
from ..utils import lazy_method
from itertools import combinations


class LinearModel(BaseStatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta_hat = None
        self.intercept = None

    def _pre_processing_x(self, X):
        X = self.standardize(X)
        return X

    def train(self):
        raise NotImplementedError

    def predict(self, X):
        X = self._pre_processing_x(X)
        X = np.insert(X, 0, 1, axis=1)
        return X @ self.beta_hat

    @property
    @lazy_method
    def sigma_hat(self):
        """
        (3.8)
        """
        # because we already add intercept to train x, so col_num is (p+1)

        N, col_num = self.train_x.shape
        return self.math.sum((self.y_hat - self._raw_train_y)**2) / (N-col_num)

    @property
    @lazy_method
    def std_err(self):
        var_beta_hat = self.math.inv(self.train_x.T @ self.train_x) * self.sigma_hat
        return var_beta_hat.diagonal() ** 0.5

    @property
    @lazy_method
    def z_score(self):
        return self.beta_hat.flatten() / self.std_err

    @property
    @lazy_method
    def y_hat(self):
        return self.predict(self._raw_train_x)

    def F_statistic(self, remove_cols):
        """
        (3.13)
        :param remove_cols: number of list, start from 0.  or feature name match `features_name`
        :return:
        """
        if isinstance(remove_cols[0], str):
            assert self.features_name, 'features_name not define!'
            cols_index = []
            for name in remove_cols:
                index = self.features_name.index(name)
                cols_index.append(index)
        else:
            cols_index = remove_cols

        other_train_x = np.delete(self._raw_train_x, cols_index, 1)
        other = self.__class__(train_x=other_train_x, train_y=self._raw_train_y)
        other.pre_processing()
        other.train()

        rss0 = other.rss
        rss1 = self.rss
        # p1 = p + 1
        N, p1 = self.train_x.shape

        return ((rss0-rss1) / (len(cols_index))) / (rss1 / (N - p1))

    @property
    @lazy_method
    def rss(self):
        return self.math.sum((self.y_hat - self._raw_train_y)**2)


class LeastSquareModel(LinearModel):
    def _pre_processing_x(self, X):
        X = self.standardize(X)
        X = np.insert(X, 0, 1, axis=1)
        return X

    def train(self):
        x = self.train_x
        y = self.train_y
        self.beta_hat = self.math.inv(x.T @ x) @ x.T @ y

    def predict(self, X):
        X = self._pre_processing_x(X)
        return X @ self.beta_hat


class BestSubsetSelection(LinearModel):
    def __init__(self, *args, k=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.select_column = None

    def _pre_processing_x(self, X):
        X = self.standardize(X)
        X = np.insert(X, 0, 1, axis=1)
        if self.select_column:
            X = X[:, self.select_column]
        return X

    def train(self):
        X_com = self.train_x
        y = self.train_y
        k = self.k
        p = self.p
        rss_min = None

        cm = combinations(range(1, p+1), k)
        for cols in list(cm):
            cols = (0, *cols)
            X = X_com[:, cols]
            beta_hat = self.math.inv(X.T @ X) @ X.T @ y
            rss = np.sum((y - X @ beta_hat)**2)
            if (rss_min is None) or rss < rss_min:
                self.select_column = cols
                self.beta_hat = beta_hat
                rss_min = rss

    def predict(self, X):
        X = self._pre_processing_x(X)
        return X @ self.beta_hat


class RidgeModel(LinearModel):
    def __init__(self, *args, **kwargs):
        self.alpha = kwargs.pop('alpha')
        self.solve = kwargs.pop('solve', 'svd')
        super().__init__(*args, **kwargs)

    def train(self):
        X = self.train_x
        y = self.train_y
        if self.solve == 'svd':
            u, d, vt = self.math.svd(X, full_matrices=False)
            ds = (d / (d**2 + self.alpha)).reshape((-1,1))
            self.beta_hat = vt.T @ (ds * (u.T @ y))

        elif self.solve == 'raw':
            self.beta_hat = self.math.inv(X.T @ X + np.eye(self.p)*self.alpha) @ X.T @ y
        else:
            raise NotImplementedError

        self.intercept = np.mean(y)
        self.beta_hat = np.insert(self.beta_hat, 0, self.intercept, axis=0)

    @property
    @lazy_method
    def df(self):
        u, d, vt = self.math.svd(self.train_x)
        return self.math.sum(d**2/(d**2 + self.alpha))


class LassoLARModel:
    WARN = "I was unable to write out this algorithm from book page 74, " \
           "if anyone understand it , please let me know"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(self.WARN)


class PrincipalComponentsRegression(LinearModel):
    def __init__(self, *args, **kwargs):
        m = kwargs.pop('m', None)
        super().__init__(*args, **kwargs)
        self.m = m or self.p

    def transform_z(self, X):
        U, D, Vt = self.math.svd(X, full_matrices=False)
        return X @ Vt.T[:, :self.m], U, D, Vt

    def train(self):
        X = self.train_x
        y = self.train_y

        Z, U, D, Vt = self.transform_z(X)

        theta = self.math.inv(Z.T @ Z) @ (Z.T @ y)
        beta = Vt.T[:, :self.m] @ theta

        self.intercept = np.mean(y)
        self.beta_hat = np.insert(beta, 0, self.intercept, axis=0)


class PartialLeastSquare(LinearModel):
    def __init__(self, *args, M=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.M = M or self.p

    def train(self):
        X = self.train_x
        y = self.train_y

        y_hat = [np.zeros((self.N, 1)) for i in range(self.M + 1)]
        x = [np.zeros((self.N, self.p)) for i in range(self.M + 1)]
        # Page 81 1st step
        y_hat[0][...] = np.mean(y)
        x[0] = X

        # 2nd step
        for m in range(1, self.M + 1):
            psi = x[m-1].T @ y
            z_m = x[m-1] @ psi
            theta_m = self.math.pinv(z_m.T @ z_m) @ (z_m.T @ y)
            y_hat[m][...] = y_hat[m-1] + z_m @ theta_m
            x[m][...] = x[m-1] - z_m @ (self.math.pinv(z_m.T @ z_m) @ (z_m.T @ x[m-1]))

        self.intercept = np.mean(y)
        beta = self.math.pinv(X) @ (y_hat[self.M] - self.intercept)
        self.beta_hat = np.insert(beta, 0, self.intercept, axis=0)
        self._y_hat = y_hat[self.M]

    @property
    def y_hat(self):
        return self._y_hat


class IFSRModel(LinearModel):
    """
    Algorithm 3.4 on page 86, without LAR. Well, I still do not know how to write LAR.
    I also reference http://waxworksmath.com/Authors/G_M/Hastie/WriteUp/weatherwax_epstein_hastie_solutions_manual.pdf
    """

    def __init__(self, *args, epsilon='auto', iter_max=5000, cor_threshold=1e-6, **kwargs):
        self.iter_max = iter_max
        self.epsilon = epsilon
        self.cor_threshold = cor_threshold
        super().__init__(*args, **kwargs)

    @staticmethod
    def max_correlation_index(x, y):
        cor = np.corrcoef(x, y, rowvar=0)[:-1, -1]
        j = np.abs(cor).argmax()
        max_cor = cor[j]
        return j, max_cor

    def train(self):
        X = self.train_x
        y = self.train_y
        r = y.copy()
        beta = np.zeros((self.p, 1))
        iter_time = 0

        if self.epsilon == 'auto':
            # ref:  http://waxworksmath.com/Authors/G_M/Hastie/WriteUp/weatherwax_epstein_hastie_solutions_manual.pdf
            #       page 30

            beta_ls_sum = np.sum(np.abs(self.math.pinv(X.T @ X) @ X.T @ y))
            epsilon = beta_ls_sum / (2 * self.iter_max)
        else:
            epsilon = self.epsilon

        while True:
            j, max_cor = self.max_correlation_index(X, r)
            xj = X[:, [j]]
            theta = np.sign(xj.T @ r) * epsilon
            r -= xj @ theta
            beta[j] = beta[j] + theta

            iter_time += 1

            if iter_time > self.iter_max or max_cor < self.cor_threshold:
                break

            self.intercept = np.mean(y)
            self.beta_hat = np.insert(beta, 0, self.intercept, axis=0)


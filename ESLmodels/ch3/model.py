from ESLmodels.base import BaseStatModel
from ESLmodels.base import np
from ESLmodels.utils import lazy_method
from itertools import combinations



class LinearModel(BaseStatModel):
    def __init__(self, train_x, train_y, features_name=None):
        super().__init__(train_x, train_y, features_name)
        self.beta_hat = None

    def pre_processing_x(self, x):
        x = self.standardize(x)
        return x

    def predict(self, x):
        x = self.pre_processing_x(x)
        return  x @ self.beta_hat


    @property
    @lazy_method
    def sigma_hat(self):
        """
        (3.8)
        """
        # because we already add intercept to train x, so col_num is (p+1)

        N, col_num = self.train_x.shape

        return self.math.sum((self.y_hat - self.train_y)**2) / (N-col_num)


    @property
    @lazy_method
    def std_err(self):
        var_beta_hat = self.math.inv(self.train_x.T @ self.train_x) * self.sigma_hat
        return var_beta_hat.diagonal() ** 0.5


    @property
    @lazy_method
    def z_score(self):
        return self.beta_hat / self.std_err



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
        other = self.__class__(train_x=other_train_x, train_y=self.train_y)
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
        return self.math.sum((self.y_hat - self.train_y)**2)


class LeastSquareModel(LinearModel):
    def pre_processing_x(self, x):
        x = super().pre_processing_x(x)
        x = np.insert(x, 0, 1, axis=1)
        return x

    def train(self):
        x = self.train_x
        y = self.train_y
        self.beta_hat = self.math.inv(x.T @ x) @ x.T @ y



class BestSubsetSelection(LinearModel):
    def __init__(self, *args, k=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.select_column = None

    def pre_processing_x(self, x):
        x = super().pre_processing_x(x)
        x = np.insert(x, 0, 1, axis=1)
        if self.select_column:
            x = x[:, self.select_column]
        return x

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




class RidgeModel(LinearModel):
    def __init__(self, *args, **kwargs ):
        self.alpha = kwargs.pop('alpha')
        self.solve = kwargs.pop('solve', 'svd')
        super().__init__(*args, **kwargs)

    def train(self):
        X = self.train_x
        y = self.train_y

        if self.solve == 'svd':
            u, d, vt = self.math.svd(X, full_matrices=False)
            ds = (d / (d**2 + self.alpha))
            self.beta_hat = vt.T @ (ds * (u.T @ y))


        elif self.solve == 'raw':
            self.beta_hat = self.math.inv(X.T @ X + np.eye(self.p)*self.alpha) @ X.T @ y

        else:
            raise NotImplementedError

        self.intercept = np.mean(y)
        self.beta_hat = np.insert(self.beta_hat, 0, self.intercept)

    def predict(self, x):
        x = self.pre_processing_x(x)
        x = np.insert(x, 0, 1, axis=1)
        return x @ self.beta_hat

    @property
    @lazy_method
    def df(self):
        u, d, vt = self.math.svd(self.train_x)
        return self.math.sum(d**2/(d**2 + self.alpha))




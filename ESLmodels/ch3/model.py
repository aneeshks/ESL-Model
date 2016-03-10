from ESLmodels.base import BaseStatModel
from ESLmodels.base import np
from ESLmodels.utils import lazy_method


class LeastSquareModel(BaseStatModel):



    def train(self):
        x = self.train_x
        y = self.train_y
        self.beta_hat = self.math.inv(x.T @ x) @ x.T @ y

    def predict(self, x):
        x = self.pre_processing_x(x)
        return  x @ self.beta_hat


    @property
    @lazy_method
    def y_hat(self):
        return self.predict(self._raw_train_x)

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


    @property
    @lazy_method
    def rss(self):
        return self.math.sum((self.y_hat - self.train_y)**2)

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



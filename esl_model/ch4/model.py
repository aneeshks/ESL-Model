from ..base import BaseStatModel
from ..ch3.model import LeastSquareModel
import numpy as np

# class LinearRegression(BaseStatModel):
#     def pre_processing_y(self, y):



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
    def error_rate(self):
        return (1 - np.sum((self._raw_train_y == self.y_hat))/ self.N)


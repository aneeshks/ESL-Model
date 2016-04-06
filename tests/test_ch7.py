import pytest
from .utils import digit_float
import numpy as np

@pytest.fixture
def prostate_data():
    from esl_model.datasets import ProstateDataSet
    p = ProstateDataSet()
    return p.return_all()


def test_ridge_cv(prostate_data):
    from esl_model.ch7.models import RidgeCV
    train_x, train_y, test_x, test_y, features = prostate_data

    alphas = np.arange(3, 8, 0.25)
    r = RidgeCV(train_x, train_y, alphas=alphas)
    r.pre_processing()
    r.train()

    print('df', r.df)
    print('alpha', r.alpha)
    print('best alpha', r.best_alpha)
    test_error = r.test(test_x, test_y).mse
    print(test_error)
    assert digit_float(test_error) == 0.492
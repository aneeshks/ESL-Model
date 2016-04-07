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

    alphas = np.arange(0, 80, 1)
    r = RidgeCV(train_x, train_y, alphas=alphas)
    r.pre_processing()
    r.train()

    print('df', r.df)
    print('best alpha', r.best_alpha)
    test_error = r.test(test_x, test_y).mse
    print(test_error)
    print('alpha errs', r.alpha_errs)
    print('alpha std errs', r.alpha_std_errs)
    # assert digit_float(test_error) == 0.492

    # from esl_model.ch3.models import RidgeModel
    # m = RidgeModel(train_x, train_y, alpha=7.75)
    # m.pre_processing()
    # m.train()
    # print(m.test(test_x, test_y).mse)
    # assert 0
    # from sklearn.linear_model.ridge import RidgeCV
    # m = RidgeCV(cv=10, alphas=alphas)
    # m.fit(train_x, train_y)
    # print(m.alpha_)
    assert 0


def test_pcr_cv(prostate_data):
    from esl_model.ch7.models import PCRCV
    train_x, train_y, test_x, test_y, features = prostate_data

    alphas = np.arange(0,9)
    cv = PCRCV(train_x, train_y, alphas=alphas)
    cv.pre_processing()
    cv.train()

    print('best alpha', cv.best_alpha)
    print('alpha erros', cv.alpha_errs)
    print('alpha std erros', cv.alpha_std_errs)
    test_error = cv.test(test_x, test_y).mse
    # print(cv.beta_hat)
    # print(cv.y_folds[1])
    # print(len(cv.x_folds))

    # from esl_model.ch3.models import PrincipalComponentsRegression
    # for i in range(0, 8):
    #     p = PrincipalComponentsRegression(train_x, train_y, m=i)
    #     p.pre_processing()
    #     p.train()
    #     print(p.test(train_x, train_y).mse, end=',')
    assert 0

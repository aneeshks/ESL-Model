"""
all CV get approximately result as book, but most CV use one standard rule can not get same best alpha with book
"""

import pytest
import numpy as np


@pytest.fixture
def prostate_data():
    from esl_model.datasets import ProstateDataSet
    p = ProstateDataSet()
    return p.return_all()


def test_ridge_cv(prostate_data):
    # FIXME: ridge CV can not get same best alpha with book
    from esl_model.ch7.models import RidgeCV
    from esl_model.math_utils import solve_df_lambda
    from esl_model.ch3.models import RidgeModel
    train_x, train_y, test_x, test_y, features = prostate_data

    # m = RidgeModel(train_x, train_y, alpha=7.75)
    # m.pre_processing()
    # train_x = m.train_x
    # train_x = scale(train_x)
    # train_y = train_y - np.mean(train_y)
    alphas = solve_df_lambda(train_x)
    print('input alphas', alphas)
    r = RidgeCV(train_x, train_y, alphas=alphas)
    r.pre_processing()
    r.train()

    print('df', r.df)
    print('best alpha', r.best_alpha)
    test_error = r.test(test_x, test_y).mse
    print(test_error)
    print('alpha errs', repr(r.alpha_errs))
    print('alpha std errs', r.alpha_std_errs)
    # assert digit_float(test_error) == 0.492

    # from esl_model.ch3.models import RidgeModel
    # m = RidgeModel(train_x, train_y, alpha=7.75)
    # m.pre_processing()
    # x = m.train_x
    # m.train()
    # print(m.test(test_x, test_y).mse)
    # assert 0
    # from sklearn.linear_model.ridge import RidgeCV
    # from sklearn.preprocessing import scale
    # m = RidgeCV(cv=10, alphas=alphas[1:])
    # m.fit(x, train_y)
    # print(m.alpha_)
    # print('len', len(r.x_folds[3]))



def test_pcr_cv(prostate_data):
    # FIXME:

    from esl_model.ch7.models import PCRCV
    train_x, train_y, test_x, test_y, features = prostate_data

    alphas = np.arange(0,9)
    cv = PCRCV(train_x, train_y, alphas=alphas[:], random=1)
    cv.pre_processing()
    cv.train()

    print('best alpha', cv.best_alpha)
    print('alpha erros', cv.alpha_errs)
    print('alpha std erros', cv.alpha_std_errs)
    test_error = cv.test(test_x, test_y).mse
    print(test_error)
    # print(cv.beta_hat)
    # print(cv.y_folds[1])
    # print(len(cv.x_folds))

    from esl_model.ch3.models import PrincipalComponentsRegression, LeastSquareModel
    for i in range(0, 9):
        p = PrincipalComponentsRegression(train_x, train_y, m=i)
        p.pre_processing()
        p.train()
        print(p.test(train_x, train_y).mse, end=',')

    # l = LeastSquareModel(train_x, train_y)
    # l.pre_processing()
    # l.train()
    # print(l.test(train_x, train_y).mse)
    # assert cv.best_alpha == 7


def test_df_solve(prostate_data):
    train_x, train_y, test_x, test_y, features = prostate_data
    from esl_model.math_utils import solve_df_lambda
    from esl_model.ch3.models import RidgeModel

    lams = solve_df_lambda(train_x)
    print(lams)
    dfs = []
    for i in range(train_x.shape[1]+1):
        r = RidgeModel(train_x, train_y, alpha=lams[i])
        r.pre_processing()
        dfs.append(r.df)

    assert np.allclose(dfs, np.arange(0, 9))


def test_pls_cv(prostate_data):
    from esl_model.ch7.models import PartialLeastSquareCV
    train_x, train_y, test_x, test_y, features = prostate_data

    alphas = np.arange(0, 9)
    cv = PartialLeastSquareCV(train_x, train_y, alphas=alphas)
    cv.pre_processing()
    cv.train()

    print('best alpha', cv.best_alpha)
    print('alpha erros', cv.alpha_errs)
    print('alpha std erros', cv.alpha_std_errs)
    print('test error', cv.test(test_x, test_y).mse)

    assert cv.best_alpha == 2



def test_best_subset_selection_cv(prostate_data):
    # FIXME
    from esl_model.ch7.models import BestSubsetSelectionCV
    train_x, train_y, test_x, test_y, features = prostate_data

    alphas = np.arange(0, 9)
    cv = BestSubsetSelectionCV(train_x, train_y, alphas=alphas)
    cv.pre_processing()
    cv.train()

    print('best alpha', cv.best_alpha)
    print('alpha erros', cv.alpha_errs)
    print('alpha std erros', cv.alpha_std_errs)
    print('test error', cv.test(test_x, test_y).mse)

import pytest
from .utils import read_data, digit_float
import os
import numpy as np







@pytest.fixture
def prostate_data():
    from ESLmodels.datasets import ProstateDataSet

    p = ProstateDataSet()
    return p.return_all()




def test_least_square_model(prostate_data):
    from ESLmodels.ch3.model import LeastSquareModel
    train_x, train_y, test_x, test_y, features = prostate_data
    lsm = LeastSquareModel(train_x=train_x, train_y=train_y, features_name=features)
    lsm.pre_processing()

    lsm.train()

    print(lsm.beta_hat)
    print('rss:',lsm.rss)
    # nx = np.delete(train_x, [2,5,6,7], axis=1)

    # other = LeastSquareModel(train_x=nx, train_y=train_y)
    # other.pre_processing()
    # other.train()
    # print('rss other:', other.rss)
    print('F-statistic', lsm.F_statistic(remove_cols=['age', 'lcp', 'gleason', 'pgg45']))
    print('z-score', lsm.z_score)

    result = lsm.test(test_x, test_y)

    print('test error: ', result.mse)

    from sklearn.linear_model import LinearRegression

    lr = LinearRegression()

    lr.fit(train_x, train_y)
    assert np.isclose(result.mse, np.mean(((lr.predict(test_x)) - test_y) **2))


def test_best_select(prostate_data):
    from ESLmodels.ch3.model import BestSubsetSelection
    from ESLmodels.ch3.model import LeastSquareModel
    train_x, train_y, test_x, test_y, features = prostate_data
    bss = BestSubsetSelection(train_x=train_x, train_y=train_y, k=2, features_name=features)
    bss.pre_processing()
    bss.train()
    print(bss.select_column)
    print('cof', bss.beta_hat)
    print('rss:', bss.rss)
    print('test err', bss.test(test_x, test_y).mse)
    lsm = LeastSquareModel(train_x=train_x[:,:2], train_y=train_y, features_name=features)
    lsm.pre_processing()

    lsm.train()

    assert lsm.rss == bss.rss
    # print('gg', sum((bss.pre_processing_x(train_x) @ lsm.beta_hat)**2))
    # assert  0

def test_ridge(prostate_data):
    from ESLmodels.ch3.model import RidgeModel
    train_x, train_y, test_x, test_y, features = prostate_data
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeCV, ridge_regression

    # als=[(i/100) for i in range(1, 1000)]
    # rcv = RidgeCV(alphas=[22.78])
    # rcv.fit(std(train_x), train_y)
    # print(rcv.alpha_)
    # print(rcv.intercept_, rcv.coef_)
    # print('te',np.mean((rcv.predict(std(test_x)) -test_y)**2))

    target_df = 5
    min_df = 9999999999999999
    best_alpha = None
    for i in range(1, 5000):
        alpha = i/200
        r = RidgeModel(train_x=train_x, train_y=train_y, alpha=alpha, solve='svd')
        r.pre_processing()
        r.train()
        if abs(r.df - target_df) < min_df:
            best_alpha = alpha
            min_df = r.df - target_df

    r = RidgeModel(train_x=train_x, train_y=train_y, alpha=best_alpha, solve='raw')
    r.pre_processing()
    r.train()
    # print(ridge_regression(r.standardize(train_x), train_y, alpha=best_alpha, solver='auto'))

    print('df', r.df)
    print('alpha', r.alpha)
    # print('rss', min_df, r.rss)
    test_error = r.test(test_x, test_y).mse
    print(test_error)
    print(r.beta_hat)

    assert float('{:.3f}'.format(test_error)) == 0.492


def _test_lars_lasso(prostate_data):
    train_x, train_y, test_x, test_y, features = prostate_data

    from sklearn.linear_model.least_angle import LassoLars
    from sklearn.preprocessing import StandardScaler

    std_train_x = (train_x - train_x.mean(axis=0)) / train_x.std(axis=0, ddof=1)

    alpha = 0.3
    larl = LassoLars(alpha=alpha, verbose=2)

    larl.fit(std_train_x, train_y)

    # print(larl.alphas_)
    # print(larl.active_)
    # print(larl.intercept_, larl.coef_)

    # from ESLmodels.ch3.model import LassoLARModel
    # lar = LassoLARModel(train_x=train_x, train_y=train_y, alpha=alpha)
    # lar.pre_processing()
    # lar.train()
    # print('lar', lar.beta_hat)
    # print(' lar act', lar.active)

    assert  0

def test_PCR(prostate_data):
    train_x, train_y, test_x, test_y, features = prostate_data
    from ESLmodels.ch3.model import PrincipalComponentsRegression
    # page 80 says m=7
    pcr = PrincipalComponentsRegression(train_x=train_x, train_y=train_y, m=7)
    pcr.pre_processing()
    pcr.train()
    print('beta hat', pcr.beta_hat)
    test_err = pcr.test(test_x, test_y).mse
    print('test error:', test_err)

    assert digit_float(test_err) == 0.448


def test_PLS(prostate_data):
    train_x, train_y, test_x, test_y, features = prostate_data
    from ESLmodels.ch3.model import PartialLeastSquare
    pls = PartialLeastSquare(train_x=train_x, train_y=train_y, M=2)
    pls.pre_processing()
    pls.train()
    print(pls.beta_hat)
    te = pls.test(test_x, test_y).mse
    print('te', te)
    assert digit_float(te) == 0.536


def test_datasets():
    from ESLmodels.datasets import ProstateDataSet
    p = ProstateDataSet()
    print(p.test_y)
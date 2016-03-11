from ESLmodels.ch3.model import LeastSquareModel
import pytest
from .utils import read_data
import os
from ESLmodels.base import np

@pytest.fixture
def data():
    DIR = os.path.dirname(__file__)
    filename = os.path.join(DIR, 'data/prostate.txt')
    prostate_data = read_data(filename)
    prostate_data.drop(prostate_data.columns[0], axis=1, inplace=True)
    return prostate_data


@pytest.fixture
def prostate_data(data):

    train_data = data[data.train == 'T'].drop('train', axis=1)
    test_data = data[data.train == 'F'].drop('train', axis=1)

    train_y = train_data.pop('lpsa')
    train_x = train_data.copy()

    test_y = test_data.pop('lpsa')
    test_x = test_data.copy()

    features = list(train_x.columns)
    return train_x.values, train_y.values, features, test_x.values, test_y.values


def _test_least_square_model(prostate_data):
    train_x, train_y, features, test_x, test_y = prostate_data
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


def _test_best_select(prostate_data):
    from ESLmodels.ch3.model import BestSubsetSelection
    train_x, train_y, features, test_x, test_y = prostate_data
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
    train_x, train_y, features, test_x, test_y = prostate_data
    from sklearn.preprocessing import scale
    from sklearn.linear_model import RidgeCV, ridge_regression

    # als=[(i/100) for i in range(1, 10000)]
    # rcv = RidgeCV(alphas=als, solver='svd')
    # rcv.fit(scale(train_x), train_y)
    # print(rcv.alpha_)
    # print(rcv.intercept_, rcv.coef_)
    # print('te',np.mean((rcv.predict(scale(test_x)) -test_y)**2))

    # min_df = 99999999999999999
    # best_alpha = None
    # for i in range(1, 1000):
    #     alpha = i/100
    #     r = RidgeModel(train_x=train_x, train_y=train_y, alpha=alpha, solve='svd')
    #     r.pre_processing()
    #     r.train()
    #     if abs(r.df - 7) < min_df:
    #         best_alpha = alpha
    #         min_df = r.df - 7

    r = RidgeModel(train_x=train_x, train_y=train_y, alpha=4, solve='svd')
    r.pre_processing()
    r.train()
    print(ridge_regression(scale(train_x), train_y, alpha=4, solver='svd'))

    # print('df', r.df)
    # print('alpha', r.alpha)
    # print('rss', min_df, r.rss)
    print(r.test(test_x, test_y).mse)
    # print(skr.pred)
    print(r.beta_hat)
    assert 0
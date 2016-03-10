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


def test_least_square_model(prostate_data):
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

    print('test error: ', result.rss)

    from sklearn.linear_model import LinearRegression

    lr = LinearRegression()

    lr.fit(train_x, train_y)
    print('sklean', np.mean(((lr.predict(test_x)) - test_y) **2))


    assert 0
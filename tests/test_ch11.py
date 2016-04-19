import pytest
import numpy as np


@pytest.fixture
def zipcode_data():
    from esl_model.datasets import ZipCodeDataSet
    d = ZipCodeDataSet()

    return d.return_all()

def test_zipcode(zipcode_data):
    train_x, train_y, test_x, test_y, features = zipcode_data
    assert train_y[0] == 6
    assert test_y[1] == 6
    assert train_x[0, 1] == -1


def test_nn1(zipcode_data):
    from esl_model.ch11.models import NeuralNetworkN1
    train_x, train_y, test_x, test_y, features = zipcode_data

    model = NeuralNetworkN1(train_x[:320], train_y[:320], n_class=10, alpha=0.01)
    model.pre_processing()
    model.train()

    print(model.y_hat[:10].flatten())
    print(train_y[:10])
    print(model.rss)
    print(model.test(test_x, test_y).error_rate)


    # from esl_model.ch4.models import LinearRegressionIndicatorMatrix as g
    # m = g(train_x, train_y, n_class=10)
    # m.pre_processing()
    # m.train()
    # print('lda', m.beta_hat)
    assert 0

def test_n1(zipcode_data):
    from esl_model.ch11.models import NN1
    train_x, train_y, test_x, test_y, features = zipcode_data

    model = NN1(train_x, train_y, n_class=10, alpha=0.1)
    model.pre_processing()
    model.train()

    print(model.y_hat[:10].flatten())
    print(train_y[:10])
    print(model.rss)
    print(model.test(test_x, test_y).error_rate)
    assert 0

def test_nn2(zipcode_data):
    from esl_model.ch11.models import NN2
    train_x, train_y, test_x, test_y, features = zipcode_data

    model = NN2(train_x[:520], train_y[:520], n_class=10, alpha=1, iter_time=150)
    model.pre_processing()
    model.train()

    print(model.y_hat[:10].flatten())
    print(train_y[:10])
    print(model.rss)
    print(model.test(test_x, test_y).error_rate)
    assert 0
    from sklearn.neural_network import BernoulliRBM
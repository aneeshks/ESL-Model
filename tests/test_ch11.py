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


def test_intuitive_nn1(zipcode_data):
    from esl_model.ch11.models import IntuitiveNeuralNetworkN1
    train_x, train_y, test_x, test_y, features = zipcode_data

    model = IntuitiveNeuralNetworkN1(train_x[:320], train_y[:320], n_class=10, alpha=0.01, n_iter=10)
    model.pre_processing()
    model.train()

    print(model.y_hat[:10].flatten())
    print(train_y[:10])
    print(model.rss)
    print(model.test(test_x, test_y).error_rate)


def test_intuitive_batch_nn1(zipcode_data):
    from esl_model.ch11.models import IntuitiveMiniBatchNN1
    train_x, train_y, test_x, test_y, features = zipcode_data

    model = IntuitiveMiniBatchNN1(train_x[:320], train_y[:320], n_class=10, alpha=0.01, n_iter=5, mini_batch=10)
    model.pre_processing()
    model.train()

    print(model.y_hat[:10].flatten())
    print(train_y[:10])
    print(model.rss)
    print(model.test(test_x, test_y).error_rate)


def test_intuitive_nn2(zipcode_data):
    from esl_model.ch11.models import IntuitiveNeuralNetwork2
    train_x, train_y, test_x, test_y, features = zipcode_data

    model = IntuitiveNeuralNetwork2(train_x[:320], train_y[:320], n_class=10, alpha=1, iter_time=10)
    model.pre_processing()
    model.train()

    print(model.y_hat[:10].flatten())
    print(train_y[:10])
    print(model.rss)
    print(model.test(test_x, test_y).error_rate)



def test_nn1(zipcode_data):
    from esl_model.ch11.models import MiniBatchNN
    train_x, train_y, test_x, test_y, features = zipcode_data

    model = MiniBatchNN(train_x[:520], train_y[:520], n_class=10, alpha=0.45, n_iter=25, mini_batch=10,
                        hidden_layer_shape=None)
    model.pre_processing()
    model.train()

    print(model.y_hat[:10].flatten())
    print(train_y[:10])
    print(model.rss)
    err = model.test(test_x, test_y).error_rate
    print(err)
    assert err < 0.24


def test_nn2(zipcode_data):
    from esl_model.ch11.models import MiniBatchNN
    train_x, train_y, test_x, test_y, features = zipcode_data

    model = MiniBatchNN(train_x[:320], train_y[:320], n_class=10, alpha=0.44, n_iter=25, mini_batch=5,
                        hidden_layer_shape=[12])
    model.pre_processing()
    model.train()

    print(model.y_hat[:10].flatten())
    print(train_y[:10])
    print(model.rss)
    err = model.test(test_x, test_y).error_rate
    print(err)
    assert err < 0.24


def test_nn3(zipcode_data):
    from esl_model.ch11.models import LocallyConnectNN
    train_x, train_y, test_x, test_y, features = zipcode_data
    model = LocallyConnectNN(train_x[:320], train_y[:320], n_class=10, alpha=0.97, n_iter=30, mini_batch=10,
                        hidden_layer_shape=[(8,8), (4,4)], filter_shapes=[(3,3), (5,5)], stride=2)

    model.pre_processing()
    model.train()

    print(model.y_hat[:16].flatten())
    print(train_y[:16])
    print(model.rss)
    err = model.test(test_x, test_y).error_rate
    print(err)
    assert err < 0.22

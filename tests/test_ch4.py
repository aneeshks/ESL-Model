import pytest
from .utils import digit_float
import numpy as np


vowel_data_y_dimension = 11


@pytest.fixture
def vowel_data():
    from esl_model.datasets import VowelDataSet
    data = VowelDataSet()
    return data.return_all()


@pytest.fixture
def SAHeart_data():
    from esl_model.datasets import SAHeartDataSet
    data = SAHeartDataSet()
    return data.return_all()


def test_vowel_data():
    from esl_model.datasets import VowelDataSet
    data = VowelDataSet()
    assert list(data.train_y[:5]) == list(range(1,6))


def test_indicator_matrix(vowel_data):
    from esl_model.ch4.model import LinearRegressionIndicatorMatrix

    train_x, train_y, test_x, test_y, features = vowel_data

    lrm = LinearRegressionIndicatorMatrix(train_x=train_x, train_y=train_y, K=vowel_data_y_dimension)
    lrm.pre_processing()
    lrm.train()
    print(lrm.error_rate)
    test_result= lrm.test(test_x, test_y)
    print(test_result.error_rate)

    assert digit_float(lrm.error_rate) == 0.477
    assert digit_float(test_result.error_rate) == 0.667


def test_LDA(vowel_data):
    from esl_model.ch4.model import LDAModel
    train_x, train_y, test_x, test_y, features = vowel_data

    lda = LDAModel(train_x=train_x, train_y=train_y, K=vowel_data_y_dimension)
    lda.pre_processing()
    lda.train()

    print(lda.y_hat[:10])
    print(lda.error_rate)

    te = lda.test(test_x, test_y)
    print(te.error_rate)

    assert digit_float(lda.error_rate) == 0.316
    assert digit_float(te.error_rate) == 0.556


def test_QDA(vowel_data):
    from esl_model.ch4.model import QDAModel
    train_x, train_y, test_x, test_y, features = vowel_data

    qda = QDAModel(train_x=train_x, train_y=train_y, K=vowel_data_y_dimension)
    qda.pre_processing()
    qda.train()

    print(qda.y_hat[:10])
    print(qda.error_rate)
    te = qda.test(test_x, test_y).error_rate
    print(te)
    assert digit_float(qda.error_rate) == 0.011
    assert digit_float(te) == 0.528


def test_RDA(vowel_data):
    from esl_model.ch4.model import RDAModel
    train_x, train_y, test_x, test_y, features = vowel_data

    # http://waxworksmath.com/Authors/G_M/Hastie/WriteUp/weatherwax_epstein_hastie_solutions_manual.pdf
    # pp 60
    model = RDAModel(train_x=train_x, train_y=train_y, K=vowel_data_y_dimension, alpha=0.969697)
    model.pre_processing()
    model.train()



    print(model.error_rate)
    te = model.test(test_x, test_y)
    print(te.error_rate)
    assert digit_float(te.error_rate) == 0.478


def test_LDA_computation(vowel_data):
    from esl_model.ch4.model import LDAForComputation
    train_x, train_y, test_x, test_y, features = vowel_data

    model = LDAForComputation(train_x=train_x, train_y=train_y, K=vowel_data_y_dimension)
    model.pre_processing()
    model.train()

    from esl_model.ch4.model import LDAModel
    lda = LDAModel(train_x=train_x, train_y=train_y, K=vowel_data_y_dimension)
    lda.pre_processing()
    lda.train()
    print(model.error_rate)

    assert np.isclose(model.error_rate, lda.error_rate)
    assert np.isclose(model.test(test_x, test_y).error_rate, lda.test(test_x, test_y).error_rate)


def test_RRLDA(vowel_data):
    from esl_model.ch4.model import ReducedRankLDAModel
    train_x, train_y, test_x, test_y, features = vowel_data

    model = ReducedRankLDAModel(train_x=train_x, train_y=train_y, K=vowel_data_y_dimension, L=2)
    model.pre_processing()
    model.train()

    print(model.y_hat[:5])
    print(model.error_rate)

    te = model.test(test_x, test_y)
    print(te.error_rate)

    assert digit_float(model.error_rate) == 0.350
    assert digit_float(te.error_rate) == 0.491


def test_SAHeart_data_set(SAHeart_data):
    x, y, *_ = SAHeart_data
    assert x[1,2] == 4.41
    assert list(y[:4]) == [1, 1, 0, 1]


def test_binary_logistic_regression(SAHeart_data):
    train_x, train_y, *_ = SAHeart_data

    from esl_model.ch4.model import BinaryLogisticRegression
    model = BinaryLogisticRegression(train_x=train_x, train_y=train_y, K=2)
    # model.pre_processing()
    model.train()
    print(model.beta_hat)
    print(model.error_rate)
    print(model.y_hat[:5])

    assert 0
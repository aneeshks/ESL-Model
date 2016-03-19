import pytest
from .utils import digit_float
import numpy as np

vowel_data_y_dimension = 11

@pytest.fixture
def vowel_data():
    from esl_model.datasets import VowelDataSet
    data = VowelDataSet()
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



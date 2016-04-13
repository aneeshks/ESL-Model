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


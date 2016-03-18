import pytest
from .utils import digit_float
import numpy as np


@pytest.fixture
def vowel_data():
    from esl_model.datasets import VowelDataSet
    data = VowelDataSet()
    return data.return_all()


def test_vowel_data():
    from esl_model.datasets import VowelDataSet
    data = VowelDataSet()
    assert list(data.train_y[:5]) == list(range(1,6))


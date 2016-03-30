import pandas as pd
import numpy as np
from pandas import read_csv
from os.path import join, dirname


__all__ = ['ProstateDataSet', 'VowelDataSet', 'SAHeartDataSet']


class BaseDataSet:
    data_path = ''
    multi_data = False

    def __init__(self, select_features=None):
        self._train_x = None
        self._train_y = None
        self._test_x = None
        self._test_y = None
        self.feature_names = None
        self.select_features = select_features

        if self.multi_data:
            self.df = map(self.read_data, self.data_path)
        else:
            self.df = self.read_data(self.data_path)

        self._process_data()

    @staticmethod
    def read_data(path):
        filename = join(dirname(__file__), path)
        return read_csv(filename)

    def _process_data(self):
        raise NotImplementedError

    def return_all(self, ret_feature_names=True):
        """
        all data sets
        :param ret_feature_names:
        :return:
        """
        ret = (self.train_x, self.train_y, self.test_x, self.test_y, self.feature_names)
        return ret if ret_feature_names else ret[:-1]

    def select_x(self, x: pd.DataFrame):
        """
        select special column by features name or column number
        """
        if not self.feature_names or not self.select_features:
            return x

        if isinstance(self.select_features[0], int):
            return x.iloc[:, self.select_features]

        elif all(f in self.feature_names for f in self.select_features):
            return x.loc[:, self.select_features]
        else:
            raise KeyError('Not find features in {}'.format(self.select_features))

    @property
    def train_x(self):
        return self.select_x(self._train_x).values

    @property
    def train_y(self):
        return self._train_y.values

    @property
    def test_x(self):
        return self.select_x(self._test_x).values

    @property
    def test_y(self):
        return self._test_y.values


class ProstateDataSet(BaseDataSet):
    data_path = 'data/prostate.csv'

    def _process_data(self):
        df = self.df
        train = self.df[self.df.train == 'T'].iloc[:, :-1]
        test = self.df[self.df.train == 'F'].iloc[:, :-1]

        self._train_x, self._test_x = train.iloc[:, :-1], test.iloc[:, :-1]
        self._train_y, self._test_y = train.iloc[:, -1], test.iloc[:, -1]
        self.feature_names = list(df.columns[:-1])


class VowelDataSet(BaseDataSet):
    data_path = ['data/vowel.train.csv', 'data/vowel.test.csv']
    multi_data = True

    def _process_data(self):
        train, test = self.df

        train = train.drop(train.columns[0], axis=1)
        self._train_y = train.pop('y')
        self._train_x = train

        test = test.drop(test.columns[0], axis=1)
        self._test_y = test.pop('y')
        self._test_x = test
        self.feature_names = list(train.columns)


class SAHeartDataSet(BaseDataSet):
    """
    There is not test data in this DataSet
    """

    data_path = 'data/SAheart.data.csv'

    def _process_data(self):
        df = self.df
        train = df.drop(df.columns[0], axis=1)
        self._train_y = train.pop('chd')
        train['famhist'] = pd.Categorical(train['famhist']).codes

        self._train_x = train
        self.feature_names = list(train.columns)

        # empty test data set
        self._test_x = self._test_y = pd.DataFrame()


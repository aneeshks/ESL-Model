from pandas import read_csv
from os.path import join, dirname

__all__ = ['ProstateDataSet', 'VowelDataSet']


class BaseDataSet:
    data_path = ''
    multi_data = False

    def __init__(self):
        self.data = None
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.feature_names = None


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


class ProstateDataSet(BaseDataSet):
    data_path = 'data/prostate.csv'

    def _process_data(self):
        df = self.df
        train = self.df[self.df.train == 'T'].iloc[:, :-1]
        test = self.df[self.df.train == 'F'].iloc[:, :-1]

        self.train_x, self.test_x = train.iloc[:, :-1].values, test.iloc[:, :-1].values
        self.train_y, self.test_y = train.iloc[:, -1].values, test.iloc[:, -1].values
        self.feature_names = list(df.columns[:-1])


class VowelDataSet(BaseDataSet):
    data_path = ['data/vowel.train.csv', 'data/vowel.test.csv']
    multi_data = True

    def _process_data(self):
        train, test = self.df

        train = train.drop(train.columns[0], axis=1)
        self.train_y = train.pop('y').values
        self.train_x = train.values

        test = test.drop(test.columns[0], axis=1)
        self.test_y = test.pop('y').values
        self.test_x = test.values

        self.feature_names = list(train.columns)

class SAHeartDataSet(BaseDataSet):
    data_path = 'data/SAheart.data.csv'

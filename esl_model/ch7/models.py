"""Apply K-folds CV for some model define in ch3 and ch4.

Note:All CV get approximately result as book, but most CV use one standard rule can not get same best alpha with book.
     If I use CV with randomize train data, I can get the result as same as book most time.
"""

import numpy as np
from ..ch3.models import *
from ..ch4.models import *


DIRECTION_LEFT = 'left'
DIRECTION_RIGHT = 'right'


class BaseCV:
    _bound_model = None
    # name that model use. ex, PCR use `m`, Ridge use `alpha`.
    _cv_field_name = 'alpha'
    _inc_regularization_direction = DIRECTION_LEFT
    _one_standard_rule = True

    def __init__(self, train_x, train_y, features_name=None, do_standardization=True,
                 k_folds=10, alphas=None, random=False, select_train_method='step', **kwargs):
        self.train_x = train_x
        self.train_y = train_y
        self.k_folds = k_folds
        self.alphas = alphas
        self.random = random
        self.select_train_method = select_train_method
        self.kwargs = kwargs

        self.kwargs['features_name'] = features_name
        self.kwargs['do_standardization'] = do_standardization

    def pre_processing(self):
        """Provide same API as Model, we split data to K folds here.
        """
        if self.random:
            mask = np.random.permutation(self.train_x.shape[0])
            train_x = self.train_x[mask]
            train_y = self.train_y[mask]
        else:
            train_x = self.train_x[:]
            train_y = self.train_y[:]

        if self.select_train_method == 'step':
            self.x_folds = [train_x[i::self.k_folds] for i in range(0, self.k_folds)]
            self.y_folds = [train_y[i::self.k_folds] for i in range(0, self.k_folds)]
        else:
            self.x_folds = np.array_split(train_x, self.k_folds)
            self.y_folds = np.array_split(train_y, self.k_folds)


        # for i in range(self.k_folds):
        #     self.x_folds[i] = self.train_x[0] + self.x_folds[i] + self.train_x[-1]
        #     self.y_folds[i] = self.train_y[0] + self.y_folds[i] + self.train_y[-1]


    @staticmethod
    def combine_train_folds(folds, exclude):
        """
        :return a matrix combine folds exclude specify index
        """
        train_list = folds[:exclude]
        train_list.extend(folds[exclude + 1:])
        mat = np.concatenate(train_list)
        return mat

    def _model_test(self, model, cv_x, cv_y):
        return model.test(cv_x, cv_y).mse

    def train(self):
        """
        calculate cv error and cv std error, then use one standard error to choose best alpha.

        reference
        -----------
        http://www.stat.cmu.edu/~ryantibs/datamining/lectures/19-val2-marked.pdf
        http://www.stat.cmu.edu/~ryantibs/datamining/lectures/18-val1-marked.pdf
        """

        alphas = self.alphas
        alpha_errs = np.zeros((len(alphas), 1)).flatten()
        alpha_std_errs = np.zeros((len(alphas), 1)).flatten()

        for idx, alpha in enumerate(alphas):
            err = np.zeros((self.k_folds, 1)).flatten()
            foldwise_err = np.zeros((self.k_folds, 1)).flatten()

            for k in range(self.k_folds):
                train_x = self.combine_train_folds(self.x_folds, exclude=k)
                train_y = self.combine_train_folds(self.y_folds, exclude=k)
                cv_x = self.x_folds[k]
                cv_y = self.y_folds[k]

                kwargs = self.kwargs.copy()
                kwargs[self._cv_field_name] = alpha
                model = self._bound_model(train_x, train_y, **kwargs)
                model.pre_processing()
                model.train()

                foldwise_err[k] = self._model_test(model, cv_x, cv_y)
                err[k] = self._model_test(model, cv_x, cv_y) * cv_x.shape[0]

            std_err = (np.var(foldwise_err) **0.5) / (self.k_folds**0.5)
            print('err', err)
            # std_err = foldwise_err.std() / (self.k_folds**0.5)
            # tot_err = sum(err) / (self.train_x.shape[0])
            tot_err = sum(err) / sum(len(x) for x in self.x_folds)
            alpha_std_errs[idx] = std_err
            alpha_errs[idx] = tot_err

        if self._one_standard_rule:
            # use one standard error rule to find best alpha
            alpha_hat_idx =  alpha_errs.argmin()
            # we move alpha for cease the (cv)_alpha <= (cv)_alpha_hat + (cv_std)_alpha_hat
            cv_hat = alpha_errs[alpha_hat_idx] + alpha_std_errs[alpha_hat_idx]

            if self._inc_regularization_direction is DIRECTION_LEFT:
                move_direction = reversed(range(0, alpha_hat_idx+1))
            else:
                move_direction = range(alpha_hat_idx, len(alphas))

            print('alphas len', len(alphas))
            print('alpha_hat idx', alpha_hat_idx)
            print('cv hat', cv_hat)


            self.best_alpha = -1
            # find the best_alpha
            last_idx = None
            for idx in move_direction:
                if (alpha_errs[idx] > cv_hat) and (last_idx and alpha_errs[last_idx] <= cv_hat):
                    self.best_alpha = alphas[last_idx]
                    #break
                last_idx = idx
        else:
            self.best_alpha = alphas[alpha_errs.argmin()]



        self.alpha_errs = alpha_errs
        self.alpha_std_errs = alpha_std_errs
        kwargs = self.kwargs.copy()
        kwargs[self._cv_field_name] = self.best_alpha
        model = self._bound_model(self.train_x, self.train_y, **kwargs)
        model.pre_processing()
        model.train()
        self.model = model

    def __getattr__(self, name):
        # make a proxy for self.model, when call method not define in CV model, use the method find in self.model.
        return getattr(self.model, name)


class BaseLogisticCV(BaseCV):
    def _model_test(self, model, cv_x, cv_y):
        return model.test(cv_x, cv_y).error_rate


class RidgeCV(BaseCV):
    _bound_model = RidgeModel
    _inc_regularization_direction = DIRECTION_LEFT


class PCRCV(BaseCV):
    _bound_model = PrincipalComponentsRegression
    _cv_field_name = 'm'
    _inc_regularization_direction = DIRECTION_LEFT


class PartialLeastSquareCV(BaseCV):
    _bound_model = PartialLeastSquare
    _cv_field_name = 'M'
    _inc_regularization_direction = DIRECTION_LEFT


class BestSubsetSelectionCV(BaseCV):
    _bound_model = BestSubsetSelection
    _cv_field_name = 'k'
    _inc_regularization_direction = DIRECTION_LEFT


class RDACV(BaseLogisticCV):
    _bound_model = RDAModel
    _cv_field_name = 'alpha'
    _one_standard_rule = False

    def _model_test(self, model, cv_x, cv_y):
        X = model._pre_processing_x(cv_x)
        N = X.shape[0]

        err = 0
        for k in range(model.n_class):

            d = model.quadratic_discriminant_func(X, k)
            t = d.diagonal()
            err += sum(t[cv_y ==(k+1)])

        return err*(-2) / N

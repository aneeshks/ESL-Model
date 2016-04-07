"""Apply K-folds CV for some model define in ch3 and ch4.
"""

from ..ch3.models import LinearModel, RidgeModel, PrincipalComponentsRegression
import numpy as np

class BaseCV:
    _bound_model = None
    # name that model use. ex, PCR use `m`, Ridge use `alpha`.
    _cv_field_name = 'alpha'

    def __init__(self, train_x, train_y, features_name=None, do_standardization=True,
                 k_folds=10, alphas=None, **kwargs):
        self.train_x = train_x
        self.train_y = train_y
        self.k_folds = k_folds
        self.alphas = alphas
        self.kwargs = kwargs

        self.kwargs['features_name'] = features_name
        self.kwargs['do_standardization'] = do_standardization

    def pre_processing(self):
        """Provide same API as Model, we split data to K folds here.
        """
        # `//` make result is int
        # self._x = self.train_x.copy()
        # np.random.shuffle(self.train_y)
        fold_size = self.train_x.shape[0] // self.k_folds
        residue_item = self.train_x.shape[0] - fold_size * self.k_folds

        self.x_folds = [self.train_x[i: i+fold_size] for i in range(0, fold_size * self.k_folds, fold_size)]
        self.y_folds = [self.train_y[i: i+fold_size] for i in range(0, fold_size * self.k_folds, fold_size)]
        print('now', len(self.x_folds))
        if residue_item:
            residue_x = self.train_x[-residue_item:]
            residue_y = self.train_y[-residue_item:]
            for i in range(self.k_folds):
                self.x_folds[i] = np.append(self.x_folds[i], residue_x, axis=0)
                self.y_folds[i] = np.append(self.y_folds[i], residue_y, axis=0)

        # self.train_x = self._x

    def _get_cv_alphas(self):
        raise NotImplemented

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
        alphas = self.alphas
        alpha_errs = np.zeros((len(alphas), 1)).flatten()
        for idx, alpha in enumerate(alphas):
            err = np.zeros((self.k_folds, 1)).flatten()

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

                err[k] = self._model_test(model, cv_x, cv_y) * cv_x.shape[0]
            # std_err = (np.var(err) **0.5) / (self.k_folds**0.5)
            print('err vector', repr(err))
            std_err = sum(err) / (len(self.x_folds[0]) * self.k_folds)
            alpha_errs[idx] = std_err

        self.best_alpha = alphas[alpha_errs.argmin()]
        self.alpha_errs = alpha_errs
        kwargs = self.kwargs.copy()
        kwargs[self._cv_field_name] = self.best_alpha
        model = self._bound_model(self.train_x, self.train_y, **kwargs)
        model.pre_processing()
        model.train()
        self.model = model

    def __getattr__(self, name):
        # make a proxy for self.model, when call method not define in CV model, use the method find in self.model.
        return getattr(self.model, name)



class RidgeCV(BaseCV):
    _bound_model = RidgeModel


class PCRCV(BaseCV):
    _bound_model = PrincipalComponentsRegression
    _cv_field_name = 'm'

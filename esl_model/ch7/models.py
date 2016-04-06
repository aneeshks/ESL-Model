"""Apply K-folds CV for some model define in ch3 and ch4.
"""

from ..ch3.models import LinearModel, RidgeModel
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
        fold_size = self.train_x.shape[0] // self.k_folds
        self.x_folds = [self.train_x[i: i+fold_size] for i in range(0, len(self.train_x), fold_size)]
        self.y_folds = [self.train_y[i: i+fold_size] for i in range(0, len(self.train_y), fold_size)]

        if len(self.x_folds[-1]) != fold_size:
            residue_x = self.x_folds.pop(-1)
            residue_y = self.y_folds.pop(-1)
            for i in range(self.k_folds):
                self.x_folds[i] = np.append(self.x_folds[i], residue_x, axis=0)
                self.y_folds[i] = np.append(self.y_folds[i], residue_y, axis=0)

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
            err = 0
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

                err += self._model_test(model, cv_x, cv_y)
            alpha_errs[idx] = err / self.train_x.shape[0]

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

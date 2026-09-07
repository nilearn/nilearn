"""Replacement for sklearn mixins."""

from typing import Self

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import ClassifierTags, RegressorTags

from nilearn._utils.docs import fill_doc
from nilearn._utils.param_validation import (
    check_params,
)


class _ClassifierMixin:
    _estimator_type = "classifier"  # TODO (sklearn >= 1.8) remove

    def __sklearn_tags__(self):
        """Return estimator tags.

        See the sklearn documentation for more details on tags
        https://scikit-learn.org/1.6/developers/develop.html#estimator-tags
        """
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.classifier_tags = ClassifierTags()

        return tags

    def _n_problems(self):
        if self.n_classes_ > 2:
            return self.n_classes_
        else:
            return 1

    def _binarize_y(self, y):
        """Encode target classes as -1 and 1.

        Helper function invoked just before fitting a classifier.
        """
        y = np.array(y)

        self._enc = LabelBinarizer(pos_label=1, neg_label=-1)
        y = self._enc.fit_transform(y)
        self.classes_ = self._enc.classes_
        self.n_classes_ = len(self.classes_)
        return y

    def _get_classes(self):
        return self.classes_


class _RegressorMixin:
    _estimator_type = "regressor"  # TODO (sklearn >= 1.8) remove

    def __sklearn_tags__(self):
        """Return estimator tags.

        See the sklearn documentation for more details on tags
        https://scikit-learn.org/1.6/developers/develop.html#estimator-tags
        """
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        tags.regressor_tags = RegressorTags()

        return tags

    @fill_doc
    def fit(self, X, y, groups=None) -> Self:
        """Fit the decoder (learner).

        Parameters
        ----------
        X : :obj:`list` of Niimg-like \
            or :obj:`~nilearn.surface.SurfaceImage` objects
            See :ref:`extracting_data`.
            Data on which model is to be fitted.
            If this is a list,
            the affine is considered the same for all.

        y : numpy.ndarray of shape=(n_samples) or list of length n_samples
            The dependent variable (age, sex, IQ, yes/no, etc.).
            Target variable to predict.
            Must have exactly as many elements as the input images.

        %(groups)s

        """
        check_params(self.__dict__)
        self._classes_ = ["beta"]
        # _RegressorMixin is only ever used together with a class
        # defining fit() (e.g. _BaseDecoder, BaseSpaceNet).
        return super().fit(X, y, groups=groups)  # type: ignore[misc]

    def _n_problems(self):
        return 1

    def _binarize_y(self, y):
        return y[:, np.newaxis]

    def _get_classes(self):
        return self._classes_

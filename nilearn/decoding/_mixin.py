"""Replacement for sklearn mixins."""

import numpy as np
from sklearn.preprocessing import LabelBinarizer

from nilearn._utils.tags import SKLEARN_LT_1_6


class _ClassifierMixin:
    _estimator_type = "classifier"  # TODO (sklearn >= 1.6) remove

    def __sklearn_tags__(self):
        """Return estimator tags.

        See the sklearn documentation for more details on tags
        https://scikit-learn.org/1.6/developers/develop.html#estimator-tags
        """
        # TODO (sklearn >= 1.6) get rid of if block
        # when bumping sklearn_version > 1.5
        # see https://github.com/scikit-learn/scikit-learn/pull/29677
        tags = super().__sklearn_tags__()
        if SKLEARN_LT_1_6:
            return tags

        from sklearn.utils import ClassifierTags

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


class _RegressorMixin:
    _estimator_type = "regressor"  # TODO (sklearn >= 1.6) remove

    def __sklearn_tags__(self):
        """Return estimator tags.

        See the sklearn documentation for more details on tags
        https://scikit-learn.org/1.6/developers/develop.html#estimator-tags
        """
        # TODO (sklearn >= 1.6) get rid of if block
        # when bumping sklearn_version > 1.5
        # see https://github.com/scikit-learn/scikit-learn/pull/29677
        tags = super().__sklearn_tags__()
        if SKLEARN_LT_1_6:
            tags["multioutput"] = True
            return tags
        from sklearn.utils import RegressorTags

        tags.estimator_type = "regressor"
        tags.regressor_tags = RegressorTags()

        return tags

    def _n_problems(self):
        return 1

    def _binarize_y(self, y):
        return y[:, np.newaxis]

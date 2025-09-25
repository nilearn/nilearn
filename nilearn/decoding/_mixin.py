"""Replacement for sklearn mixins."""

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted

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

    def _set_classes(self, y):
        """Encode target classes as -1 and 1.

        Helper function invoked just before fitting a classifier.
        """
        y = np.array(y)

        enc = LabelBinarizer(pos_label=1, neg_label=-1)
        y = enc.fit_transform(y)
        self.classes_ = enc.classes_
        return y

    def _get_classes(self):
        check_is_fitted(self)
        return self.classes_

    @property
    def n_classes_(self) -> int:
        """Return number of classes."""
        check_is_fitted(self)
        return len(self.classes_)


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

    def _set_classes(self, y):
        self._classes_ = ["beta"]
        return y[:, np.newaxis]

    def _get_classes(self):
        check_is_fitted(self)
        return self._classes_

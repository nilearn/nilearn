"""Replacement for sklearn mixins."""

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

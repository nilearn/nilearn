"""Replacement for sklearn mixins."""

from sklearn.utils.validation import check_is_fitted

from nilearn._utils.tags import SKLEARN_LT_1_6


class _ClassifierMixin:
    # TODO remove for sklearn>=1.6
    _estimator_type = "classifier"

    def decision_function(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : Niimg-like, :obj:`list` of either \
            Niimg-like objects or :obj:`str` or path-like
            See :ref:`extracting_data`.
            Data on prediction is to be made. If this is a list,
            the affine is considered the same for all.

        Returns
        -------
        y_pred : :class:`numpy.ndarray`, shape (n_samples,)
            Predicted class label per sample.
        """
        check_is_fitted(self)
        return self._decision_function(X)

    def __sklearn_tags__(self):
        """Return estimator tags.

        See the sklearn documentation for more details on tags
        https://scikit-learn.org/1.6/developers/develop.html#estimator-tags
        """
        # TODO
        # get rid of if block
        # bumping sklearn_version > 1.5
        # see https://github.com/scikit-learn/scikit-learn/pull/29677
        tags = super().__sklearn_tags__()
        if SKLEARN_LT_1_6:
            return tags

        from sklearn.utils import ClassifierTags

        tags.estimator_type = "classifier"
        tags.classifier_tags = ClassifierTags()

        return tags


class _RegressorMixin:
    # TODO remove for sklearn>=1.6
    _estimator_type = "regressor"

    def __sklearn_tags__(self):
        """Return estimator tags.

        See the sklearn documentation for more details on tags
        https://scikit-learn.org/1.6/developers/develop.html#estimator-tags
        """
        # TODO
        # get rid of if block
        # bumping sklearn_version > 1.5
        # see https://github.com/scikit-learn/scikit-learn/pull/29677
        tags = super().__sklearn_tags__()
        if SKLEARN_LT_1_6:
            tags["multioutput"] = True
            return tags
        from sklearn.utils import RegressorTags

        tags.estimator_type = "regressor"
        tags.regressor_tags = RegressorTags()

        return tags

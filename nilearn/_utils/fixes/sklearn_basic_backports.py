"""
ClassifierMixin and LinearClassifierMixin have been copied from sklearn
so we don't have to fight back-compat problems every second.

"""
# Author: DOHMATOB Elvis

import numbers
import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils import atleast2d_or_csr, as_float_array
from sklearn.base import RegressorMixin
from sklearn.preprocessing import LabelBinarizer


def center_data(X, y, fit_intercept, normalize=False, copy=True,
                sample_weight=None):
    """
    Centers data to have mean zero along axis 0. This is here because
    nearly all linear models will want their data to be centered.

    If sample_weight is not None, then the weighted mean of X and y
    is zero, and not the mean itself
    """
    X = as_float_array(X, copy)
    no_sample_weight = (sample_weight is None
                        or isinstance(sample_weight, numbers.Number))

    if fit_intercept:
        if sp.issparse(X):
            X_mean = np.zeros(X.shape[1])
            X_std = np.ones(X.shape[1])
        else:
            if no_sample_weight:
                X_mean = X.mean(axis=0)
            else:
                X_mean = (np.sum(X * sample_weight[:, np.newaxis], axis=0)
                          / np.sum(sample_weight))
            X -= X_mean
            if normalize:
                X_std = np.sqrt(np.sum(X ** 2, axis=0))
                X_std[X_std == 0] = 1
                X /= X_std
            else:
                X_std = np.ones(X.shape[1])
        if no_sample_weight:
            y_mean = y.mean(axis=0)
        else:
            if y.ndim <= 1:
                y_mean = (np.sum(y * sample_weight, axis=0)
                          / np.sum(sample_weight))
            else:
                # cater for multi-output problems
                y_mean = (np.sum(y * sample_weight[:, np.newaxis], axis=0)
                          / np.sum(sample_weight))
        y = y - y_mean
    else:
        X_mean = np.zeros(X.shape[1])
        X_std = np.ones(X.shape[1])
        y_mean = 0. if y.ndim == 1 else np.zeros(y.shape[1], dtype=X.dtype)
    return X, y, X_mean, y_mean, X_std


class ClassifierMixin(object):
    """Mixin class for all classifiers in scikit-learn."""

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training set.

        y : array-like, shape = [n_samples]
            Labels for X.

        Returns
        -------
        z : float

        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


class LinearClassifierMixin(ClassifierMixin):
    """Mixin for linear classifiers.

    Handles prediction for sparse and dense X.
    """

    def decision_function(self, X):
        """Predict confidence scores for samples.

        The confidence score for a sample is the signed distance of that
        sample to the hyperplane.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
            Confidence scores per (sample, class) combination. In the binary
            case, confidence score for self.classes_[1] where >0 means this
            class would be predicted.
        """
        X = atleast2d_or_csr(X)

        n_features = self.coef_.shape[1]
        if X.shape[1] != n_features:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (X.shape[1], n_features))

        scores = safe_sparse_dot(X, self.coef_.T,
                                 dense_output=True) + self.intercept_
        return scores.ravel() if scores.shape[1] == 1 else scores

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        C : array, shape = [n_samples]
            Predicted class label per sample.
        """
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(np.int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]

    def _predict_proba_lr(self, X):
        """Probability estimation for OvR logistic regression.

        Positive class probabilities are computed as
        1. / (1. + np.exp(-self.decision_function(X)));
        multiclass is handled by normalizing that over all classes.
        """
        prob = self.decision_function(X)
        prob *= -1
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)
        if len(prob.shape) == 1:
            return np.vstack([1 - prob, prob]).T
        else:
            # OvR normalization, like LibLinear's predict_probability
            prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob


def is_classifier(estimator):
    """Returns True if the given estimator is (probably) a classifier.

    XXX the sklearn version does some weird checks (that fail for our
    estimators).
    """
    return isinstance(estimator, ClassifierMixin)


def is_regressor(estimator):
    """Returns True if the given estimator is (probably) a regressor.

    XXX the sklearn version does some weird checks (that fail for our
    estimators).
    """
    return isinstance(estimator, RegressorMixin)


class MyLabelBinarizer(LabelBinarizer):
    """Binarize labels in a one-vs-all fashion

    Several regression and binary classification algorithms are
    available in the scikit. A simple way to extend these algorithms
    to the multi-class classification case is to use the so-called
    one-vs-all scheme.

    At learning time, this simply consists in learning one regressor
    or binary classifier per class. In doing so, one needs to convert
    multi-class labels to binary labels (belong or does not belong
    to the class). LabelBinarizer makes this process easy with the
    transform method.

    At prediction time, one assigns the class for which the corresponding
    model gave the greatest confidence. LabelBinarizer makes this easy
    with the inverse_transform method.

    Parameters
    ----------

    neg_label : int (default: 0)
        Value with which negative labels must be encoded.

    pos_label : int (default: 1)
        Value with which positive labels must be encoded.

    Attributes
    ----------
    `classes_` : array of shape [n_class]
        Holds the label for each class.

    `multilabel_` : boolean
        True if the transformer was fitted on a multilabel rather than a
        multiclass set of labels.

    Examples
    --------
    >>> from sklearn import preprocessing
    >>> lb = preprocessing.LabelBinarizer()
    >>> lb.fit([1, 2, 6, 4, 2])
    LabelBinarizer(neg_label=0, pos_label=1)
    >>> lb.classes_
    array([1, 2, 4, 6])
    >>> lb.multilabel_
    False
    >>> lb.transform([1, 6])
    array([[1, 0, 0, 0],
           [0, 0, 0, 1]])

    >>> lb.fit_transform([(1, 2), (3,)])
    array([[1, 1, 0],
           [0, 0, 1]])
    >>> lb.classes_
    array([1, 2, 3])
    >>> lb.multilabel_
    True

    See also
    --------
    label_binarize : function to perform the transform operation of
        LabelBinarizer with fixed classes.
    """

    def __init__(self, neg_label=0, pos_label=1):
        if neg_label >= pos_label:
            raise ValueError("neg_label must be strictly less than pos_label.")

        self.neg_label = neg_label
        self.pos_label = pos_label

    def fit_transform(self, y):
        """Transform multi-class labels to binary labels

        The output of transform is sometimes referred to by some authors as the
        1-of-K coding scheme.

        Parameters
        ----------
        y : numpy array of shape [n_samples] or sequence of sequences
            Target values. In the multilabel case the nested sequences can
            have variable lengths.

        Returns
        -------
        Y : numpy array of shape [n_samples, n_classes]
        """

        y_ = LabelBinarizer.fit_transform(self, y)

        if np.min(y_) == 0. and self.neg_label == -1:
            # fix for old sklearn versions (0.10 for example)
            y_ = 2. * (y_ == 1.) - 1.

        return y_

"""
Back-porting some fundamental sklearn API.

"""
# Author: DOHMATOB Elvis

import numbers
import numpy as np
import scipy.sparse as sp
from sklearn.utils import as_float_array
from sklearn.preprocessing import LabelBinarizer as SklLabelBinarizer


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


# Cater for the fact that in 0.10 the LabelBinarizer did not have a
# neg_label
class _LabelBinarizer(SklLabelBinarizer):

    def __init__(self, neg_label=0, pos_label=1):
        # Always call the super class's init method
        SklLabelBinarizer.__init__(self)
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

        y_ = SklLabelBinarizer.fit_transform(self, y)

        if np.min(y_) == 0. and self.neg_label == -1:
            y_ = 2. * (y_ == 1.) - 1.

        return y_


if hasattr(SklLabelBinarizer(), 'neg_label'):
    LabelBinarizer = SklLabelBinarizer
else:
    LabelBinarizer = _LabelBinarizer

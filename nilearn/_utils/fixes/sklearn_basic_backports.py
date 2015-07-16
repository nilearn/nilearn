"""
Back-porting some fundamental sklearn API.

"""
# Author: DOHMATOB Elvis

import numbers
import numpy as np
import scipy.sparse as sp
from sklearn.utils import as_float_array
from sklearn.preprocessing import LabelBinarizer as SklLabelBinarizer


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

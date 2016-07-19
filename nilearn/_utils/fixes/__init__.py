from distutils.version import LooseVersion
import sklearn


if (LooseVersion(sklearn.__version__) < LooseVersion('0.15') or
      sklearn.__version__ == '0.15-git'):
    from .sklearn_f_regression_nosparse import (
        f_regression_nosparse as f_regression)
else:
    from sklearn.feature_selection import f_regression

try:
    # scikit-learn >= 0.16
    from sklearn.cross_validation import check_cv
except ImportError:
    # scikit-learn < 0.16
    from sklearn.cross_validation import _check_cv as check_cv

# atleast2d_or_csr
try:
    from sklearn.utils import atleast2d_or_csr
except ImportError:
    # Changed in 0.15
    from sklearn.utils import check_array as atleast2d_or_csr

# roc_auc_score
try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    from sklearn.metrics import auc as roc_auc_score

try:
    from sklearn.utils import check_X_y
    from sklearn.utils import check_is_fitted
except ImportError:
    # scikit-learn < 0.16
    from .sklearn_validation import check_X_y
    from .sklearn_validation import check_is_fitted

__all__ = ['f_regression', 'atleast2d_or_csr', 'roc_auc_score',
           'check_X_y', 'check_is_fitted', 'check_cv']

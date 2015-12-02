from distutils.version import LooseVersion
import sklearn

# f_regression

if (LooseVersion(sklearn.__version__) < LooseVersion('0.15') or
          sklearn.__version__ == '0.15-git'):
    from .sklearn_f_regression_nosparse import (
        f_regression_nosparse as f_regression)
else:
    from sklearn.feature_selection import f_regression

try:
    # scikit-learn < 0.16
    from sklearn.utils import check_arrays as check_array
except:
    # scikit-learn >= 0.16
    from sklearn.utils import check_array

try:
    # scikit-learn < 0.16
    from sklearn.cross_validation import _check_cv as check_cv
except:
    # scikit-learn >= 0.16
    from sklearn.cross_validation import check_cv

# atleast2d_or_csr
# XXX change this to check_array too
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

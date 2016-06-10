from distutils.version import LooseVersion
import sklearn


if (LooseVersion(sklearn.__version__) < LooseVersion('0.15') or
          sklearn.__version__ == '0.15-git'):
    from .sklearn_f_regression_nosparse import (
        f_regression_nosparse as f_regression)
else:
    from sklearn.feature_selection import f_regression


try:
    # scikit-learn < 0.16
    from sklearn.cross_validation import _check_cv as check_cv
except:
    # scikit-learn >= 0.16
    from sklearn.cross_validation import check_cv


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
    from sklearn.utils.multiclass import type_of_target
except ImportError:
    # scikit-learn < 0.14
    from .scikit_learn_utils import type_of_target


try:
    from sklearn.metrics.scorer import check_scoring
    from sklearn.metrics.scorer import make_scorer
except ImportError:
    # scikit-learn < 0.14
    from .scikit_learn_scorer import check_scoring
    from .scikit_learn_scorer import make_scorer


try:
    from sklearn.grid_search import ParameterGrid
except ImportError:
    # scikit-learn < 0.14
    from .scikit_learn_gridsearch import ParameterGrid


__all__ = ['f_regression', 'atleast2d_or_csr', 'roc_auc_score',
           'check_X_y', 'check_is_fitted', 'check_scoring',
           'check_cv', 'make_scorer', 'type_of_target']

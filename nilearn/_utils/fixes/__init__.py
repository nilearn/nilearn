# XXX use LooseVersion to specify correct breakpoint of ALL backported APIs!

from distutils.version import LooseVersion
import sklearn

if (LooseVersion(sklearn.__version__) < LooseVersion('0.15') or
      sklearn.__version__ == '0.15-git'):
    from .sklearn_f_regression_nosparse import (
        f_regression_nosparse as f_regression)
else:
    from sklearn.feature_selection import f_regression

# LinearClassifierMixin backport
try:
    from sklearn.linear_models.base import LinearClassifierMixin
except ImportError:
    from .sklearn_basic_backports import LinearClassifierMixin

# center_data backport
try:
    from sklearn.linear_models.base import center_data
except ImportError:
    from .sklearn_basic_backports import center_data

# is_[classifier|regressor] backports
#  XXX the sklearn version does some weird checks (that fail for our
from .sklearn_basic_backports import is_classifier, is_regressor

# roc_auc_score
try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    from sklearn.metrics import auc as roc_auc_score

# LabelBinarizer backport
from .sklearn_basic_backports import MyLabelBinarizer as LabelBinarizer

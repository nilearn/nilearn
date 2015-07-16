# XXX use LooseVersion to specify correct breakpoint of ALL backported APIs!

from distutils.version import LooseVersion
import sklearn

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

# LabelBinarizer backport
from .sklearn_basic_backports import LabelBinarizer

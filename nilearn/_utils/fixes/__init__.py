from distutils.version import LooseVersion
import sklearn

from sklearn.model_selection import check_cv
from sklearn.model_selection import cross_val_score


from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted

from sklearn.linear_model.base import _preprocess_data as center_data

__all__ = ['check_X_y', 'check_is_fitted', 'check_cv', 'cross_val_score',
           'center_data']

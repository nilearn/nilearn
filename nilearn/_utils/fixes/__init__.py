
try:
    # 0.18 > scikit-learn >= 0.16
    from sklearn.model_selection import check_cv
except ImportError:
    # scikit-learn < 0.16
    from .sklearn_model_selection import check_cv


try:
    from sklearn.utils import check_X_y
    from sklearn.utils import check_is_fitted
except ImportError:
    # scikit-learn < 0.16
    from .sklearn_validation import check_X_y
    from .sklearn_validation import check_is_fitted

__all__ = ['check_X_y', 'check_is_fitted', 'check_cv']

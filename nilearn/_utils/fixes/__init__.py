from distutils.version import LooseVersion
import sklearn

if LooseVersion(sklearn.__version__) < LooseVersion('0.12'):
    from .sklearn_f_regression_nosparse import (
        f_regression_nosparse as f_regression)
elif (LooseVersion(sklearn.__version__) < LooseVersion('0.15') or
      sklearn.__version__ == '0.15-git'):
    from .sklearn_f_regression import f_regression
else:
    from sklearn.feature_selection import f_regression

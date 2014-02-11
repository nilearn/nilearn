from distutils.version import LooseVersion
import sklearn

if LooseVersion(sklearn.__version__) < '0.15':
    from .sklearn_f_regression import f_regression
else:
    from sklearn.feature_selection import f_regression

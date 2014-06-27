from distutils.version import LooseVersion
import sklearn

if (LooseVersion(sklearn.__version__) < LooseVersion('0.14')):
    from .sklearn_center_data import center_data
else:
    from sklearn.feature_selection import f_regression
    from sklearn.linear_model.base import center_data

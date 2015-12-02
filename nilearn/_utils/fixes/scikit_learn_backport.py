

try:
    import sklearn.metrics.scorer as scorer
except ImportError:
    # metrics doesn't exist before scikit-learn 0.14.x
    from . import sklearn_learn_scorer as scorer


try:
    from sklearn.utils import check_X_y
except ImportError:
    # scikit-learn < 0.16
    from . import sklearn as check_X_y


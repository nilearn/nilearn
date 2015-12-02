try:
    from sklearn.metrics.scorer import check_scoring
    from sklearn.metrics.scorer import make_scorer
except ImportError:
    # scikit-learn < 0.14
    from .scikit_learn_scorer import check_scoring
    from .scikit_learn_scorer import make_scorer


try:
    from sklearn.utils import check_X_y
except ImportError:
    # scikit-learn < 0.16
    from .scikit_learn_validation import check_X_y


try:
    from sklearn.grid_search import ParameterGrid
except ImportError:
    # scikit-learn < 0.14
    from .scikit_learn_gridsearch import ParameterGrid

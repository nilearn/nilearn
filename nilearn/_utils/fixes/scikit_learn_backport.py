# try:
#     from sklearn.utils.multiclass import type_of_target
# except ImportError:
#     # scikit-learn < 0.14
# # try:
#     from .scikit_learn_utils import type_of_target
# # except:
# #     print "nooo"

# try:
#     from sklearn.metrics.scorer import check_scoring
#     from sklearn.metrics.scorer import make_scorer
# except ImportError:
    # scikit-learn < 0.14
try:
    from .scikit_learn_scorer import check_scoring
    from .scikit_learn_scorer import make_scorer
except:
    print "no"


try:
    from sklearn.utils import check_X_y
    from sklearn.utils import check_is_fitted
except ImportError:
    # scikit-learn < 0.16
    from .scikit_learn_validation import check_X_y
    from .scikit_learn_validation import check_is_fitted


try:
    from sklearn.grid_search import ParameterGrid
except ImportError:
    # scikit-learn < 0.14
    from .scikit_learn_gridsearch import ParameterGrid




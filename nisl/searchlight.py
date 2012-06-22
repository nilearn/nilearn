"""
Searchlight:
The searchlight [Kriegeskorte 06] is a widely used approach for the study
of the fine-grained patterns of information in fMRI analysis.
Its principle is relatively simple: a small group of neighboring features
is extracted from the data, and the prediction function is instantiated
on these features only. The resulting prediction accuracy is thus associated
with all the features within the group, or only with the feature on the center.
This yields a map of local fine-grained information, that can be used for
assessing hypothesis on the local spatial layout of the neural
code under investigation.

Nikolaus Kriegeskorte, Rainer Goebel & Peter Bandettini.
Information-based functional brain mapping.
Proceedings of the National Academy of Sciences
of the United States of America,
vol. 103, no. 10, pages 3863-3868, March 2006

Authors : Vincent Michel (vm.michel@gmail.com)
          Alexandre Gramfort (alexandre.gramfort@inria.fr)
License: BSD 3 clause
"""
import numpy as np

from joblib.parallel import Parallel, delayed

from sklearn.cross_validation import cross_val_score
from sklearn.base import BaseEstimator

def search_light(X, y, estimator, A, score_func=None, cv=None, n_jobs=-1,
                 verbose=True):
    """
    Function for computing a search_light

    Parameters
    ----------
    X: array-like of shape at least 2D
        The data to fit.

    y: array-like
    The target variable to try to predict.

    estimator: estimator object implementing 'fit'
    The object to use to fit the data

    A : sparse matrix.
    adjacency matrix. Defines for each sample the neigbhoring samples
    following a given structure of the data.

    score_func: callable, optional
    callable taking as arguments the fitted estimator, the
    test data (X_test) and the test target (y_test) if y is
    not None.

    cv: cross-validation generator, optional
    A cross-validation generator. If None, a 3-fold cross
    validation is used or 3-fold stratified cross-validation
    when y is supplied.

    n_jobs: integer, optional
    The number of CPUs to use to do the computation. -1 means
    'all CPUs'.

    verbose: boolean, optional
    The verbosity level. Defaut is False

    Return
    ------
    scores: array-like of shape (number of rows in A)
            search_light scores
    """
    scores = np.zeros(len(A.rows), dtype=float)
    group_iter = GroupIterator(X.shape[1], n_jobs)
    scores = Parallel(n_jobs=n_jobs)\
             (delayed(_group_iter_search_light)(list_i, A.rows[list_i],
              estimator, X, y, score_func, cv, verbose)
              for list_i in group_iter)
    return np.concatenate(scores)


class GroupIterator(object):
    """Group iterator

    Provides group of features for search_light loop
    that may be used with Parallel.
    """

    def __init__(self, n_features, n_jobs=1):
        """Group iterator

        Provides group of features for search_light loop

        Parameters
        ===========
        n_features: int
                    Total number of features
        n_jobs: integer, optional
                The number of CPUs to use to do the computation. -1 means
                'all CPUs'. Defaut is 1
        """
        self.n_features = n_features
        if n_jobs == -1:
            import multiprocessing
            n_jobs = multiprocessing.cpu_count()
        self.n_jobs = n_jobs

    def __iter__(self):
        frac = np.int(self.n_features / self.n_jobs)
        for i in range(self.n_jobs):
            if i != (self.n_jobs - 1):
                list_i = range(i * frac, (i + 1) * frac)
            else:
                list_i = range(i * frac, self.n_features)
            yield list_i


def _group_iter_search_light(list_i, list_rows, estimator, X, y, score_func,
                            cv, verbose):
    """
    Function for grouped iterations of search_light
    """
    par_scores = np.zeros(len(list_rows))
    for i, row in enumerate(list_rows):
        if list_i[i] not in row:
            row.append(list_i[i])
        if cv:
            par_scores[i] = np.mean(cross_val_score(estimator, X[:, row],
                    y, score_func, cv, n_jobs=1))
        else:
            par_scores[i] = estimator.fit(X[:, row], y).score(X[:, row], y)
        if verbose:
            print "%d / %d : %2.2f" % (list_i[i], X.shape[1], par_scores[i])
    return par_scores


##############################################################################
### Class for search_light ###################################################
##############################################################################


class SearchLight(BaseEstimator):
    """
    SearchLight class.
    Class to perform a search_light using an arbitrary type of classifier.
    """

    def __init__(self, A, estimator, n_jobs=-1):
        """
        Parameters
        ----------
        A : sparse matrix.
        adjacency matrix. Defines for each sample the neigbhoring samples
        following a given structure of the data.

        estimator: estimator object implementing 'fit'
        The object to use to fit the data

        n_jobs: integer, optional. Default is -1.
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.
        """
        self.A = A.tolil()
        self.estimator = estimator
        self.n_jobs = n_jobs

    def fit(self, X, y, score_func=None, cv=None, verbose=False):
        """
        Fit the search_light

        X: array-like of shape at least 2D
        The data to fit.

        y: array-like
        The target variable to try to predict.

        score_func: callable, optional
        callable taking as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.

        cv: cross-validation generator, optional
        A cross-validation generator. If None, a 3-fold cross
        validation is used or 3-fold stratified cross-validation
        when y is supplied.

        verbose: boolean, optional
        The verbosity level. Defaut is False

        Attributs
        ---------
        scores: array-like of shape (number of rows in A)
            search_light scores
        """
        self.score_func = score_func
        self.cv = cv
        self.verbose = verbose
        self.scores = search_light(X, y, self.estimator, self.A,
                      self.score_func, self.cv, self.n_jobs, self.verbose)
        return self

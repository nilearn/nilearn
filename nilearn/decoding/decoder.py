"""High-level decoding object that exposes standard classification

and regression strategies such as SVM, LogisticRegression and Ridge,
with optional feature selection, and integrated parameter selection.
"""
# Authors : Yannick Schwartz
#           Andres Hoyos-Idrobo
#
# License: simplified BSD

from distutils.version import LooseVersion
import sklearn
import itertools
import warnings
import numpy as np
from sklearn.externals.joblib import Parallel, delayed, Memory
from sklearn.linear_model.ridge import Ridge, RidgeClassifier, _BaseRidge
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVR
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm.bounds import l1_min_c
from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearModel
from sklearn.utils.extmath import safe_sparse_dot
from sklearn import clone

try:
    from sklearn.utils import atleast2d_or_csr
except ImportError: # sklearn 0.15
    from sklearn.utils import check_array as atleast2d_or_csr

try:
    from sklearn.grid_search import ParameterGrid
except ImportError: # sklearn 0.18
    from sklearn.model_selection import ParameterGrid

from ..input_data.masker_validation import check_embedded_nifti_masker
from .._utils.param_validation import _adjust_screening_percentile
from .._utils.fixes import check_scoring
from .._utils.fixes import check_X_y
from .._utils.fixes import check_is_fitted
from .._utils.compat import _basestring
from .._utils.fixes import check_cv
from .._utils.param_validation import check_feature_screening
from .._utils import CacheMixin


SUPPORTED_ESTIMATORS = dict(
    svc_l1=LinearSVC(penalty='l1', dual=False),
    svc_l2=LinearSVC(penalty='l2'),
    svc=LinearSVC(penalty='l2'),
    logistic_l1=LogisticRegression(penalty='l1'),
    logistic_l2=LogisticRegression(penalty='l2'),
    logistic=LogisticRegression(penalty='l2'),
    ridge_classifier=RidgeClassifier(),
    ridge_regression=Ridge(),
    ridge=Ridge(),
    svr=SVR(kernel='linear'),)


def _check_param_grid(estimator, X, y, param_grid):
    """Check param_grid and return sensible default if none is given.
    """
    if param_grid is None:
        param_grid = {}
        # define loss function
        if isinstance(estimator, LogisticRegression):
            loss = 'log'
        elif isinstance(estimator, (LinearSVC, _BaseRidge, SVR)):
            loss = 'l2'

        if hasattr(estimator, 'penalty') and (estimator.penalty == 'l1'):
            min_c = l1_min_c(X, y, loss=loss)
        else:
            min_c = 0.5
        param_grid['C'] = np.array([2, 20, 200]) * min_c

        if isinstance(estimator, _BaseRidge):
            param_grid['alpha'] = 1. / (param_grid.pop('C') * 2)
    return param_grid


def _parallel_fit(estimator, X, y, train, test, param_grid, is_classif, scorer,
                  mask_img, class_index, screening_percentile=None):
    """Find the best estimator for a fold within a job."""

    # unprocessed test labels indices, they are used to measure the overal
    # perfromance.
    y_true_indices = test
    n_features = X.shape[1]

    selector = check_feature_screening(screening_percentile, mask_img,
                                       is_classif)

    do_screening = (n_features > 100) and (screening_percentile < 100.)

    X_train = X[train].copy()
    y_train = y[train].copy()
    X_test = X[test].copy()
    y_test = y[test].copy()

    if (selector is not None) and do_screening:
        X_train = selector.fit_transform(X[train], y[train])
        X_test = selector.transform(X[test])

    param_grid = _check_param_grid(estimator, X_train, y_train, param_grid)
    test_scores = []
    best_score = None
    for param in ParameterGrid(param_grid):
        estimator = clone(estimator).set_params(**param)
        estimator.fit(X_train, y_train)

        if is_classif:
            if hasattr(estimator, 'predict_proba'):
                y_prob = estimator.predict_proba(X_test)
                y_prob = y_prob[:, 1]
                inverse_prob = 1 - y_prob
            else:
                decision = estimator.decision_function(X_test)
                if decision.ndim == 2:
                    y_prob = decision[:, 1]
                    inverse_prob = np.abs(decision[:, 0])
                else:
                    y_prob = decision
                    inverse_prob = -decision
            score = scorer(estimator, X_test, y_test)
            if np.all(estimator.coef_ == 0):
                score = 0
        else:  # regression
            y_prob = estimator.predict(X_test)
            score = scorer(estimator, X_test, y_test)

        test_scores.append(score)
        if (best_score is None) or (score >= best_score):
            best_score = score
            best_coef = estimator.coef_
            best_intercept = estimator.intercept_
            best_y = {}
            best_y['y_prob'] = y_prob
            best_y['y_true_indices'] = y_true_indices
            best_param = param
            if is_classif:
                best_y['inverse'] = inverse_prob

    if selector is not None:
        best_coef = selector.inverse_transform(best_coef)
    # scikit-learn > 1.7
    if isinstance(estimator, SVR):
        best_coef = -best_coef

    return (class_index, best_coef, best_intercept, best_y, best_param,
            test_scores)


class BaseDecoder(LinearModel, RegressorMixin, CacheMixin):
    """A wrapper for popular classification/regression strategies for
    neuroimgaging.

    The `Decoder` object supports classification and regression methods.
    It implements a model selection scheme that averages the best models
    within a cross validation loop. The resulting average model is the
    one used as a classifier or a regressor. The `Decoder` object also
    leverages the `NiftiMaskers` to provide a direct interface with the
    nifti files on disk.

    Parameters
    -----------
    estimator : str, optional
        The estimator to choose among: 'svc', 'svc_l1', 'logistic',
        'logistic_l1', 'ridge_classifier', 'ridge_regression',
        and 'svr'. Defaults to 'svc'.

    mask: filename, NiImage, NiftiMasker, or MultiNiftiMasker, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a masker with default
        parameters.

    cv : cross-validation generator, optional
        A cross-validation generator. If None, a 10-fold cross
        validation is used for regression or 10-fold stratified
        cross-validation for classification.

    param_grid : dict of string to sequence, or sequence of such
        The parameter grid to explore, as a dictionary mapping estimator
        parameters to sequences of allowed values.

        An empty dict signifies default parameters.

        A sequence of dicts signifies a sequence of grids to search, and is
        useful to avoid exploring parameter combinations that make no sense
        or have no effect. See scikit-learn documentation for more information.

    screening_percentile: int, float, optional, in the closed interval [0, 100]
        Perform an univariate feature selection based on the Anova F-value for
        the input data. A float according to a percentile of the highest
        scores. Default: 20.

    scoring : string, callable or None, optional. Default: None
        The scoring strategy to use. See the scikit-learn documentation
        If callable, takes as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.

        For regression: 'r2', 'mean_absolute_error', or 'mean_squared_error'.
        For classification: 'accuracy', 'f1', 'precision', or 'recall'.

    smoothing_fwhm: float, optional. Default: None
        If smoothing_fwhm is not None, it gives the size in millimeters of the
        spatial smoothing to apply to the signal.

    standardize : boolean, optional. Default: None
        If standardize is True, the time-series are centered and normed:
        their variance is put to 1 in the time dimension.

    target_affine: 3x3 or 4x4 matrix, optional. Default: None
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    target_shape: 3-tuple of integers, optional. Default: None
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    low_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    t_r : float, optional. Default: None
        This parameter is passed to signal.clean. Please see the related
        documentation for details.

    mask_strategy: {'background' or 'epi'}, optional. Default: 'background'
        The strategy used to compute the mask: use 'background' if your
        images present a clear homogeneous background, and 'epi' if they
        are raw EPI images. Depending on this value, the mask will be
        computed from masking.compute_background_mask or
        masking.compute_epi_mask. Default is 'background'.

    memory: instance of joblib.Memory or string
        Used to cache the masking process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level: integer, optional. Default: 0
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    n_jobs : int, optional. Default: 1.
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    verbose : int, optional. Default: False.
        Verbosity level.
    """
    def __init__(self, estimator='svc', mask=None, cv=None, param_grid=None,
                 screening_percentile=20, scoring=None, smoothing_fwhm=None,
                 standardize=True, target_affine=None, target_shape=None,
                 low_pass=None, high_pass=None, t_r=None,
                 mask_strategy='background', is_classif=True, memory=None,
                 memory_level=0, n_jobs=1, verbose=False):
        self.estimator = estimator
        self.mask = mask
        self.cv = cv
        self.param_grid = param_grid
        self.screening_percentile = screening_percentile
        self.scoring = scoring
        self.is_classif = is_classif
        self.smoothing_fwhm = smoothing_fwhm
        self.standardize = standardize
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.mask_strategy = mask_strategy
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.memory = memory
        self.memory_level = memory_level
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y):
        """Fit the decoder (learner).

        Parameters
        ----------
        X : list of Niimg-like objects
            See http://nilearn.github.io/manipulating_images/input_output.html
            Data on which model is to be fitted. If this is a list,
            the affine is considered the same for all.

        y : array or list of length n_samples
            The dependent variable (age, sex, QI, etc.).
            Target variable to predict. Must have exactly as many elements as
            3D images in niimg.

        Attributes
        ----------
        `masker_` : instance of NiftiMasker
            The nifti masker used to mask the data.

        `mask_img_`: NiImage
            Mask computed by the masker object.

        `classes_`: numpy.ndarray
            Classes to predict. For classification only.

        `screening_percentile_` : float
            Screening percentile corrected according to volume of mask,
            relative to the volume of standard brain.

        `coef_`: numpy.ndarray, shape=(n_classes, n_features)
            Contains the mean of the models weight vector across
            fold for each class.

        `intercept_` : narray, shape (nclasses -1,)
            Intercept (a.k.a. bias) added to the decision function.
            It is available only when parameter intercept is set to True.

        `cv_` : list of pairs of lists
            List of the (n_folds,) folds. For the corresponding fold,
            each pair is composed of two lists of indices,
            one for the train samples and one for the test samples.

        `std_coef_`: numpy.ndarray, shape=(n_classes, n_features)
            Contains the standard deviation of the models weight vector across
            fold for each class.

        `coef_img_`: dict of NiImage
            Dictionary containing `coef_` with class names as keys,
            and `coef_` transformed in NiImages as values. In the case
            of a regression, it contains a single NiImage at the key 'beta'.

        `std_coef_img_`: dict of NiImage
            Dictionary containing `std_coef_` with class names as keys,
            and `coef_` transformed in NiImages as values. In the case
            of a regression, it contains a single NiImage at the key 'beta'.

        `cv_y_true_` : numpy.ndarray, shape=(n_samples * n_folds, n_classes)
            Ground truth labels for left out samples in inner cross-validation.

        `cv_y_pred_` : numpy.ndarray, shape=(n_samples * n_folds, n_classes)
            Predicted labels for left out samples in inner cross-validation.

        `cv_params_`: dict of lists
            Best point in the parameter grid for each tested fold
            in the inner cross validation loop.

        `cv_indices_` : numpy.ndarray, shape=(n_samples * n_folds)
            Indices of the inner cross-validation folds.

        `cv_scores_` : ndarray, shape (n_parameters, n_folds)
            Scores (misclassification) for each parameter, and on each fold
        """
        # Setup memory, parallel and masker
        if isinstance(self.memory, _basestring) or (self.memory is None):
            self.memory_ = Memory(cachedir=self.memory, verbose=self.verbose)
        else:
            self.memory_ = self.memory

        # nifti masking
        self.masker_ = check_embedded_nifti_masker(self, multi_subject=False)
        X = self.masker_.fit_transform(X)
        self.mask_img_ = self.masker_.mask_img_

        X, y = check_X_y(X, y, ['csr', 'csc', 'coo'], dtype=np.float,
                         multi_output=True, y_numeric=True)

        # Setup model
        if not isinstance(self.estimator, _basestring):
            warnings.warn('Use a custom estimator at your own risk.')

        elif self.estimator in list(SUPPORTED_ESTIMATORS.keys()):
            estimator = SUPPORTED_ESTIMATORS.get(self.estimator,
                                                 self.estimator)

        scorer = check_scoring(estimator, self.scoring)

        # Setup cv
        if LooseVersion(sklearn.__version__) >= LooseVersion('0.18'):
            # scikit-learn >= 0.18
            self.cv_ = list(check_cv(
                self.cv, y=y, classifier=self.is_classif).split(X, y))
        else:
            # scikit-learn < 0.18
            self.cv_ = list(check_cv(self.cv, X=X, y=y,
                                     classifier=self.is_classif))

        # number of problems to solve
        if self.is_classif:
            y = self._binarize_y(y)
        else:
            y = y[:, np.newaxis]
        if self.is_classif and self.n_classes_ > 2:
            n_problems = self.n_classes_
        else:
            n_problems = 1

        self.screening_percentile_ = _adjust_screening_percentile(
            self.screening_percentile, self.mask_img_, verbose=self.verbose)

        coefs = {}
        intercepts = {}
        cv_y_prob = {}
        cv_y_true = {}
        cv_indices = {}
        cv_scores = {}
        self.cv_params_ = {}

        parallel = Parallel(n_jobs=self.n_jobs, verbose=2 * self.verbose)

        for i, (c, coef, intercept, y_info, params, scores) in enumerate(
            parallel(
                delayed(self._cache(_parallel_fit))(
                    estimator, X, y[:, class_index], train, test,
                    self.param_grid, self.is_classif, scorer, self.mask_img_,
                    class_index, self.screening_percentile_)
                for class_index, (train, test) in itertools.product(
                    range(n_problems), self.cv_))):

            classes = self.classes_

            # Models to aggregate
            coefs.setdefault(classes[c], []).append(coef)
            intercepts.setdefault(classes[c], []).append(intercept)

            cv_y_prob.setdefault(classes[c], []).append(y_info['y_prob'])
            cv_indices.setdefault(classes[c], []).extend(
                [i] * len(y_info['y_prob']))
            cv_scores.setdefault(classes[c], []).append(scores)

            self.cv_params_.setdefault(classes[c], {})
            for k in params:
                self.cv_params_[classes[c]].setdefault(k, []).append(params[k])

            if self.is_classif:
                cv_y_true.setdefault(classes[c], []).extend(
                    self._enc.inverse_transform(y[y_info['y_true_indices']]))
            else:
                cv_y_true.setdefault(classes[c], []).extend(
                    y[y_info['y_true_indices']])

            if (n_problems <= 2) and self.is_classif:
                # Binary classification
                other_class = np.setdiff1d(classes, classes[c])[0]
                coefs.setdefault(other_class, []).append(-coef)
                intercepts.setdefault(other_class, []).append(-intercept)
                cv_y_prob.setdefault(other_class, []).append(y_info['inverse'])
                # misc
                cv_scores.setdefault(other_class, []).append(scores)
                cv_y_true.setdefault(other_class, []).extend(
                    self._enc.inverse_transform(y[y_info['y_true_indices']]))

        # Saving the mean score
        self.cv_scores_ = np.mean(
            [np.vstack(cv_scores[c]).T for c in classes], axis=0)

        self.cv_y_true_ = np.array(cv_y_true[cv_y_true.keys()[0]])
        self.cv_indices_ = np.array(cv_indices[cv_indices.keys()[0]])
        self.cv_y_prob_ = np.vstack(
            [np.hstack(cv_y_prob[c]) for c in classes]).T

        if self.is_classif:
            if self.n_classes_ == 2:
                self.cv_y_prob_ = self.cv_y_prob_[0, :]
                indices = (self.cv_y_prob_ > 0).astype(np.int)
            else:
                indices = np.argmax(self.cv_y_prob_, axis=1)
            self.cv_y_pred_ = classes[indices]
        else:
            self.cv_y_pred_ = self.cv_y_prob_

        # Build the final model (the aggregated one).
        self.coef_ = np.vstack([np.mean(coefs[c], axis=0) for c in classes])
        std_coef = np.vstack([np.std(coefs[c], axis=0) for c in classes])
        self.intercept_ = np.hstack([np.mean(intercepts[c], axis=0)
                                     for c in classes])
        coef_img = {}
        std_coef_img = {}
        for c, coef, std in zip(classes, self.coef_, std_coef):
            coef_img[c] = self.masker_.inverse_transform(coef)
            std_coef_img[c] = self.masker_.inverse_transform(std)

        if self.is_classif and (self.n_classes_ == 2):
            self.coef_ = self.coef_[0, :][np.newaxis, :]
            self.intercept_ = self.intercept_[0]

        self.coef_img_ = coef_img
        self.std_coef_img_ = std_coef_img

    def decision_function(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : list of Niimg-like objects
            See http://nilearn.github.io/manipulating_images/input_output.html
            Data on prediction is to be made. If this is a list,
            the affine is considered the same for all.

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            Predicted class label per sample.
        """
        X = self.masker_.transform(X)
        X = atleast2d_or_csr(X)

        n_features = self.coef_.shape[1]
        if X.shape[1] != n_features:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (X.shape[1], n_features))

        scores = safe_sparse_dot(X, self.coef_.T,
                                 dense_output=True) + self.intercept_

        return scores.ravel() if scores.shape[1] == 1 else scores

    def predict(self, X):
        """Predict a label for all X vectors indexed by the first axis.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Retruns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
            Confidence scores per (sample, class) combination. In the binary
            case, confidence score for self.classes_[1] where >0 means this
            class would be predicted.
        """

        check_is_fitted(self, "coef_")
        check_is_fitted(self, "masker_")

        scores = self.decision_function(X)

        if self.is_classif:
            if len(scores.shape) == 1:
                indices = (scores > 0).astype(np.int)
            else:
                indices = scores.argmax(axis=1)
            return self.classes_[indices]

        return scores


class Decoder(BaseDecoder):

    def __init__(self, estimator='svc', mask=None, cv=None, param_grid=None,
                 screening_percentile=20, scoring='roc_auc',
                 smoothing_fwhm=None, standardize=True, target_affine=None,
                 target_shape=None, mask_strategy='background',
                 low_pass=None, high_pass=None, t_r=None, memory=None,
                 memory_level=0, n_jobs=1, verbose=False):
        super(Decoder, self).__init__(
            estimator=estimator, mask=mask, cv=cv, param_grid=param_grid,
            screening_percentile=screening_percentile, scoring=scoring,
            smoothing_fwhm=smoothing_fwhm, standardize=standardize,
            target_affine=target_affine, target_shape=target_shape,
            mask_strategy=mask_strategy, memory=memory, is_classif=True,
            memory_level=memory_level, verbose=verbose, n_jobs=n_jobs)

    def _binarize_y(self, y):
        """Helper function invoked just before fitting a classifier."""
        y = np.array(y)

        # encode target classes as -1 and 1
        self._enc = LabelBinarizer(pos_label=1, neg_label=-1)
        y = self._enc.fit_transform(y)
        self.classes_ = self._enc.classes_
        self.n_classes_ = len(self.classes_)
        return y


class DecoderRegressor(BaseDecoder):

    def __init__(self, estimator='svc', mask=None, cv=None, param_grid=None,
                 screening_percentile=20, scoring='r2',
                 smoothing_fwhm=None, standardize=True, target_affine=None,
                 target_shape=None, mask_strategy='background',
                 low_pass=None, high_pass=None, t_r=None, memory=None,
                 memory_level=0, n_jobs=1, verbose=False):
        self.classes_ = ['beta']

        super(DecoderRegressor, self).__init__(
            estimator=estimator, mask=mask, cv=cv, param_grid=param_grid,
            screening_percentile=screening_percentile, scoring=scoring,
            smoothing_fwhm=smoothing_fwhm, standardize=standardize,
            target_affine=target_affine, target_shape=target_shape,
            low_pass=low_pass, high_pass=high_pass, t_r=t_r,
            mask_strategy=mask_strategy, memory=memory, is_classif=False,
            memory_level=memory_level, verbose=verbose, n_jobs=n_jobs)

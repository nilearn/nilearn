"""High-level decoding object.

Exposes standard classification and
regression strategies such as SVM, LogisticRegression and Ridge,
with optional feature selection,
integrated hyper-parameter selection and aggregation
strategy in which the best models within a cross validation loop are averaged.

Also exposes a high-level method FREM that uses clustering and model
ensembling to achieve state of the art performance
"""

# Authors: Yannick Schwartz
#          Andres Hoyos-Idrobo
#          Binh Nguyen <tuan-binh.nguyen@inria.fr>
#          Thomas Bazeille
#

import itertools
import warnings
from collections.abc import Iterable

import numpy as np
from joblib import Parallel, delayed
from sklearn import clone
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MultiOutputMixin,
    RegressorMixin,
)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import (
    LassoCV,
    LogisticRegressionCV,
    RidgeClassifierCV,
    RidgeCV,
)
from sklearn.metrics import check_scoring, get_scorer
from sklearn.model_selection import (
    LeaveOneGroupOut,
    ParameterGrid,
    ShuffleSplit,
    StratifiedShuffleSplit,
    check_cv,
)
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVR, LinearSVC, l1_min_c
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted, check_X_y

from nilearn._utils import CacheMixin, fill_doc
from nilearn._utils.cache_mixin import _check_memory
from nilearn._utils.masker_validation import (
    check_compatibility_mask_and_images,
    check_embedded_masker,
)
from nilearn._utils.param_validation import check_feature_screening
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn.maskers import SurfaceMasker
from nilearn.regions.rena_clustering import ReNA
from nilearn.surface import SurfaceImage

SUPPORTED_ESTIMATORS = {
    "svc_l1": LinearSVC(penalty="l1", dual=False, max_iter=10000),
    "svc_l2": LinearSVC(penalty="l2", dual=True, max_iter=10000),
    "svc": LinearSVC(penalty="l2", dual=True, max_iter=10000),
    "logistic_l1": LogisticRegressionCV(penalty="l1", solver="liblinear"),
    "logistic_l2": LogisticRegressionCV(penalty="l2", solver="liblinear"),
    "logistic": LogisticRegressionCV(penalty="l2", solver="liblinear"),
    "ridge_classifier": RidgeClassifierCV(),
    "ridge_regressor": RidgeCV(),
    "ridge": RidgeCV(),
    "lasso": LassoCV(),
    "lasso_regressor": LassoCV(),
    "svr": SVR(kernel="linear", max_iter=10000),
    "dummy_classifier": DummyClassifier(strategy="stratified", random_state=0),
    "dummy_regressor": DummyRegressor(strategy="mean"),
}


@fill_doc
def _check_param_grid(estimator, X, y, param_grid=None):
    """Check param_grid and return sensible default if param_grid is None.

    Parameters
    ----------
    estimator : str
        The estimator to choose among:
        %(classifier_options)s
        %(regressor_options)s

    X : list of Niimg-like objects
        See :ref:`extracting_data`.
        Data on which model is to be fitted. If this is a list,
        the affine is considered the same for all.

    y : array or list of shape (n_samples)
        The dependent variable (age, sex, IQ, yes/no, etc.).
        Target variable to predict. Must have exactly as many elements as
        3D images in niimg.

    param_grid : dict of str to sequence, or sequence of such. Default None
        The parameter grid to explore, as a dictionary mapping estimator
        parameters to sequences of allowed values.

        An empty dict signifies default parameters.

        A sequence of dicts signifies a sequence of grids to search, and is
        useful to avoid exploring parameter combinations that make no sense
        or have no effect. See scikit-learn documentation for more information.

        For Dummy estimators, parameter grid defaults to empty as these
        estimators do not have hyperparameters to grid search.

    Returns
    -------
    param_grid : dict of str to sequence, or sequence of such. Sensible default
    dict has size 1 for linear models.

    """
    if param_grid is None:
        param_grid = _default_param_grid(estimator, X, y)

    elif isinstance(estimator, (RidgeCV, RidgeClassifierCV)):
        param_grid = _wrap_param_grid(param_grid, "alphas")
    elif isinstance(estimator, LogisticRegressionCV):
        param_grid = _replace_param_grid_key(param_grid, "C", "Cs")
        param_grid = _wrap_param_grid(param_grid, "Cs")
    elif isinstance(estimator, LassoCV):
        param_grid = _wrap_param_grid(param_grid, "alphas")

    return param_grid


def _default_param_grid(estimator, X, y):
    """Generate sensible default for param_grid.

    Parameters
    ----------
    estimator : str
        The estimator to choose among:
        %(classifier_options)s
        %(regressor_options)s

    X : list of Niimg-like objects
        See :ref:`extracting_data`.
        Data on which model is to be fitted. If this is a list,
        the affine is considered the same for all.

    y : array or list of shape (n_samples)
        The dependent variable (age, sex, IQ, yes/no, etc.).
        Target variable to predict. Must have exactly as many elements as
        3D images in niimg.

    Returns
    -------
    param_grid : dict of str to sequence, or sequence of such. Sensible default
    dict has size 1 for linear models.
    """
    param_grid = {}

    # validate estimator
    if isinstance(estimator, (DummyClassifier, DummyRegressor)):
        if estimator.strategy == "constant":
            message = (
                "Dummy classification implemented only for strategies"
                ' "most_frequent", "prior", "stratified"'
            )
            raise NotImplementedError(message)
    elif not isinstance(
        estimator,
        (
            LogisticRegressionCV,
            LinearSVC,
            RidgeCV,
            RidgeClassifierCV,
            SVR,
            LassoCV,
        ),
    ):
        raise ValueError(
            "Invalid estimator. The supported estimators are:"
            f" {list(SUPPORTED_ESTIMATORS.keys())}"
        )

    # use l1_min_c to get lower bound for estimators with L1 penalty
    if hasattr(estimator, "penalty") and (estimator.penalty == "l1"):
        # define loss function
        if isinstance(estimator, LogisticRegressionCV):
            loss = "log"
        elif isinstance(estimator, LinearSVC):
            loss = "squared_hinge"

        min_c = l1_min_c(X, y, loss=loss)

    # otherwise use 0.5 which will give param_grid["C"] = [1, 10, 100]
    else:
        min_c = 0.5

    # define sensible default for different types of estimators
    if isinstance(estimator, (RidgeCV, RidgeClassifierCV)):
        param_grid["alphas"] = [np.geomspace(1e-3, 1e4, 8)]
    elif isinstance(estimator, LogisticRegressionCV):
        # min_c value is set to 0.5 unless the estimator uses L1 penalty,
        # in which case min_c is computed with sklearn.svm.l1_min_c(),
        # so for L2 penalty, param_grid["Cs"] is either 1e-3, ..., 1e4, and
        # for L1 penalty the values are obtained in a more data-driven way
        param_grid["Cs"] = [np.geomspace(2e-3, 2e4, 8) * min_c]
    elif isinstance(estimator, LassoCV):
        # the default is to generate 30 alphas based on the data
        # (alpha values can also be set with the 'alphas' parameter, in which
        # case 'n_alphas' is ignored)
        param_grid["n_alphas"] = [30]
    elif isinstance(estimator, (LinearSVC, SVR)):
        # similar logic as above:
        # - for L2 penalty this is [1, 10, 100]
        # - for L1 penalty the values depend on the data
        param_grid["C"] = np.array([2, 20, 200]) * min_c
    else:
        param_grid = {}

    return param_grid


def _wrap_param_grid(param_grid, param_name):
    """Wrap a parameter's sequence of values with an outer list.

    This can be desirable for models featuring built-in cross-validation,
    as it would leave it to the model's internal (optimized) cross-validation
    to loop over hyperparameter values. Does nothing if the parameter is
    already wrapped.

    Parameters
    ----------
    param_grid : dict of str to sequence, or sequence of such
        The parameter grid to wrap, as a dictionary mapping estimator
        parameters to sequences of allowed values.
    param_name : str
        Name of parameter whose sequence of values should be wrapped

    Returns
    -------
    dict of str to sequence, or sequence of such
        The updated parameter grid
    """
    if param_grid is None:
        return param_grid

    # param_grid can be either a dict or a sequence of dicts
    # we make sure that it is a sequence we can loop over
    input_is_dict = isinstance(param_grid, dict)
    if input_is_dict:
        param_grid = [param_grid]

    # process dicts one by one and add them to a new list
    new_param_grid = []
    for param_grid_item in param_grid:
        if param_name in param_grid_item and not isinstance(
            param_grid_item[param_name][0], Iterable
        ):
            warnings.warn(
                f"parameter '{param_name}' should be a sequence of iterables"
                f" (e.g., {{param_name: [[1, 10, 100]]}}) to benefit from"
                " the built-in cross-validation of the estimator."
                f" Wrapping {param_grid_item[param_name]} in an outer list."
            )

            param_grid_item = dict(param_grid_item)  # make a new dict
            param_grid_item[param_name] = [param_grid_item[param_name]]

        new_param_grid.append(param_grid_item)

    # return a dict (not a list) if the original input was a dict
    if input_is_dict:
        new_param_grid = new_param_grid[0]

    return new_param_grid


def _replace_param_grid_key(param_grid, key_to_replace, new_key):
    """Replace a parameter name by another one.

    Parameters
    ----------
    param_grid : dict of str to sequence, or sequence of such
        The parameter grid to process, as a dictionary mapping estimator
        parameters to sequences of allowed values.
    key_to_replace : str
        Name of parameter to replace
    new_key : str
        New parameter name. If this key already exists in the parameter grid,
        it is overwritten

    Returns
    -------
    dict of str to sequence, or sequence of such
        The updated parameter grid
    """
    # ensure param_grid is a list so that we can loop over it
    input_is_dict = isinstance(param_grid, dict)
    if input_is_dict:
        param_grid = [param_grid]

    # replace old key by new key if needed
    new_param_grid = []
    for param_grid_item in param_grid:
        param_grid_item = dict(param_grid_item)  # make a new dict
        if key_to_replace in param_grid_item:
            warnings.warn(
                f'The "{key_to_replace}" parameter in "param_grid" is'
                f' being replaced by "{new_key}" due to a change in the'
                " choice of underlying scikit-learn estimator. In a future"
                " version, this will result in an error.",
                DeprecationWarning,
                stacklevel=13,
            )
            param_grid_item[new_key] = param_grid_item.pop(key_to_replace)
        new_param_grid.append(param_grid_item)

    # return a dict if input was a dict
    if input_is_dict:
        new_param_grid = new_param_grid[0]

    return new_param_grid


def _check_estimator(estimator):
    if not isinstance(estimator, str):
        warnings.warn(
            "Use a custom estimator at your own risk "
            "of the process not working as intended."
        )
    elif estimator in SUPPORTED_ESTIMATORS:
        estimator = SUPPORTED_ESTIMATORS.get(estimator)
    else:
        raise ValueError(
            "Invalid estimator. Known estimators are: "
            f"{list(SUPPORTED_ESTIMATORS.keys())}"
        )

    return estimator


def _parallel_fit(
    estimator,
    X,
    y,
    train,
    test,
    param_grid,
    is_classification,
    selector,
    scorer,
    mask_img,
    class_index,
    clustering_percentile,
):
    """Find the best estimator for a fold within a job.

    This function tries several parameters for the estimator for the train and
    test fold provided and save the one that performs best.

    Fit may be performed after some preprocessing step :
    * clustering with ReNA if clustering_percentile < 100
    * feature screening if screening_percentile < 100
    """
    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]

    # for FREM Classifier and Regressor : start by doing a quick ReNA
    # clustering to reduce the number of feature by agglomerating similar ones

    if clustering_percentile < 100:
        n_clusters = int(X_train.shape[1] * clustering_percentile / 100.0)
        clustering = ReNA(
            mask_img,
            n_clusters=n_clusters,
            n_iter=20,
            threshold=1e-7,
            scaling=False,
        )
        X_train = clustering.fit_transform(X_train)
        X_test = clustering.transform(X_test)

    do_screening = (X_train.shape[1] > 100) and selector is not None

    if do_screening:
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)

    # If there is no parameter grid, then we use a suitable grid (by default)
    param_grid = ParameterGrid(
        _check_param_grid(estimator, X_train, y_train, param_grid)
    )

    # collect all parameter names from the grid
    all_params = set()
    for params in param_grid:
        all_params.update(params.keys())

    best_score = None
    for params in param_grid:
        estimator = clone(estimator).set_params(**params)
        estimator.fit(X_train, y_train)

        if is_classification:
            score = scorer(estimator, X_test, y_test)
            if hasattr(estimator, "coef_") and np.all(estimator.coef_ == 0):
                score = 0
        else:  # regression
            score = scorer(estimator, X_test, y_test)

        # Store best parameters and estimator coefficients
        if (best_score is None) or (score >= best_score):
            best_score = score
            if hasattr(estimator, "coef_"):
                best_coef = np.reshape(estimator.coef_, (1, -1))
                best_intercept = estimator.intercept_
                dummy_output = None
            else:
                best_coef, best_intercept = None, None
                if isinstance(estimator, DummyClassifier):
                    dummy_output = estimator.class_prior_
                elif isinstance(estimator, DummyRegressor):
                    dummy_output = estimator.constant_

            if isinstance(estimator, (RidgeCV, RidgeClassifierCV, LassoCV)):
                params["best_alpha"] = estimator.alpha_
            elif isinstance(estimator, LogisticRegressionCV):
                params["best_C"] = estimator.C_.item()
            best_params = params

            # fill in any missing param from param_grid
            for param in all_params:
                if param not in best_params:
                    best_params[param] = getattr(estimator, param)

    if best_coef is not None:
        if do_screening:
            best_coef = selector.inverse_transform(best_coef)

        if clustering_percentile < 100:
            best_coef = clustering.inverse_transform(best_coef)

    return (
        class_index,
        best_coef,
        best_intercept,
        best_params,
        best_score,
        dummy_output,
    )


@fill_doc
class _BaseDecoder(CacheMixin, BaseEstimator):
    """A wrapper for popular classification/regression strategies in \
    neuroimaging.

    The `BaseDecoder` object supports classification and regression methods.
    It implements a model selection scheme that averages the best models
    within a cross validation loop (a technique sometimes known as CV bagging).
    The resulting average model is the one used as a classifier or a regressor.
    This object also leverages the `NiftiMaskers` to provide a direct interface
    with the Nifti files on disk.

    Parameters
    ----------
    estimator : str, default='svc'
        The estimator to use. For classification, choose among:
        %(classifier_options)s
        For regression, choose among:
        %(regressor_options)s

    mask : filename, Nifti1Image, NiftiMasker, MultiNiftiMasker, or\
          SurfaceMasker, default=None
        Mask to be used on data. If an instance of masker is passed,
        then its mask and parameters will be used. If no mask is given, mask
        will be computed automatically from provided images by an inbuilt
        masker with default parameters. Refer to NiftiMasker, MultiNiftiMasker
        or SurfaceMasker to check for default parameters. For use with
        SurfaceImage data, a SurfaceMasker instance must be passed.

    cv : cross-validation generator or int, default=10
        A cross-validation generator.
        See: https://scikit-learn.org/stable/modules/cross_validation.html

    param_grid : dict of str to sequence, or sequence of such, default=None
        The parameter grid to explore, as a dictionary mapping estimator
        parameters to sequences of allowed values.

        None or an empty dict signifies default parameters.

        A sequence of dicts signifies a sequence of grids to search, and is
        useful to avoid exploring parameter combinations that make no sense
        or have no effect. See scikit-learn documentation for more information,
        for example: https://scikit-learn.org/stable/modules/grid_search.html

        For Dummy estimators, parameter grid defaults to empty dictionary.

    clustering_percentile : int, float, in the [0, 100], default=100
        Percentile of features to keep after clustering. If it is lower
        than 100, a ReNA clustering is performed as a first step of fit
        to agglomerate similar features together. ReNA is typically efficient
        for clustering_percentile equal to 10. Only used with
        :class:`nilearn.decoding.FREMClassifier` and
        :class:`nilearn.decoding.FREMRegressor`.

    screening_percentile : int, float, \
                          in the closed interval [0, 100], \
                          default=20
        The percentage of brain volume that will be kept with respect to a full
        MNI template. In particular, if it is lower than 100, a univariate
        feature selection based on the Anova F-value for the input data will be
        performed. A float according to a percentile of the highest
        scores. If None is passed, the percentile is set to 100.

    scoring : str, callable or None,
             default=None
        The scoring strategy to use. See the scikit-learn documentation at
        https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules
        If callable, takes as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.
        e.g. scorer(estimator, X_test, y_test)

        For regression, valid entries are: 'r2', 'neg_mean_absolute_error', or
        'neg_mean_squared_error'. Defaults to 'r2'.

        For classification, valid entries are: 'accuracy', 'f1', 'precision',
        'recall' or 'roc_auc'. Defaults to 'roc_auc'.

    %(smoothing_fwhm)s

    %(standardize)s

    %(target_affine)s

    %(target_shape)s

    %(low_pass)s

    %(high_pass)s

    %(t_r)s

    %(mask_strategy)s

        .. note::
            This parameter will be ignored if a mask image is provided.

        .. note::
            Depending on this value, the mask will be computed from
            :func:`nilearn.masking.compute_background_mask`,
            :func:`nilearn.masking.compute_epi_mask`, or
            :func:`nilearn.masking.compute_brain_mask`.

        Default is 'background'.

    %(memory)s

    %(memory_level)s

    %(n_jobs)s

    %(verbose0)s

    See Also
    --------
    nilearn.decoding.Decoder: Classification strategies for Neuroimaging,
    nilearn.decoding.DecoderRegressor: Regression strategies for Neuroimaging,
    nilearn.decoding.FREMClassifier: State of the art classification pipeline
        for Neuroimaging
    nilearn.decoding.FREMRegressor: State of the art regression pipeline
        for Neuroimaging
    nilearn.decoding.SpaceNetClassifier: Graph-Net and TV-L1 priors/penalties

    """

    def __init__(
        self,
        estimator="svc",
        mask=None,
        cv=10,
        param_grid=None,
        clustering_percentile=100,
        screening_percentile=20,
        scoring=None,
        smoothing_fwhm=None,
        standardize=True,
        target_affine=None,
        target_shape=None,
        low_pass=None,
        high_pass=None,
        t_r=None,
        mask_strategy="background",
        is_classification=True,
        memory=None,
        memory_level=0,
        n_jobs=1,
        verbose=0,
    ):
        self.estimator = estimator
        self.mask = mask
        self.cv = cv
        self.param_grid = param_grid
        self.screening_percentile = screening_percentile
        self.scoring = scoring
        self.is_classification = is_classification
        self.clustering_percentile = clustering_percentile
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

    def fit(self, X, y, groups=None):
        """Fit the decoder (learner).

        Parameters
        ----------
        X : list of Niimg-like or :obj:`~nilearn.surface.SurfaceImage` objects
            See :ref:`extracting_data`.
            Data on which model is to be fitted. If this is a list,
            the affine is considered the same for all.

        y : numpy.ndarray of shape=(n_samples) or list of length n_samples
            The dependent variable (age, sex, IQ, yes/no, etc.).
            Target variable to predict. Must have exactly as many elements as
            3D images in niimg.

        groups : None
            Group labels for the samples used while splitting the dataset into
            train/test set. Default None.

            Note that this parameter must be specified in some scikit-learn
            cross-validation generators to calculate the number of splits, e.g.
            sklearn.model_selection.LeaveOneGroupOut or
            sklearn.model_selection.LeavePGroupsOut.

            For more details see
            https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data

        Attributes
        ----------
        masker_ : instance of NiftiMasker, MultiNiftiMasker, or SurfaceMasker
            The masker used to mask the data.

        mask_img_ : Nifti1Image or :obj:`~nilearn.surface.SurfaceImage`
            Mask computed by the masker object.

        classes_ : numpy.ndarray
            Classes to predict. For classification only.

        screening_percentile_ : float
            Screening percentile corrected according to volume of mask,
            relative to the volume of standard brain.

        coef_ : numpy.ndarray, shape=(n_classes, n_features)
            Contains the mean of the models weight vector across
            fold for each class. Returns None for Dummy estimators.

        coef_img_ : dict of Nifti1Image
            Dictionary containing `coef_` with class names as keys,
            and `coef_` transformed in Nifti1Images as values. In the case of
            a regression, it contains a single Nifti1Image at the key 'beta'.
            Ignored if Dummy estimators are provided.

        intercept_ : ndarray, shape (nclasses,)
            Intercept (a.k.a. bias) added to the decision function.
            Ignored if Dummy estimators are provided.

        cv_ : list of pairs of lists
            List of the (n_folds,) folds. For the corresponding fold,
            each pair is composed of two lists of indices,
            one for the train samples and one for the test samples.

        std_coef_ : numpy.ndarray, shape=(n_classes, n_features)
            Contains the standard deviation of the models weight vector across
            fold for each class. Note that folds are not independent, see
            https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data
            Ignored if Dummy estimators are provided.

        std_coef_img_ : dict of Nifti1Image
            Dictionary containing `std_coef_` with class names as keys,
            and `coef_` transformed in Nifti1Image as values. In the case of
            a regression, it contains a single Nifti1Image at the key 'beta'.
            Ignored if Dummy estimators are provided.

        cv_params_ : dict of lists
            Best point in the parameter grid for each tested fold
            in the inner cross validation loop. The grid is empty
            when Dummy estimators are provided. Note: if the estimator used its
            built-in cross-validation, this will include an additional key for
            the single best value estimated by the built-in cross-validation
            ('best_C' for LogisticRegressionCV and 'best_alpha' for
            RidgeCV/RidgeClassifierCV/LassoCV), in addition to the input list
            of values.

        scorer_ : function
            Scorer function used on the held out data to choose the best
            parameters for the model.

        cv_scores_ : dict, (classes, n_folds)
            Scores (misclassification) for each parameter, and on each fold

        n_outputs_ : int
            Number of outputs (column-wise)

        dummy_output_ : ndarray, shape=(n_classes, 2) \
                       or shape=(1, 1) for regression
            Contains dummy estimator attributes after class predictions
            using strategies of :class:`sklearn.dummy.DummyClassifier`
            (class_prior)
            and  :class:`sklearn.dummy.DummyRegressor` (constant)
            from scikit-learn.
            This attribute is necessary for estimating class predictions
            after fit.
            Returns None if non-dummy estimators are provided.
        """
        self.estimator = _check_estimator(self.estimator)
        self.memory_ = _check_memory(self.memory, self.verbose)

        X = self._apply_mask(X)
        X, y = check_X_y(X, y, dtype=np.float64, multi_output=True)

        self.n_outputs_ = 1 if y.ndim == 1 else y.shape[1]

        self._set_scorer()

        # Setup cross-validation object. Default is StratifiedKFold when groups
        # is None. If groups is specified but self.cv is not set to custom CV
        # splitter, default is LeaveOneGroupOut. If self.cv is manually set to
        # a CV splitter object do check_cv regardless of groups parameter.
        cv = self.cv
        if (isinstance(cv, int) or cv is None) and groups is not None:
            warnings.warn(
                "groups parameter is specified but "
                "cv parameter is not set to custom CV splitter. "
                "Using default object LeaveOneGroupOut()."
            )
            cv_object = LeaveOneGroupOut()
        else:
            cv_object = check_cv(cv, y=y, classifier=self.is_classification)

        self.cv_ = list(cv_object.split(X, y, groups=groups))

        # Define the number problems to solve. In case of classification this
        # number corresponds to the number of binary problems to solve
        y = self._binarize_y(y) if self.is_classification else y[:, np.newaxis]

        if self.is_classification and self.n_classes_ > 2:
            n_problems = self.n_classes_
        else:
            n_problems = 1

        # Check if the size of the mask image and the number of features allow
        # to perform feature screening.
        # If the input data is a SurfaceImage, the number of vertices in the
        # mesh is needed to perform feature screening.
        mesh_n_vertices = (
            self.mask_img_.mesh.n_vertices
            if isinstance(self.mask_img_, SurfaceImage)
            else None
        )
        selector = check_feature_screening(
            self.screening_percentile,
            self.mask_img_,
            self.is_classification,
            mesh_n_vertices=mesh_n_vertices,
        )

        # Return a suitable screening percentile according to the mask image
        if hasattr(selector, "percentile"):
            self.screening_percentile_ = selector.percentile
        elif self.screening_percentile is None:
            self.screening_percentile_ = 100.0
        else:
            self.screening_percentile_ = self.screening_percentile

        n_final_features = int(
            X.shape[1]
            * self.screening_percentile_
            * self.clustering_percentile
            / 10000
        )
        if n_final_features < 50:
            warnings.warn(
                "After clustering and screening, the decoding model will "
                f"be trained only on {n_final_features} features. "
                "Consider raising clustering_percentile or "
                "screening_percentile parameters.",
                UserWarning,
            )

        parallel = Parallel(n_jobs=self.n_jobs, verbose=2 * self.verbose)

        parallel_fit_outputs = parallel(
            delayed(self._cache(_parallel_fit))(
                estimator=self.estimator,
                X=X,
                y=y[:, c],
                train=train,
                test=test,
                param_grid=self.param_grid,
                is_classification=self.is_classification,
                selector=selector,
                scorer=self.scorer_,
                mask_img=self.mask_img_,
                class_index=c,
                clustering_percentile=self.clustering_percentile,
            )
            for c, (train, test) in itertools.product(
                range(n_problems), self.cv_
            )
        )

        coefs, intercepts = self._fetch_parallel_fit_outputs(
            parallel_fit_outputs, y, n_problems
        )

        # Build the final model (the aggregated one)
        if not isinstance(self.estimator, (DummyClassifier, DummyRegressor)):
            self.coef_ = np.vstack(
                [
                    np.mean(coefs[class_index], axis=0)
                    for class_index in self.classes_
                ]
            )
            self.std_coef_ = np.vstack(
                [
                    np.std(coefs[class_index], axis=0)
                    for class_index in self.classes_
                ]
            )
            self.intercept_ = np.hstack(
                [
                    np.mean(intercepts[class_index], axis=0)
                    for class_index in self.classes_
                ]
            )

            self.coef_img_, self.std_coef_img_ = self._output_image(
                self.classes_, self.coef_, self.std_coef_
            )

            if self.is_classification and (self.n_classes_ == 2):
                self.coef_ = self.coef_[0, :][np.newaxis, :]
                self.intercept_ = self.intercept_[0]
        else:
            # For Dummy estimators
            self.coef_ = None
            self.dummy_output_ = np.vstack(
                [
                    np.mean(self.dummy_output_[class_index], axis=0)
                    for class_index in self.classes_
                ]
            )
            if self.is_classification and (self.n_classes_ == 2):
                self.dummy_output_ = self.dummy_output_[0, :][np.newaxis, :]

    def score(self, X, y, *args):
        """Compute the prediction score using the scoring \
        metric defined by the scoring attribute.

        Parameters
        ----------
        X : Niimg-like, :obj:`list` of either \
            Niimg-like objects or :obj:`str` or path-like
            See :ref:`extracting_data`.
            Data on which prediction is to be made.

        y : :class:`numpy.ndarray`
            Target values.

        args : Optional arguments that can be passed to
            scoring metrics. Example: sample_weight.

        Returns
        -------
        score : float
            Prediction score.

        """
        check_is_fitted(self, "coef_")
        check_is_fitted(self, "masker_")
        return self.scorer_(self, X, y, *args)

    def decision_function(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : Niimg-like, :obj:`list` of either \
            Niimg-like objects or :obj:`str` or path-like
            See :ref:`extracting_data`.
            Data on prediction is to be made. If this is a list,
            the affine is considered the same for all.

        Returns
        -------
        y_pred : :class:`numpy.ndarray`, shape (n_samples,)
            Predicted class label per sample.
        """
        # for backwards compatibility - apply masker transform if X is
        # niimg-like or a list of strings
        if not isinstance(X, np.ndarray) or len(np.shape(X)) == 1:
            X = self.masker_.transform(X)
        n_features = self.coef_.shape[1]
        if X.shape[1] != n_features:
            raise ValueError(
                f"X has {X.shape[1]} features per sample;"
                f" expecting {n_features}"
            )

        scores = (
            safe_sparse_dot(X, self.coef_.T, dense_output=True)
            + self.intercept_
        )

        return scores.ravel() if scores.shape[1] == 1 else scores

    def predict(self, X):
        """Predict a label for all X vectors indexed by the first axis.

        Parameters
        ----------
        X : Niimg-like, :obj:`list` of either \
            Niimg-like objects or :obj:`str` or path-like
            See :ref:`extracting_data`.
            Data on which prediction is to be made.

        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
            Confidence scores per (sample, class) combination. In the binary
            case, confidence score for self.classes_[1] where >0 means this
            class would be predicted.
        """
        check_is_fitted(self, "coef_")
        check_is_fitted(self, "masker_")

        n_samples = np.shape(X)[-1]

        # Prediction for dummy estimator is different from others as there is
        # no fitted coefficient
        if isinstance(self.estimator, (DummyClassifier, DummyRegressor)):
            scores = self._predict_dummy(n_samples)
        else:
            scores = self.decision_function(X)

        if self.is_classification:
            if scores.ndim == 1:
                indices = (scores > 0).astype(int)
            else:
                indices = scores.argmax(axis=1)
            return self.classes_[indices]

        return scores

    def _apply_mask(self, X):
        masker_type = "nii"
        # all elements of X should be of the similar type by now
        # so we can only check the first one
        to_check = X[0] if isinstance(X, Iterable) else X
        if isinstance(self.mask, (SurfaceMasker, SurfaceImage)) or (
            isinstance(to_check, SurfaceImage)
        ):
            masker_type = "surface"

        self.masker_ = check_embedded_masker(self, masker_type=masker_type)
        check_compatibility_mask_and_images(self.mask, X)

        X = self.masker_.fit_transform(X)
        self.mask_img_ = self.masker_.mask_img_

        return X

    def _fetch_parallel_fit_outputs(
        self,
        parallel_fit_outputs,
        y,  # noqa: ARG002
        n_problems,
    ):
        """Fetch the outputs from parallel_fit to be ready for ensembling.

        Parameters
        ----------
        parallel_fit_outputs : list of tuples,
            each tuple contains results of
            one _parallel_fit for each cv fold (and each classification in the
            case of multiclass classification).

        y : ndarray, shape = (n_samples, )
            Vector of responses.

        Returns
        -------
        coefs : dict
            Coefficients for each classification/regression problem
        intercepts : dict
            Intercept for each classification/regression problem
        """
        coefs = {}
        intercepts = {}
        cv_scores = {}
        self.cv_params_ = {}
        self.dummy_output_ = {}
        classes = self.classes_

        for (
            class_index,
            coef,
            intercept,
            params,
            scores,
            dummy_output,
        ) in parallel_fit_outputs:
            coefs.setdefault(classes[class_index], []).append(coef)
            intercepts.setdefault(classes[class_index], []).append(intercept)

            cv_scores.setdefault(classes[class_index], []).append(scores)

            self.cv_params_.setdefault(classes[class_index], {})
            if isinstance(self.estimator, (DummyClassifier, DummyRegressor)):
                self.dummy_output_.setdefault(classes[class_index], []).append(
                    dummy_output
                )
            else:
                self.dummy_output_.setdefault(classes[class_index], []).append(
                    None
                )
            for k in params:
                self.cv_params_[classes[class_index]].setdefault(k, []).append(
                    params[k]
                )

            if (n_problems <= 2) and self.is_classification:
                # Binary classification
                other_class = np.setdiff1d(classes, classes[class_index])[0]
                if coef is not None:
                    coefs.setdefault(other_class, []).append(-coef)
                    intercepts.setdefault(other_class, []).append(-intercept)
                else:
                    coefs.setdefault(other_class, []).append(None)
                    intercepts.setdefault(other_class, []).append(None)

                cv_scores.setdefault(other_class, []).append(scores)
                self.cv_params_[other_class] = self.cv_params_[
                    classes[class_index]
                ]
                if isinstance(
                    self.estimator, (DummyClassifier, DummyRegressor)
                ):
                    self.dummy_output_.setdefault(other_class, []).append(
                        dummy_output
                    )
                else:
                    self.dummy_output_.setdefault(other_class, []).append(None)

        self.cv_scores_ = cv_scores

        return coefs, intercepts

    def _set_scorer(self):
        if self.scoring is not None:
            self.scorer_ = check_scoring(self.estimator, self.scoring)
        elif self.is_classification:
            self.scorer_ = get_scorer("accuracy")
        else:
            self.scorer_ = get_scorer("r2")

    def _output_image(self, classes, coefs, std_coef):
        coef_img = {}
        std_coef_img = {}
        for class_index, coef, std in zip(classes, coefs, std_coef):
            coef_img[class_index] = self.masker_.inverse_transform(coef)
            std_coef_img[class_index] = self.masker_.inverse_transform(std)

        return coef_img, std_coef_img

    def _binarize_y(self, y):
        """Encode target classes as -1 and 1.

        Helper function invoked just before fitting a classifier.
        """
        y = np.array(y)

        self._enc = LabelBinarizer(pos_label=1, neg_label=-1)
        y = self._enc.fit_transform(y)
        self.classes_ = self._enc.classes_
        self.n_classes_ = len(self.classes_)
        return y

    def _predict_dummy(self, n_samples):
        """Non-sparse scikit-learn based prediction steps for classification \
        and regression.
        """
        if len(self.dummy_output_) == 1:
            dummy_output = self.dummy_output_[0]
        else:
            dummy_output = self.dummy_output_[:, 1]
        if isinstance(self.estimator, DummyClassifier):
            strategy = self.estimator.get_params()["strategy"]
            if strategy in ["most_frequent", "prior"]:
                scores = np.tile(dummy_output, reps=(n_samples, 1))
            elif strategy == "stratified":
                rs = np.random.default_rng(0)
                scores = rs.multinomial(1, dummy_output, size=n_samples)

        elif isinstance(self.estimator, DummyRegressor):
            scores = np.full(
                (n_samples, self.n_outputs_),
                self.dummy_output_,
                dtype=np.array(self.dummy_output_).dtype,
            )
        return scores.ravel() if scores.shape[1] == 1 else scores

    def _more_tags(self):
        """Return estimator tags.

        TODO remove when bumping sklearn_version > 1.5
        """
        return self.__sklearn_tags__()

    def __sklearn_tags__(self):
        """Return estimator tags.

        See the sklearn documentation for more details on tags
        https://scikit-learn.org/1.6/developers/develop.html#estimator-tags
        """
        # TODO
        # get rid of if block
        # bumping sklearn_version > 1.5
        # see https://github.com/scikit-learn/scikit-learn/pull/29677
        if SKLEARN_LT_1_6:
            from nilearn._utils.tags import tags

            return tags(require_y=True, niimg_like=True, surf_img=True)

        from nilearn._utils.tags import InputTags

        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.input_tags = InputTags(niimg_like=True, surf_img=True)
        return tags


@fill_doc
class Decoder(ClassifierMixin, _BaseDecoder):
    """A wrapper for popular classification strategies in neuroimaging.

    The `Decoder` object supports classification methods.
    It implements a model selection scheme that averages the best models
    within a cross validation loop. The resulting average model is the
    one used as a classifier. This object also leverages the`NiftiMaskers` to
    provide a direct interface with the Nifti files on disk.

    Parameters
    ----------
    estimator : str, default='svc'
        The estimator to choose among:
        %(classifier_options)s

    mask : filename, Nifti1Image, NiftiMasker, MultiNiftiMasker, \
           :obj:`~nilearn.surface.SurfaceImage` \
           or :obj:`~nilearn.maskers.SurfaceMasker`, default=None
        Mask to be used on data. If an instance of masker is passed,
        then its mask and parameters will be used. If no mask is given, mask
        will be computed automatically from provided images by an inbuilt
        masker with default parameters.
        Refer to :obj:`~nilearn.maskers.NiftiMasker` or
        :obj:`~nilearn.maskers.MultiNiftiMasker` or
        :obj:`~nilearn.maskers.SurfaceMasker`
        to check for default parameters.

    cv : cross-validation generator or int, default=10
        A cross-validation generator.
        See: https://scikit-learn.org/stable/modules/cross_validation.html.
        The default 10 refers to K = 10 folds of
        :class:`~sklearn.model_selection.StratifiedKFold` when groups is None
        in the fit method for this class. If groups is specified but ``cv``
        is not set to custom CV splitter, default is
        :class:`~sklearn.model_selection.LeaveOneGroupOut`.

    param_grid : dict of str to sequence, or sequence of such, default=None
        The parameter grid to explore, as a dictionary mapping estimator
        parameters to sequences of allowed values.

        None or an empty dict signifies default parameters.

        A sequence of dicts signifies a sequence of grids to search, and is
        useful to avoid exploring parameter combinations that make no sense
        or have no effect. See scikit-learn documentation for more information,
        for example: https://scikit-learn.org/stable/modules/grid_search.html

        For DummyClassifier, parameter grid defaults to empty dictionary, class
        predictions are estimated using default strategy.

    screening_percentile : int, float, optional, \
                          in the closed interval [0, 100], default=20
        The percentage of brain volume that will be kept with respect to a full
        MNI template. In particular, if it is lower than 100, a univariate
        feature selection based on the Anova F-value for the input data will be
        performed. A float according to a percentile of the highest scores.

    scoring : str, callable or None, default='roc_auc'
        The scoring strategy to use. See the scikit-learn documentation at
        https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules
        If callable, takes as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.
        e.g. scorer(estimator, X_test, y_test)

        For classification, valid entries are: 'accuracy', 'f1', 'precision',
        'recall' or 'roc_auc'.

    %(smoothing_fwhm)s

    %(standardize)s

    %(target_affine)s

    %(target_shape)s

    %(low_pass)s

    %(high_pass)s

    %(t_r)s

    %(mask_strategy)s

        .. note::
            This parameter will be ignored if a mask image is provided.

        .. note::
            Depending on this value, the mask will be computed from
            :func:`nilearn.masking.compute_background_mask`,
            :func:`nilearn.masking.compute_epi_mask`, or
            :func:`nilearn.masking.compute_brain_mask`.

        Default='background'.

    %(memory)s

    %(memory_level)s

    %(n_jobs)s

    %(verbose0)s

    See Also
    --------
    nilearn.decoding.DecoderRegressor: regression strategies for Neuro-imaging,
    nilearn.decoding.FREMClassifier: State of the art classification pipeline
        for Neuroimaging
    nilearn.decoding.SpaceNetClassifier: Graph-Net and TV-L1 priors/penalties
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        estimator="svc",
        mask=None,
        cv=10,
        param_grid=None,
        screening_percentile=20,
        scoring="roc_auc",
        smoothing_fwhm=None,
        standardize=True,
        target_affine=None,
        target_shape=None,
        mask_strategy="background",
        low_pass=None,
        high_pass=None,
        t_r=None,
        memory=None,
        memory_level=0,
        n_jobs=1,
        verbose=0,
    ):
        super().__init__(
            estimator=estimator,
            mask=mask,
            cv=cv,
            param_grid=param_grid,
            screening_percentile=screening_percentile,
            scoring=scoring,
            smoothing_fwhm=smoothing_fwhm,
            standardize=standardize,
            target_affine=target_affine,
            target_shape=target_shape,
            mask_strategy=mask_strategy,
            low_pass=low_pass,
            high_pass=high_pass,
            t_r=t_r,
            memory=memory,
            is_classification=True,
            memory_level=memory_level,
            verbose=verbose,
            n_jobs=n_jobs,
        )

    def _more_tags(self):
        """Return estimator tags.

        TODO remove when bumping sklearn_version > 1.5
        """
        return self.__sklearn_tags__()

    def __sklearn_tags__(self):
        """Return estimator tags.

        See the sklearn documentation for more details on tags
        https://scikit-learn.org/1.6/developers/develop.html#estimator-tags
        """
        # TODO
        # get rid of if block
        # bumping sklearn_version > 1.5
        # see https://github.com/scikit-learn/scikit-learn/pull/29677
        tags = super().__sklearn_tags__()
        if SKLEARN_LT_1_6:
            return tags
        tags.estimator_type = "classifier"
        return tags


@fill_doc
class DecoderRegressor(MultiOutputMixin, RegressorMixin, _BaseDecoder):
    """A wrapper for popular regression strategies in neuroimaging.

    The `DecoderRegressor` object supports regression methods.
    It implements a model selection scheme that averages the best models
    within a cross validation loop. The resulting average model is the
    one used as a regressor. This object also leverages the `NiftiMaskers`
    to provide a direct interface with the Nifti files on disk.

    Parameters
    ----------
    estimator : str, optional
        The estimator to choose among:
        %(regressor_options)s
        Default 'svr'.

    mask : filename, Nifti1Image, NiftiMasker, or MultiNiftiMasker, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask and parameters will be used. If no mask is given, mask
        will be computed automatically from provided images by an inbuilt
        masker with default parameters. Refer to NiftiMasker or
        MultiNiftiMasker to check for default parameters. Default None

    cv : cross-validation generator or int, default=10
        A cross-validation generator.
        See: https://scikit-learn.org/stable/modules/cross_validation.html.
        The default 10 refers to K = 10 folds of
        :class:`~sklearn.model_selection.StratifiedKFold` when groups is None
        in the fit method for this class. If groups is specified but ``cv``
        is not set to custom CV splitter, default is
        :class:`~sklearn.model_selection.LeaveOneGroupOut`.

    param_grid : dict of str to sequence, or sequence of such, default=None
        The parameter grid to explore, as a dictionary mapping estimator
        parameters to sequences of allowed values.

        None or an empty dict signifies default parameters.

        A sequence of dicts signifies a sequence of grids to search, and is
        useful to avoid exploring parameter combinations that make no sense
        or have no effect. See scikit-learn documentation for more information,
        for example: https://scikit-learn.org/stable/modules/grid_search.html

        For DummyRegressor, parameter grid defaults to empty dictionary, class
        predictions are estimated using default strategy.

    screening_percentile : int, float, \
                          in the closed interval [0, 100], \
                          default=20
        The percentage of brain volume that will be kept with respect to a full
        MNI template. In particular, if it is lower than 100, a univariate
        feature selection based on the Anova F-value for the input data will be
        performed. A float according to a percentile of the highest
        scores.

    scoring : str, callable or None, optional. default='r2'
        The scoring strategy to use. See the scikit-learn documentation at
        https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules
        If callable, takes as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.
        e.g. scorer(estimator, X_test, y_test)

        For regression, valid entries are: 'r2', 'neg_mean_absolute_error',
        or 'neg_mean_squared_error'.

    %(smoothing_fwhm)s

    %(standardize)s

    %(target_affine)s

    %(target_shape)s

    %(low_pass)s

    %(high_pass)s

    %(t_r)s

    %(mask_strategy)s

        .. note::
            This parameter will be ignored if a mask image is provided.

        .. note::
            Depending on this value, the mask will be computed from
            :func:`nilearn.masking.compute_background_mask`,
            :func:`nilearn.masking.compute_epi_mask`, or
            :func:`nilearn.masking.compute_brain_mask`.

        Default='background'.

    %(memory)s

    %(memory_level)s

    %(n_jobs)s

    %(verbose0)s

    See Also
    --------
    nilearn.decoding.Decoder: classification strategies for Neuroimaging,
    nilearn.decoding.FREMRegressor: State of the art regression pipeline
        for Neuroimaging
    nilearn.decoding.SpaceNetClassifier: Graph-Net and TV-L1 priors/penalties
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        estimator="svr",
        mask=None,
        cv=10,
        param_grid=None,
        screening_percentile=20,
        scoring="r2",
        smoothing_fwhm=None,
        standardize=True,
        target_affine=None,
        target_shape=None,
        mask_strategy="background",
        low_pass=None,
        high_pass=None,
        t_r=None,
        memory=None,
        memory_level=0,
        n_jobs=1,
        verbose=0,
    ):
        self.classes_ = ["beta"]

        super().__init__(
            estimator=estimator,
            mask=mask,
            cv=cv,
            param_grid=param_grid,
            screening_percentile=screening_percentile,
            scoring=scoring,
            smoothing_fwhm=smoothing_fwhm,
            standardize=standardize,
            target_affine=target_affine,
            target_shape=target_shape,
            low_pass=low_pass,
            high_pass=high_pass,
            t_r=t_r,
            mask_strategy=mask_strategy,
            memory=memory,
            is_classification=False,
            memory_level=memory_level,
            verbose=verbose,
            n_jobs=n_jobs,
        )

    def _more_tags(self):
        """Return estimator tags.

        TODO remove when bumping sklearn_version > 1.5
        """
        return self.__sklearn_tags__()

    def __sklearn_tags__(self):
        """Return estimator tags.

        See the sklearn documentation for more details on tags
        https://scikit-learn.org/1.6/developers/develop.html#estimator-tags
        """
        # TODO
        # get rid of if block
        # bumping sklearn_version > 1.5
        # see https://github.com/scikit-learn/scikit-learn/pull/29677
        tags = super().__sklearn_tags__()
        if SKLEARN_LT_1_6:
            tags["multioutput"] = True
            return tags
        tags.estimator_type = "regressor"
        return tags


@fill_doc
class FREMRegressor(_BaseDecoder):
    """State of the art :term:`decoding` scheme applied \
       to usual regression estimators.

    FREM uses an implicit spatial regularization through fast clustering and
    aggregates a high number of estimators trained on various splits of the
    training set, thus returning a very robust decoder
    at a lower computational cost
    than other spatially regularized methods :footcite:p:`Hoyos-Idrobo2018`.

    Parameters
    ----------
    estimator : str, optional
        The estimator to choose among:
        %(regressor_options)s
        Default 'svr'.

    mask : filename, Nifti1Image, NiftiMasker, or MultiNiftiMasker, optional,
        (default None)
        Mask to be used on data. If an instance of masker is passed,
        then its mask and parameters will be used. If no mask is given, mask
        will be computed automatically from provided images by an inbuilt
        masker with default parameters. Refer to NiftiMasker or
        MultiNiftiMasker to check for default parameters.

    cv : int or cross-validation generator, optional (default 30)
        If int, number of shuffled splits returned, which is usually the right
        way to train many different classifiers. A good trade-off between
        stability of the aggregated model and computation time is 50 splits.
        Shuffled splits are seeded by default for reproducibility.
        Can also be a cross-validation generator.

    param_grid : dict of str to sequence, or sequence of such. Default None
        The parameter grid to explore, as a dictionary mapping estimator
        parameters to sequences of allowed values.

        None or an empty dict signifies default parameters.

        A sequence of dicts signifies a sequence of grids to search, and is
        useful to avoid exploring parameter combinations that make no sense
        or have no effect. See scikit-learn documentation for more information,
        for example: https://scikit-learn.org/stable/modules/grid_search.html

    clustering_percentile : int, float, optional, in closed interval [0, 100]\
        (default 10)
        Used to perform a fast ReNA clustering on input data as a first step of
        fit. It agglomerates similar features together to reduce their number
        by this percentile. ReNA is typically efficient for cluster_percentile
        equal to 10.

    screening_percentile : int, float, optional, in closed interval [0, 100]\
        (default 20)
        The percentage of brain volume that will be kept with respect to a full
        MNI template. In particular, if it is lower than 100, a univariate
        feature selection based on the Anova F-value for the input data will be
        performed. A float according to a percentile of the highest
        scores.

    scoring : str, callable or None, default= 'r2'

        The scoring strategy to use. See the scikit-learn documentation at
        https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules
        If callable, takes as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.
        e.g. scorer(estimator, X_test, y_test)

        For regression, valid entries are: 'r2', 'neg_mean_absolute_error',
        or 'neg_mean_squared_error'.
    %(smoothing_fwhm)s
    %(standardize)s
    %(target_affine)s
    %(target_shape)s
    %(low_pass)s
    %(high_pass)s
    %(t_r)s
    %(mask_strategy)s

        .. note::
            This parameter will be ignored if a mask image is provided.

        .. note::
            Depending on this value, the mask will be computed from
            :func:`nilearn.masking.compute_background_mask`,
            :func:`nilearn.masking.compute_epi_mask`, or
            :func:`nilearn.masking.compute_brain_mask`.

        Default='background'.
    %(memory)s
    %(memory_level)s
    %(n_jobs)s
    %(verbose0)s

    References
    ----------
    .. footbibliography::

    See Also
    --------
    nilearn.decoding.DecoderRegressor: Regression strategies for Neuroimaging,
    nilearn.decoding.FREMClassifier: State of the art classification pipeline
        for Neuroimaging
    """

    def __init__(
        self,
        estimator="svr",
        mask=None,
        cv=30,
        param_grid=None,
        clustering_percentile=10,
        screening_percentile=20,
        scoring="r2",
        smoothing_fwhm=None,
        standardize=True,
        target_affine=None,
        target_shape=None,
        mask_strategy="background",
        low_pass=None,
        high_pass=None,
        t_r=None,
        memory=None,
        memory_level=0,
        n_jobs=1,
        verbose=0,
    ):
        self.classes_ = ["beta"]

        if isinstance(cv, int):
            cv = ShuffleSplit(cv, random_state=0)

        super().__init__(
            estimator=estimator,
            mask=mask,
            cv=cv,
            param_grid=param_grid,
            clustering_percentile=clustering_percentile,
            screening_percentile=screening_percentile,
            scoring=scoring,
            smoothing_fwhm=smoothing_fwhm,
            standardize=standardize,
            target_affine=target_affine,
            target_shape=target_shape,
            low_pass=low_pass,
            high_pass=high_pass,
            t_r=t_r,
            mask_strategy=mask_strategy,
            memory=memory,
            is_classification=False,
            memory_level=memory_level,
            verbose=verbose,
            n_jobs=n_jobs,
        )


@fill_doc
class FREMClassifier(_BaseDecoder):
    """State of the art :term:`decoding` scheme applied to usual classifiers.

    FREM uses an implicit spatial regularization through fast clustering and
    aggregates a high number of estimators trained on various splits of the
    training set, thus returning a very robust decoder
    at a lower computational cost
    than other spatially regularized methods :footcite:p:`Hoyos-Idrobo2018`.

    Parameters
    ----------
    estimator : str, optional, (default 'svc')
        The estimator to choose among:
        %(classifier_options)s


    mask : filename, Nifti1Image, NiftiMasker, or MultiNiftiMasker, optional,\
        (default None)
        Mask to be used on data. If an instance of masker is passed,
        then its mask and parameters will be used. If no mask is given, mask
        will be computed automatically from provided images by an inbuilt
        masker with default parameters. Refer to NiftiMasker or
        MultiNiftiMasker to check for default parameters.

    cv : int or cross-validation generator, optional (default 30)
        If int, number of stratified shuffled splits returned, which is usually
        the right way to train many different classifiers. A good trade-off
        between stability of the aggregated model and computation time is
        50 splits. Shuffled splits are seeded by default for reproducibility.
        Can also be a cross-validation generator.

    param_grid : dict of str to sequence, or sequence of such. Default None
        The parameter grid to explore, as a dictionary mapping estimator
        parameters to sequences of allowed values.

        None or an empty dict signifies default parameters.

        A sequence of dicts signifies a sequence of grids to search, and is
        useful to avoid exploring parameter combinations that make no sense
        or have no effect. See scikit-learn documentation for more information,
        for example: https://scikit-learn.org/stable/modules/grid_search.html

    clustering_percentile : int, float, optional, in closed interval [0, 100]\
        (default 10)
        Used to perform a fast ReNA clustering on input data as a first step of
        fit. It agglomerates similar features together to reduce their number
        down to this percentile. ReNA is typically efficient for
        cluster_percentile equal to 10.

    screening_percentile : int, float, optional, in closed interval [0, 100],\
        (default 20)
        The percentage of brain volume that will be kept with respect to a full
        MNI template. In particular, if it is lower than 100, a univariate
        feature selection based on the Anova F-value for the input data will be
        performed. A float according to a percentile of the highest
        scores.

    scoring : str, callable or None, optional. (default: 'roc_auc')
        The scoring strategy to use. See the scikit-learn documentation at
        https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules
        If callable, takes as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.
        e.g. scorer(estimator, X_test, y_test)

        For classification, valid entries are: 'accuracy', 'f1', 'precision',
        'recall' or 'roc_auc'. (default 'roc_auc').
    %(smoothing_fwhm)s
    %(standardize)s
    %(target_affine)s
    %(target_shape)s
    %(low_pass)s
    %(high_pass)s
    %(t_r)s
    %(mask_strategy)s

        .. note::
            This parameter will be ignored if a mask image is provided.

        .. note::
            Depending on this value, the mask will be computed from
            :func:`nilearn.masking.compute_background_mask`,
            :func:`nilearn.masking.compute_epi_mask`, or
            :func:`nilearn.masking.compute_brain_mask`.

        Default='background'.
    %(memory)s
    %(memory_level)s
    %(n_jobs)s
    %(verbose0)s

    References
    ----------
    .. footbibliography::

    See Also
    --------
    nilearn.decoding.Decoder: Classification strategies for Neuroimaging,
    nilearn.decoding.FREMRegressor: State of the art regression pipeline
        for Neuroimaging

    """

    def __init__(
        self,
        estimator="svc",
        mask=None,
        cv=30,
        param_grid=None,
        clustering_percentile=10,
        screening_percentile=20,
        scoring="roc_auc",
        smoothing_fwhm=None,
        standardize=True,
        target_affine=None,
        target_shape=None,
        mask_strategy="background",
        low_pass=None,
        high_pass=None,
        t_r=None,
        memory=None,
        memory_level=0,
        n_jobs=1,
        verbose=0,
    ):
        if isinstance(cv, int):
            cv = StratifiedShuffleSplit(cv, random_state=0)

        super().__init__(
            estimator=estimator,
            mask=mask,
            cv=cv,
            param_grid=param_grid,
            clustering_percentile=clustering_percentile,
            screening_percentile=screening_percentile,
            scoring=scoring,
            smoothing_fwhm=smoothing_fwhm,
            standardize=standardize,
            target_affine=target_affine,
            target_shape=target_shape,
            mask_strategy=mask_strategy,
            memory=memory,
            is_classification=True,
            memory_level=memory_level,
            verbose=verbose,
            n_jobs=n_jobs,
            low_pass=low_pass,
            high_pass=high_pass,
            t_r=t_r,
        )

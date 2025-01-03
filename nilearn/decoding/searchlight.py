"""The searchlight is a widely used approach for the study \
of the fine-grained patterns of information in fMRI analysis, \
in which multivariate statistical relationships are iteratively tested \
in the neighborhood of each location of a domain.
"""

# Authors : Vincent Michel (vm.michel@gmail.com)
#           Alexandre Gramfort (alexandre.gramfort@inria.fr)
#           Philippe Gervais (philippe.gervais@inria.fr)
#

import time
import warnings

import numpy as np
from joblib import Parallel, cpu_count, delayed
from sklearn import svm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold, cross_val_score

from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn.image import new_img_like
from nilearn.maskers.nifti_spheres_masker import _apply_mask_and_get_affinity

from .. import masking
from .._utils import check_niimg_4d, fill_doc, logger
from ..image.resampling import coord_transform

ESTIMATOR_CATALOG = {"svc": svm.LinearSVC, "svr": svm.SVR}


@fill_doc
def search_light(
    X,
    y,
    estimator,
    A,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=-1,
    verbose=0,
):
    """Compute a search_light.

    Parameters
    ----------
    X : array-like of shape at least 2D
        data to fit.

    y : array-like
        target variable to predict.

    estimator : estimator object implementing 'fit'
        object to use to fit the data

    A : scipy sparse matrix.
        adjacency matrix. Defines for each feature the neigbhoring features
        following a given structure of the data.

    groups : array-like, optional, (default None)
        group label for each sample for cross validation.

    scoring : string or callable, optional
        The scoring strategy to use. See the scikit-learn documentation
        for possible values.
        If callable, it takes as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.

    cv : cross-validation generator, optional
        A cross-validation generator. If None, a 3-fold cross
        validation is used or 3-fold stratified cross-validation
        when y is supplied.

    %(n_jobs_all)s

    %(verbose0)s

    Returns
    -------
    scores : array-like of shape (number of rows in A)
        search_light scores
    """
    group_iter = GroupIterator(A.shape[0], n_jobs)
    with warnings.catch_warnings():  # might not converge
        warnings.simplefilter("ignore", ConvergenceWarning)
        scores = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_group_iter_search_light)(
                A.rows[list_i],
                estimator,
                X,
                y,
                groups,
                scoring,
                cv,
                thread_id + 1,
                A.shape[0],
                verbose,
            )
            for thread_id, list_i in enumerate(group_iter)
        )
    return np.concatenate(scores)


@fill_doc
class GroupIterator:
    """Group iterator.

    Provides group of features for search_light loop
    that may be used with Parallel.

    Parameters
    ----------
    n_features : int
        Total number of features
    %(n_jobs)s

    """

    def __init__(self, n_features, n_jobs=1):
        self.n_features = n_features
        if n_jobs == -1:
            n_jobs = cpu_count()
        self.n_jobs = n_jobs

    def __iter__(self):
        yield from np.array_split(np.arange(self.n_features), self.n_jobs)


def _group_iter_search_light(
    list_rows,
    estimator,
    X,
    y,
    groups,
    scoring,
    cv,
    thread_id,
    total,
    verbose=0,
):
    """Perform grouped iterations of search_light.

    Parameters
    ----------
    list_rows : array of arrays of int
        adjacency rows. For a voxel with index i in X, list_rows[i] is the list
        of neighboring voxels indices (in X).

    estimator : estimator object implementing 'fit'
        object to use to fit the data

    X : array-like of shape at least 2D
        data to fit.

    y : array-like or None
        Target variable to predict. If `y` is provided, it must be
         an array-like object
        with the same length as the number of samples in `X`.
        When `y` is `None`, a dummy
        target is generated internally with half the samples
        labeled as `0` and the other
        half labeled as `1`. This is useful during transformations
        where the model is applied without ground truth labels.

    groups : array-like, optional
        group label for each sample for cross validation.

    scoring : string or callable, optional
        Scoring strategy to use. See the scikit-learn documentation.
        If callable, takes as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.

    cv : cross-validation generator, optional
        A cross-validation generator. If None, a 3-fold cross validation is
        used or 3-fold stratified cross-validation when y is supplied.

    thread_id : int
        process id, used for display.

    total : int
        Total number of voxels, used for display

    verbose : int, default=0
        The verbosity level.

    Returns
    -------
    par_scores : numpy.ndarray
        score for each voxel. dtype: float64.
    """
    par_scores = np.zeros(len(list_rows))
    t0 = time.time()
    for i, row in enumerate(list_rows):
        kwargs = {"scoring": scoring, "groups": groups}
        if isinstance(cv, KFold):
            kwargs = {"scoring": scoring}

        if y is None:
            y_dummy = np.array(
                [0] * (X.shape[0] // 2) + [1] * (X.shape[0] // 2)
            )
            estimator.fit(
                X[:, row], y_dummy[: X.shape[0]]
            )  # Ensure the size matches X
            par_scores[i] = np.mean(estimator.decision_function(X[:, row]))
        else:
            par_scores[i] = np.mean(
                cross_val_score(
                    estimator, X[:, row], y, cv=cv, n_jobs=1, **kwargs
                )
            )

        if verbose > 0:
            # One can't print less than each 10 iterations
            step = 11 - min(verbose, 10)
            if i % step == 0:
                # If there is only one job, progress information is fixed
                crlf = "\r" if total == len(list_rows) else "\n"
                percent = float(i) / len(list_rows)
                percent = round(percent * 100, 2)
                dt = time.time() - t0
                # We use a max to avoid a division by zero
                remaining = (100.0 - percent) / max(0.01, percent) * dt
                logger.log(
                    f"Job #{thread_id}, processed {i}/{len(list_rows)} steps "
                    f"({percent:0.2f}%, "
                    f"{remaining:0.1f} seconds remaining){crlf}",
                    stack_level=6,
                )
    return par_scores


##############################################################################
# Class for search_light #####################################################
##############################################################################
@fill_doc
class SearchLight(TransformerMixin, BaseEstimator):
    """Implement search_light analysis using an arbitrary type of classifier.

    Parameters
    ----------
    mask_img : Niimg-like object
        See :ref:`extracting_data`.
        Boolean image giving location of voxels containing usable signals.

    process_mask_img : Niimg-like object, optional
        See :ref:`extracting_data`.
        Boolean image giving voxels on which searchlight should be
        computed.

    radius : float, default=2.
        radius of the searchlight ball, in millimeters.

    estimator : 'svr', 'svc', or an estimator object implementing 'fit'
        The object to use to fit the data
    %(n_jobs)s
    scoring : string or callable, optional
        The scoring strategy to use. See the scikit-learn documentation
        If callable, takes as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.

    cv : cross-validation generator, optional
        A cross-validation generator. If None, a 3-fold cross
        validation is used or 3-fold stratified cross-validation
        when y is supplied.
    %(verbose0)s

    Attributes
    ----------
    scores_ : numpy.ndarray
        3D array containing searchlight scores for each voxel, aligned
         with the mask.

         .. versionadded:: 0.11.0

    process_mask_ : numpy.ndarray
        Boolean mask array representing the voxels included in the
         searchlight computation.

         .. versionadded:: 0.11.0

    masked_scores_ : numpy.ndarray
        1D array containing the searchlight scores corresponding
        to the masked region only.

        .. versionadded:: 0.11.0

    Notes
    -----
    The searchlight [Kriegeskorte 06] is a widely used approach for the
    study of the fine-grained patterns of information in fMRI analysis.
    Its principle is relatively simple: a small group of neighboring
    features is extracted from the data, and the prediction function is
    instantiated on these features only. The resulting prediction
    accuracy is thus associated with all the features within the group,
    or only with the feature on the center. This yields a map of local
    fine-grained information, that can be used for assessing hypothesis
    on the local spatial layout of the neural code under investigation.

    Nikolaus Kriegeskorte, Rainer Goebel & Peter Bandettini.
    Information-based functional brain mapping.
    Proceedings of the National Academy of Sciences
    of the United States of America,
    vol. 103, no. 10, pages 3863-3868, March 2006
    """

    def __init__(
        self,
        mask_img,
        process_mask_img=None,
        radius=2.0,
        estimator="svc",
        n_jobs=1,
        scoring=None,
        cv=None,
        verbose=0,
    ):
        self.mask_img = mask_img
        self.process_mask_img = process_mask_img
        self.radius = radius
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.cv = cv
        self.verbose = verbose

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

        if SKLEARN_LT_1_6:
            from nilearn._utils.tags import tags

            return tags()

        from nilearn._utils.tags import InputTags

        tags = super().__sklearn_tags__()
        tags.input_tags = InputTags()
        return tags

    def fit(self, imgs, y, groups=None):
        """Fit the searchlight.

        Parameters
        ----------
        imgs : Niimg-like object
            See :ref:`extracting_data`.
            4D image.

        y : 1D array-like
            Target variable to predict. Must have exactly as many elements as
            3D images in img.

        groups : array-like, optional
            group label for each sample for cross validation. Must have
            exactly as many elements as 3D images in img. default None
        """
        # check if image is 4D
        imgs = check_niimg_4d(imgs)

        # Get the seeds
        process_mask_img = self.process_mask_img or self.mask_img

        # Compute world coordinates of the seeds
        process_mask, process_mask_affine = masking.load_mask_img(
            process_mask_img
        )

        self.process_mask_ = process_mask
        process_mask_coords = np.where(process_mask != 0)
        process_mask_coords = coord_transform(
            process_mask_coords[0],
            process_mask_coords[1],
            process_mask_coords[2],
            process_mask_affine,
        )
        process_mask_coords = np.asarray(process_mask_coords).T

        X, A = _apply_mask_and_get_affinity(
            process_mask_coords,
            imgs,
            self.radius,
            True,
            mask_img=self.mask_img,
        )

        estimator = self.estimator
        if estimator == "svc":
            estimator = ESTIMATOR_CATALOG[estimator](dual=True)
        elif isinstance(estimator, str):
            estimator = ESTIMATOR_CATALOG[estimator]()

        scores = search_light(
            X,
            y,
            estimator,
            A,
            groups,
            self.scoring,
            self.cv,
            self.n_jobs,
            self.verbose,
        )
        self.masked_scores_ = scores
        self.scores_ = np.zeros(process_mask.shape)
        self.scores_[np.where(process_mask)] = scores
        return self

    def __sklearn_is_fitted__(self):
        return (
            hasattr(self, "scores_")
            and hasattr(self, "process_mask_")
            and self.scores_ is not None
            and self.process_mask_ is not None
        )

    def _check_fitted(self):
        if not self.__sklearn_is_fitted__():
            raise ValueError("The model has not been fitted yet.")

    @property
    def scores_img_(self):
        """Convert the 3D scores array into a NIfTI image."""
        self._check_fitted()
        return new_img_like(self.mask_img, self.scores_)

    def transform(self, imgs):
        """Apply the fitted searchlight on new images."""
        self._check_fitted()

        imgs = check_niimg_4d(imgs)

        X, A = _apply_mask_and_get_affinity(
            np.asarray(np.where(self.process_mask_)).T,
            imgs,
            self.radius,
            True,
            mask_img=self.mask_img,
        )

        estimator = self.estimator
        if estimator == "svc":
            estimator = ESTIMATOR_CATALOG[estimator](dual=True)

        # Use the modified `_group_iter_search_light` logic to avoid `y` issues
        result = search_light(
            X,
            None,
            estimator,
            A,
            None,
            self.scoring,
            self.cv,
            self.n_jobs,
            self.verbose,
        )

        reshaped_result = np.zeros(self.process_mask_.shape)
        reshaped_result[np.where(self.process_mask_)] = result
        reshaped_result = np.abs(reshaped_result)

        return reshaped_result

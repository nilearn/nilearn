"""Contains the GLM and contrast classes that are meant to be the main \
objects of fMRI data analyses.

Author: Bertrand Thirion, Martin Perez-Guevara, 2016

"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed
from nibabel import Nifti1Image
from sklearn.base import clone
from sklearn.cluster import KMeans

from nilearn._utils import fill_doc, logger, stringify_path
from nilearn._utils.glm import check_and_load_tables
from nilearn._utils.masker_validation import (
    check_compatibility_mask_and_images,
)
from nilearn._utils.niimg_conversions import check_niimg
from nilearn._utils.param_validation import check_run_sample_masks
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn.datasets import load_fsaverage
from nilearn.glm._base import BaseGLM
from nilearn.glm.contrasts import (
    compute_fixed_effect_contrast,
    expression_to_contrast_vector,
)
from nilearn.glm.first_level.design_matrix import (
    make_first_level_design_matrix,
)
from nilearn.glm.regression import (
    ARModel,
    OLSModel,
    RegressionResults,
    SimpleRegressionResults,
)
from nilearn.image import get_data
from nilearn.interfaces.bids import get_bids_files, parse_bids_filename
from nilearn.interfaces.bids.query import (
    infer_repetition_time_from_dataset,
    infer_slice_timing_start_time_from_dataset,
)
from nilearn.interfaces.bids.utils import bids_entities, check_bids_label
from nilearn.interfaces.fmriprep.load_confounds import load_confounds
from nilearn.maskers import SurfaceMasker
from nilearn.surface import SurfaceImage


def mean_scaling(Y, axis=0):
    """Scaling of the data to have percent of baseline change \
    along the specified axis.

    Parameters
    ----------
    Y : array of shape (n_time_points, n_voxels)
       The input data.

    axis : :obj:`int`, default=0
        Axis along which the scaling mean should be calculated.

    Returns
    -------
    Y : array of shape (n_time_points, n_voxels),
       The data after mean-scaling, de-meaning and multiplication by 100.

    mean : array of shape (n_voxels,)
        The data mean.

    """
    mean = Y.mean(axis=axis)
    if (mean == 0).any():
        warn(
            "Mean values of 0 observed. "
            "The data have probably been centered."
            "Scaling might not work as expected",
            UserWarning,
            stacklevel=2,
        )
    mean = np.maximum(mean, 1)
    Y = 100 * (Y / mean - 1)
    return Y, mean


def _ar_model_fit(X, val, Y):
    """Wrap fit method of ARModel to allow joblib parallelization."""
    return ARModel(X, val).fit(Y)


def _yule_walker(x, order):
    """Compute Yule-Walker (adapted from MNE and statsmodels).

    Operates along the last axis of x.
    """
    from scipy.linalg import toeplitz

    if order < 1:
        raise ValueError("AR order must be positive")
    if type(order) is not int:
        raise TypeError("AR order must be an integer")
    if x.ndim < 1:
        raise TypeError("Input data must have at least 1 dimension")

    denom = x.shape[-1] - np.arange(order + 1)
    n = np.prod(np.array(x.shape[:-1], int))
    r = np.zeros((n, order + 1), np.float64)
    y = x - x.mean()
    y.shape = (n, x.shape[-1])  # inplace
    r[:, 0] += (y[:, np.newaxis, :] @ y[:, :, np.newaxis])[:, 0, 0]
    for k in range(1, order + 1):
        r[:, k] += (y[:, np.newaxis, 0:-k] @ y[:, k:, np.newaxis])[:, 0, 0]
    r /= denom * x.shape[-1]
    rt = np.array([toeplitz(rr[:-1]) for rr in r], np.float64)

    # extra dimension added to r for compatibility with numpy <2 and >2
    # see https://numpy.org/devdocs/release/2.0.0-notes.html
    # section removed-ambiguity-when-broadcasting-in-np-solve
    rho = np.linalg.solve(rt, r[:, 1:, None])[..., 0]

    rho.shape = x.shape[:-1] + (order,)
    return rho


def run_glm(
    Y, X, noise_model="ar1", bins=100, n_jobs=1, verbose=0, random_state=None
):
    """:term:`GLM` fit for an :term:`fMRI` data matrix.

    Parameters
    ----------
    Y : array of shape (n_time_points, n_voxels)
        The :term:`fMRI` data.

    X : array of shape (n_time_points, n_regressors)
        The design matrix.

    noise_model : {'ar(N)', 'ols'}, default='ar1'
        The temporal variance model.
        To specify the order of an autoregressive model place the
        order after the characters `ar`, for example to specify a third order
        model use `ar3`.

    bins : :obj:`int`, default=100
        Maximum number of discrete bins for the AR coef histogram.
        If an autoregressive model with order greater than one is specified
        then adaptive quantification is performed and the coefficients
        will be clustered via K-means with `bins` number of clusters.

    n_jobs : :obj:`int`, default=1
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    verbose : :obj:`int`, default=0
        The verbosity level.

    random_state : :obj:`int` or numpy.random.RandomState, default=None
        Random state seed to sklearn.cluster.KMeans for autoregressive models
        of order at least 2 ('ar(N)' with n >= 2).

        .. versionadded:: 0.9.1

    Returns
    -------
    labels : array of shape (n_voxels,),
        A map of values on voxels used to identify the corresponding model.

    results : :obj:`dict`,
        Keys correspond to the different labels values
        values are RegressionResults instances corresponding to the voxels.

    """
    acceptable_noise_models = ["ols", "arN"]
    if (noise_model[:2] != "ar") and (noise_model != "ols"):
        raise ValueError(
            f"Acceptable noise models are {acceptable_noise_models}. "
            f"You provided 'noise_model={noise_model}'."
        )
    if Y.shape[0] != X.shape[0]:
        raise ValueError(
            "The number of rows of Y "
            "should match the number of rows of X.\n"
            f"You provided X with shape {X.shape} "
            f"and Y with shape {Y.shape}."
        )

    # Create the model
    ols_result = OLSModel(X).fit(Y)

    if noise_model[:2] == "ar":
        err_msg = (
            "AR order must be a positive integer specified as arN, "
            "where N is an integer. E.g. ar3. "
            f"You provided {noise_model}."
        )
        try:
            ar_order = int(noise_model[2:])
        except ValueError:
            raise ValueError(err_msg)

        # compute the AR coefficients
        ar_coef_ = _yule_walker(ols_result.residuals.T, ar_order)
        del ols_result
        if len(ar_coef_[0]) == 1:
            ar_coef_ = ar_coef_[:, 0]

        # Either bin the AR1 coefs or cluster ARN coefs
        if ar_order == 1:
            for idx in range(len(ar_coef_)):
                ar_coef_[idx] = (ar_coef_[idx] * bins).astype(int) * 1.0 / bins
            labels = np.array([str(val) for val in ar_coef_])
        else:  # AR(N>1) case
            n_clusters = np.min([bins, Y.shape[1]])
            kmeans = KMeans(
                n_clusters=n_clusters, n_init=10, random_state=random_state
            ).fit(ar_coef_)
            ar_coef_ = kmeans.cluster_centers_[kmeans.labels_]

            # Create a set of rounded values for the labels with _ between
            # each coefficient
            cluster_labels = kmeans.cluster_centers_.copy()
            cluster_labels = np.array(
                ["_".join(map(str, np.round(a, 2))) for a in cluster_labels]
            )
            # Create labels and coef per voxel
            labels = np.array([cluster_labels[i] for i in kmeans.labels_])

        unique_labels = np.unique(labels)
        results = {}

        # Fit the AR model according to current AR(N) estimates
        ar_result = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_ar_model_fit)(
                X, ar_coef_[labels == val][0], Y[:, labels == val]
            )
            for val in unique_labels
        )

        # Converting the key to a string is required for AR(N>1) cases
        results = dict(zip(unique_labels, ar_result))
        del unique_labels
        del ar_result

    else:
        labels = np.zeros(Y.shape[1])
        results = {0.0: ols_result}

    return labels, results


def _check_trial_type(events):
    """Check that the event files contain a "trial_type" column.

    Parameters
    ----------
    events : :obj:`list` of :obj:`str` or :obj:`pathlib.Path``.
              A list of paths of events.tsv files.

    """
    file_names = []

    for event_ in events:
        events_df = pd.read_csv(event_, sep="\t")
        if "trial_type" not in events_df.columns:
            file_names.append(Path(event_).name)

    if file_names:
        file_names = "\n -".join(file_names)
        warn(
            f"No column named 'trial_type' found in:{file_names}.\n "
            "All rows in those files will be treated "
            "as if they are instances of same experimental condition.\n"
            "If there is a column in the dataframe "
            "corresponding to trial information, "
            "consider renaming it to 'trial_type'."
        )


@fill_doc
class FirstLevelModel(BaseGLM):
    """Implement the General Linear Model for single run :term:`fMRI` data.

    Parameters
    ----------
    t_r : :obj:`float` or None, default=None
        This parameter indicates :term:`repetition times<TR>`
        of the experimental runs.
        In seconds. It is necessary to correctly consider times in the design
        matrix. This parameter is also passed to :func:`nilearn.signal.clean`.
        Please see the related documentation for details.

    slice_time_ref : :obj:`float`, default=0.0
        This parameter indicates the time of the reference slice used in the
        slice timing preprocessing step of the experimental runs.
        It is expressed as a fraction of the ``t_r`` (repetition time),
        so it can have values between 0. and 1.
    %(hrf_model)s
        Default='glover'.
    drift_model : :obj:`str`, default='cosine'
        This parameter specifies the desired drift model for the design
        matrices. It can be 'polynomial', 'cosine' or None.

    high_pass : :obj:`float`, default=0.01
        This parameter specifies the cut frequency of the high-pass filter in
        Hz for the design matrices. Used only if drift_model is 'cosine'.

    drift_order : :obj:`int`, default=1
        This parameter specifies the order of the drift model (in case it is
        polynomial) for the design matrices.

    fir_delays : array of shape(n_onsets), :obj:`list` or None, default=None
        Will be set to ``[0]`` if ``None`` is passed.
        In case of :term:`FIR` design,
        yields the array of delays used in the :term:`FIR` model,
        in scans.

    min_onset : :obj:`float`, default=-24
        This parameter specifies the minimal onset relative to the design
        (in seconds). Events that start before (slice_time_ref * t_r +
        min_onset) are not considered.

    mask_img : Niimg-like, NiftiMasker, :obj:`~nilearn.surface.SurfaceImage`,\
             :obj:`~nilearn.maskers.SurfaceMasker`, False or \
             None, default=None
        Mask to be used on data.
        If an instance of masker is passed, then its mask will be used.
        If None is passed, the mask will be computed automatically
        by a NiftiMasker
        or :obj:`~nilearn.maskers.SurfaceMasker` with default parameters.
        If False is given then the data will not be masked.
        In the case of surface analysis, passing None or False will lead to
        no masking.

    target_affine : 3x3 or 4x4 matrix, or None, default=None
        This parameter is passed to nilearn.image.resample_img.
        Please see the related documentation for details.

    target_shape : 3-tuple of :obj:`int`, or None, default=None
        This parameter is passed to nilearn.image.resample_img.
        Please see the related documentation for details.
    %(smoothing_fwhm)s
    memory : :obj:`str` or pathlib.Path, default=None
        Path to the directory used to cache the masking process
        and the glm fit.
        By default, no caching is done.
        Creates instance of joblib.Memory.
        If ``None`` is passed will default to ``Memory(location=None)``.

    memory_level : :obj:`int` or None, default=None
        Rough estimator of the amount of memory used by caching.
        Higher value means more memory for caching.


    standardize : :obj:`bool`, default=False
        If standardize is True, the time-series are centered and normed:
        their variance is put to 1 in the time dimension.

    signal_scaling : False, :obj:`int` or (int, int), default=0
        If not False, fMRI signals are
        scaled to the mean value of scaling_axis given,
        which can be 0, 1 or (0, 1).
        0 refers to mean scaling each voxel with respect to time,
        1 refers to mean scaling each time point with respect to all voxels &
        (0, 1) refers to scaling with respect to voxels and time,
        which is known as grand mean scaling.
        Incompatible with standardize (standardize=False is enforced when
        signal_scaling is not False).

    noise_model : {'ar1', 'ols'}, default='ar1'
        The temporal variance model.

    verbose : :obj:`int`, default=0
        Indicate the level of verbosity. By default, nothing is printed.
        If 0 prints nothing. If 1 prints progress by computation of
        each run. If 2 prints timing details of masker and GLM. If 3
        prints masker computation details.

    n_jobs : :obj:`int`, default=1
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on.

    minimize_memory : :obj:`bool`, default=True
        Gets rid of some variables on the model fit results that are not
        necessary for contrast computation and would only be useful for
        further inspection of model details. This has an important impact
        on memory consumption.

    subject_label : :obj:`str`, optional
        This id will be used to identify a `FirstLevelModel` when passed to
        a `SecondLevelModel` object.

    random_state : :obj:`int` or numpy.random.RandomState, default=None.
        Random state seed to sklearn.cluster.KMeans
        for autoregressive models
        of order at least 2 ('ar(N)' with n >= 2).

        .. versionadded:: 0.9.1

    Attributes
    ----------
    labels_ : array of shape (n_voxels,),
        a map of values on voxels used to identify the corresponding model

    results_ : :obj:`dict`,
        with keys corresponding to the different labels values.
        Values are SimpleRegressionResults corresponding to the voxels,
        if minimize_memory is True,
        RegressionResults if minimize_memory is False

    """

    def __init__(
        self,
        t_r=None,
        slice_time_ref=0.0,
        hrf_model="glover",
        drift_model="cosine",
        high_pass=0.01,
        drift_order=1,
        fir_delays=None,
        min_onset=-24,
        mask_img=None,
        target_affine=None,
        target_shape=None,
        smoothing_fwhm=None,
        memory=None,
        memory_level=1,
        standardize=False,
        signal_scaling=0,
        noise_model="ar1",
        verbose=0,
        n_jobs=1,
        minimize_memory=True,
        subject_label=None,
        random_state=None,
    ):
        # design matrix parameters
        self.t_r = t_r
        self.slice_time_ref = slice_time_ref
        self.hrf_model = hrf_model
        self.drift_model = drift_model
        self.high_pass = high_pass
        self.drift_order = drift_order
        self.fir_delays = fir_delays
        self.min_onset = min_onset

        # glm parameters
        self.mask_img = mask_img
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.smoothing_fwhm = smoothing_fwhm
        self.memory = memory
        self.memory_level = memory_level
        self.standardize = standardize
        self.signal_scaling = signal_scaling

        self.noise_model = noise_model
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.minimize_memory = minimize_memory

        # attributes
        self.subject_label = subject_label
        self.random_state = random_state

    def _check_fit_inputs(
        self,
        run_imgs,
        events,
        confounds,
        sample_masks,
        design_matrices,
    ):
        """Run input validation and ensure inputs are compatible."""
        # Raise a warning if both design_matrices and confounds are provided
        if design_matrices is not None and (
            confounds is not None or events is not None
        ):
            warn(
                "If design matrices are supplied, "
                "confounds and events will be ignored."
            )

        if events is not None:
            _check_events_file_uses_tab_separators(events_files=events)

        if not isinstance(run_imgs, (list, tuple)):
            run_imgs = [run_imgs]

        check_compatibility_mask_and_images(self.mask_img, run_imgs)

        if design_matrices is None:
            if events is None:
                raise ValueError("events or design matrices must be provided")
            if self.t_r is None:
                raise ValueError(
                    "t_r not given to FirstLevelModel object"
                    " to compute design from events"
                )
        else:
            design_matrices = _check_run_tables(
                run_imgs, design_matrices, "design_matrices"
            )

        # Check that number of events and confound files match number of runs
        # Also check that events and confound files can be loaded as DataFrame
        if events is not None:
            events = _check_run_tables(run_imgs, events, "events")

        if confounds is not None:
            confounds = _check_run_tables(run_imgs, confounds, "confounds")

        if sample_masks is not None:
            sample_masks = check_run_sample_masks(len(run_imgs), sample_masks)

        return (
            run_imgs,
            events,
            confounds,
            sample_masks,
            design_matrices,
        )

    def _log(
        self, step, run_idx=None, n_runs=None, t0=None, time_in_second=None
    ):
        """Generate and log messages for different step of the model fit."""
        if step == "progress":
            msg = self._report_progress(run_idx, n_runs, t0)
        elif step == "running":
            msg = "Performing GLM computation."
        elif step == "run_done":
            msg = f"GLM took {int(time_in_second)} seconds."
        elif step == "masking":
            msg = "Performing mask computation."
        elif step == "masking_done":
            msg = f"Masking took {int(time_in_second)} seconds."
        elif step == "done":
            msg = (
                f"Computation of {n_runs} runs done "
                f"in {int(time_in_second)} seconds."
            )

        logger.log(msg, verbose=self.verbose, stack_level=2)

    def _report_progress(self, run_idx, n_runs, t0):
        remaining = "go take a coffee, a big one"
        if run_idx != 0:
            percent = float(run_idx) / n_runs
            percent = round(percent * 100, 2)
            dt = time.time() - t0
            # We use a max to avoid a division by zero
            remaining = (100.0 - percent) / max(0.01, percent) * dt
            remaining = f"{int(remaining)} seconds remaining"

        return (
            f"Computing run {run_idx + 1} "
            f"out of {n_runs} runs ({remaining})."
        )

    def _fit_single_run(self, sample_masks, bins, run_img, run_idx):
        """Fit the model for a single and keep only the regression results."""
        design = self.design_matrices_[run_idx]

        sample_mask = None
        if sample_masks is not None:
            sample_mask = sample_masks[run_idx]
            design = design.iloc[sample_mask, :]
            self.design_matrices_[run_idx] = design

        # Mask and prepare data for GLM
        self._log("masking")
        t_masking = time.time()
        Y = self.masker_.transform(run_img, sample_mask=sample_mask)
        del run_img  # Delete unmasked image to save memory
        self._log("masking_done", time_in_second=time.time() - t_masking)

        if self.signal_scaling is not False:
            Y, _ = mean_scaling(Y, self.signal_scaling)

        if self.memory:
            mem_glm = self.memory.cache(run_glm, ignore=["n_jobs"])
        else:
            mem_glm = run_glm

        # compute GLM
        t_glm = time.time()
        self._log("running")

        labels, results = mem_glm(
            Y,
            design.values,
            noise_model=self.noise_model,
            bins=bins,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

        self._log("run_done", time_in_second=time.time() - t_glm)

        self.labels_.append(labels)

        # We save memory if inspecting model details is not necessary
        if self.minimize_memory:
            results = {
                k: SimpleRegressionResults(v) for k, v in results.items()
            }
        self.results_.append(results)
        del Y

    def _create_all_designs(
        self, run_imgs, events, confounds, design_matrices
    ):
        """Build experimental design of all runs."""
        if design_matrices is not None:
            return design_matrices

        design_matrices = []

        for run_idx, run_img in enumerate(run_imgs):
            if isinstance(run_img, SurfaceImage):
                n_scans = run_img.shape[1]
            else:
                run_img = check_niimg(run_img, ensure_ndim=4)
                n_scans = get_data(run_img).shape[3]

            design = self._create_single_design(
                n_scans, events, confounds, run_idx
            )

            design_matrices.append(design)

        return design_matrices

    def _create_single_design(self, n_scans, events, confounds, run_idx):
        """Build experimental design of a single run.

        Parameters
        ----------
        n_scans: int

        events : list of pandas.DataFrame

        confounds : list of pandas.DataFrame or numpy.arrays

        run_idx : int
        """
        confounds_matrix = None
        confounds_names = None
        if confounds is not None:
            confounds_matrix = confounds[run_idx]

            if isinstance(confounds_matrix, pd.DataFrame):
                confounds_names = confounds[run_idx].columns.tolist()
                confounds_matrix = confounds_matrix.to_numpy()
            else:
                # create dummy names when dealing with numpy arrays
                confounds_names = [
                    f"confound_{i}" for i in range(confounds_matrix.shape[1])
                ]

            if confounds_matrix.shape[0] != n_scans:
                raise ValueError(
                    "Rows in confounds does not match "
                    "n_scans in run_img "
                    f"at index {run_idx}."
                )

        start_time = self.slice_time_ref * self.t_r
        end_time = (n_scans - 1 + self.slice_time_ref) * self.t_r
        frame_times = np.linspace(start_time, end_time, n_scans)
        design = make_first_level_design_matrix(
            frame_times,
            events[run_idx],
            self.hrf_model,
            self.drift_model,
            self.high_pass,
            self.drift_order,
            self.fir_delays,
            confounds_matrix,
            confounds_names,
            self.min_onset,
        )

        return design

    def __sklearn_is_fitted__(self):
        return (
            hasattr(self, "labels_")
            and hasattr(self, "results_")
            and self.labels_ is not None
            and self.results_ is not None
        )

    def _check_fitted(self):
        if not self.__sklearn_is_fitted__():
            raise ValueError("The model has not been fit yet.")

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

            return tags(niimg_like=True, surf_img=True)

        from nilearn._utils.tags import InputTags

        tags = super().__sklearn_tags__()
        tags.input_tags = InputTags(niimg_like=True, surf_img=True)
        return tags

    def fit(
        self,
        run_imgs,
        events=None,
        confounds=None,
        sample_masks=None,
        design_matrices=None,
        bins=100,
    ):
        """Fit the :term:`GLM`.

        For each run:
        1. create design matrix X
        2. do a masker job: fMRI_data -> Y
        3. fit regression to (Y, X)

        Parameters
        ----------
        run_imgs : Niimg-like object, \
                   :obj:`list` or :obj:`tuple` of Niimg-like objects, \
                   SurfaceImage object, \
                   or :obj:`list` or \
                   :obj:`tuple` of :obj:`~nilearn.surface.SurfaceImage`
            Data on which the :term:`GLM` will be fitted.
            If this is a list, the affine is considered the same for all.

            .. warning::

                If the FirstLevelModel object was instantiated
                with a ``mask_img``,
                then ``run_imgs`` must be compatible with ``mask_img``.
                For example, if ``mask_img`` is
                a :class:`nilearn.maskers.NiftiMasker` instance
                or a Niimng-like object, then ``run_imgs`` must be a
                Niimg-like object, \
                a :obj:`list` or a :obj:`tuple` of Niimg-like objects.
                If ``mask_img`` is
                a :obj:`~nilearn.maskers.SurfaceMasker`
                or :obj:`~nilearn.surface.SurfaceImage` instance,
                then ``run_imgs`` must be a
                :obj:`~nilearn.surface.SurfaceImage`, \
                a :obj:`list` or \
                a :obj:`tuple` of :obj:`~nilearn.surface.SurfaceImage`.

        events : :obj:`pandas.DataFrame` or :obj:`str` or \
                 :obj:`pathlib.Path` to a TSV file, or \
                 :obj:`list` of \
                 :obj:`pandas.DataFrame`, :obj:`str` or \
                 :obj:`pathlib.Path` to a TSV file, \
                 or None, default=None
            :term:`fMRI` events used to build design matrices.
            One events object expected per run_img.
            Ignored in case designs is not None.
            If string, then a path to a csv or tsv file is expected.
            See :func:`~nilearn.glm.first_level.make_first_level_design_matrix`
            for details on the required content of events files.

        confounds : :class:`pandas.DataFrame`, :class:`numpy.ndarray` or \
                    :obj:`str` or :obj:`list` of :class:`pandas.DataFrame`, \
                    :class:`numpy.ndarray` or :obj:`str`, default=None
            Each column in a DataFrame corresponds to a confound variable
            to be included in the regression model of the respective run_img.
            The number of rows must match the number of volumes in the
            respective run_img.
            Ignored in case designs is not None.
            If string, then a path to a csv file is expected.

        sample_masks : array_like, or :obj:`list` of array_like, default=None
            shape of array: (number of scans - number of volumes remove)
            Indices of retained volumes. Masks the niimgs along time/fourth
            dimension to perform scrubbing (remove volumes with high motion)
            and/or remove non-steady-state volumes.

            .. versionadded:: 0.9.2

        design_matrices : :obj:`pandas.DataFrame` or :obj:`str` or \
                          :obj:`pathlib.Path` to a CSV or TSV file, or \
                          :obj:`list` of \
                          :obj:`pandas.DataFrame`, :obj:`str` or \
                          :obj:`pathlib.Path` to a CSV or TSV file, \
                          or None, default=None
            Design matrices that will be used to fit the GLM.
            If given it takes precedence over events and confounds.

        bins : :obj:`int`, default=100
            Maximum number of discrete bins for the AR coef histogram.
            If an autoregressive model with order greater than one is specified
            then adaptive quantification is performed and the coefficients
            will be clustered via K-means with `bins` number of clusters.

        """
        #  check attributes passed at construction
        if self.t_r is not None:
            _check_repetition_time(self.t_r)

        if self.slice_time_ref is not None:
            _check_slice_time_ref(self.slice_time_ref)

        if self.fir_delays is None:
            self.fir_delays = [0]

        self.memory = stringify_path(self.memory)
        if self.memory is None:
            self.memory = Memory(None)
        if isinstance(self.memory, str):
            self.memory = Memory(self.memory)

        if self.signal_scaling not in {False, 1, (0, 1)}:
            raise ValueError(
                'signal_scaling must be "False", "0", "1" or "(0, 1)"'
            )
        if self.signal_scaling in [0, 1, (0, 1)]:
            self.standardize = False

        if not isinstance(
            run_imgs, (str, Path, Nifti1Image, SurfaceImage, list, tuple)
        ) or (
            isinstance(run_imgs, (list, tuple))
            and not all(
                isinstance(x, (str, Path, Nifti1Image, SurfaceImage))
                for x in run_imgs
            )
        ):
            input_type = type(run_imgs)
            if isinstance(run_imgs, list):
                input_type = [type(x) for x in run_imgs]
            raise TypeError(
                "'run_imgs' must be a single instance / a list "
                "of any of the following:\n"
                "- string\n"
                "- pathlib.Path\n"
                "- NiftiImage\n"
                "- SurfaceImage\n"
                f"Got: {input_type}"
            )

        self.labels_ = None
        self.results_ = None

        run_imgs, events, confounds, sample_masks, design_matrices = (
            self._check_fit_inputs(
                run_imgs,
                events,
                confounds,
                sample_masks,
                design_matrices,
            )
        )

        # Initialize masker_ to None such that attribute exists
        self.masker_ = None

        self._prepare_mask(run_imgs[0])

        self.design_matrices_ = self._create_all_designs(
            run_imgs, events, confounds, design_matrices
        )

        # For each run fit the model and keep only the regression results.
        self.labels_, self.results_ = [], []
        n_runs = len(run_imgs)
        t0 = time.time()
        for run_idx, run_img in enumerate(run_imgs):
            self._log("progress", run_idx=run_idx, n_runs=n_runs, t0=t0)
            self._fit_single_run(sample_masks, bins, run_img, run_idx)

        self._log("done", n_runs=n_runs, time_in_second=time.time() - t0)

        return self

    def compute_contrast(
        self,
        contrast_def,
        stat_type=None,
        output_type="z_score",
    ):
        """Generate different outputs corresponding to \
        the contrasts provided e.g. z_map, t_map, effects and variance.

        In multi-run case, outputs the fixed effects map.

        Parameters
        ----------
        contrast_def : str or array of shape (n_col) or list of (string or\
                       array of shape (n_col))

            where ``n_col`` is the number of columns of the design matrix,
            (one array per run). If only one array is provided when there
            are several runs, it will be assumed that
            the same :term:`contrast` is
            desired for all runs. One can use the name of the conditions as
            they appear in the design matrix of the fitted model combined with
            operators +- and combined with numbers with operators +-`*`/. In
            this case, the string defining the contrasts must be a valid
            expression for compatibility with :meth:`pandas.DataFrame.eval`.

        stat_type : {'t', 'F'}, optional
            Type of the contrast.

        output_type : :obj:`str`, default='z_score'
            Type of the output map. Can be 'z_score', 'stat', 'p_value',
            :term:`'effect_size'<Parameter Estimate>`, 'effect_variance' or
            'all'.

        Returns
        -------
        output : Nifti1Image, :obj:`~nilearn.surface.SurfaceImage`, \
                 or :obj:`dict`
            The desired output image(s).
            If ``output_type == 'all'``,
            then the output is a dictionary of images,
            keyed by the type of image.

        """
        self._check_fitted()

        if isinstance(contrast_def, (np.ndarray, str)):
            con_vals = [contrast_def]
        elif isinstance(contrast_def, (list, tuple)):
            con_vals = contrast_def
        else:
            raise ValueError(
                "contrast_def must be an array or str or list of"
                " (array or str)."
            )

        n_runs = len(self.labels_)
        n_contrasts = len(con_vals)
        if n_contrasts == 1 and n_runs > 1:
            warn(
                f"One contrast given, assuming it for all {n_runs} runs",
                category=UserWarning,
                stacklevel=2,
            )
            con_vals = con_vals * n_runs
        elif n_contrasts != n_runs:
            raise ValueError(
                f"{n_contrasts} contrasts given, "
                f"while there are {n_runs} runs."
            )

        # Translate formulas to vectors
        for cidx, (con, design_mat) in enumerate(
            zip(con_vals, self.design_matrices_)
        ):
            design_columns = design_mat.columns.tolist()
            if isinstance(con, str):
                con_vals[cidx] = expression_to_contrast_vector(
                    con, design_columns
                )

        valid_types = [
            "z_score",
            "stat",
            "p_value",
            "effect_size",
            "effect_variance",
            "all",  # must be the final entry!
        ]
        if output_type not in valid_types:
            raise ValueError(f"output_type must be one of {valid_types}")
        contrast = compute_fixed_effect_contrast(
            self.labels_, self.results_, con_vals, stat_type
        )
        output_types = (
            valid_types[:-1] if output_type == "all" else [output_type]
        )
        outputs = {}
        for output_type_ in output_types:
            estimate_ = getattr(contrast, output_type_)()
            # Prepare the returned images
            output = self.masker_.inverse_transform(estimate_)
            contrast_name = str(con_vals)
            if not isinstance(output, SurfaceImage):
                output.header["descrip"] = (
                    f"{output_type_} of contrast {contrast_name}"
                )

            outputs[output_type_] = output

        return outputs if output_type == "all" else output

    def _get_voxelwise_model_attribute(self, attribute, result_as_time_series):
        """Transform RegressionResults instances within a dictionary \
        (whose keys represent the autoregressive coefficient under the 'ar1' \
        noise model or only 0.0 under 'ols' noise_model and values are the \
        RegressionResults instances) into input nifti space.

        Parameters
        ----------
        attribute : :obj:`str`
            an attribute of a RegressionResults instance.
            possible values include: residuals, normalized_residuals,
            predicted, SSE, r_square, MSE.

        result_as_time_series : :obj:`bool`
            whether the RegressionResult attribute has a value
            per timepoint of the input nifti image.

        Returns
        -------
        output : :obj:`list`
            A list of Nifti1Image(s).

        """
        # check if valid attribute is being accessed.
        all_attributes = dict(vars(RegressionResults)).keys()
        possible_attributes = [
            prop for prop in all_attributes if "__" not in prop
        ]
        if attribute not in possible_attributes:
            msg = f"attribute must be one of: {possible_attributes}"
            raise ValueError(msg)

        if self.minimize_memory:
            raise ValueError(
                "To access voxelwise attributes like "
                "R-squared, residuals, and predictions, "
                "the `FirstLevelModel`-object needs to store "
                "there attributes. "
                "To do so, set `minimize_memory` to `False` "
                "when initializing the `FirstLevelModel`-object."
            )

        self._check_fitted()

        output = []

        for design_matrix, labels, results in zip(
            self.design_matrices_, self.labels_, self.results_
        ):
            if result_as_time_series:
                voxelwise_attribute = np.zeros(
                    (design_matrix.shape[0], len(labels))
                )
            else:
                voxelwise_attribute = np.zeros((1, len(labels)))

            for label_ in results:
                label_mask = labels == label_
                voxelwise_attribute[:, label_mask] = getattr(
                    results[label_], attribute
                )

            output.append(self.masker_.inverse_transform(voxelwise_attribute))

        return output

    def _prepare_mask(self, run_img):
        """Set up the masker.

        Parameters
        ----------
        run_img : Niimg-like or :obj:`~nilearn.surface.SurfaceImage` object
            Used for setting up the masker object.
        """
        # Local import to prevent circular imports
        from nilearn.maskers import NiftiMasker

        # Learn the mask
        if self.mask_img is False:
            # We create a dummy mask to preserve functionality of api
            if isinstance(run_img, SurfaceImage):
                surf_data = {
                    part: np.ones(
                        run_img.data.parts[part].shape[0], dtype=bool
                    )
                    for part in run_img.mesh.parts
                }
                self.mask_img = SurfaceImage(mesh=run_img.mesh, data=surf_data)
            else:
                ref_img = check_niimg(run_img)
                self.mask_img = Nifti1Image(
                    np.ones(ref_img.shape[:3]), ref_img.affine
                )

        if isinstance(run_img, SurfaceImage) and not isinstance(
            self.mask_img, SurfaceMasker
        ):
            if self.smoothing_fwhm is not None:
                warn(
                    "Parameter smoothing_fwhm is not "
                    "yet supported for surface data",
                    UserWarning,
                    stacklevel=3,
                )
            self.masker_ = SurfaceMasker(
                mask_img=self.mask_img,
                smoothing_fwhm=self.smoothing_fwhm,
                standardize=self.standardize,
                t_r=self.t_r,
                memory=self.memory,
                memory_level=self.memory_level,
            )
            self.masker_.fit(run_img)

        elif not isinstance(
            self.mask_img, (NiftiMasker, SurfaceMasker, SurfaceImage)
        ):
            self.masker_ = NiftiMasker(
                mask_img=self.mask_img,
                smoothing_fwhm=self.smoothing_fwhm,
                target_affine=self.target_affine,
                standardize=self.standardize,
                mask_strategy="epi",
                t_r=self.t_r,
                memory=self.memory,
                verbose=max(0, self.verbose - 2),
                target_shape=self.target_shape,
                memory_level=self.memory_level,
            )
            self.masker_.fit(run_img)

        else:
            # Make sure masker has been fitted otherwise no attribute mask_img_
            self.mask_img._check_fitted()
            if self.mask_img.mask_img_ is None and self.masker_ is None:
                self.masker_ = clone(self.mask_img)
                for param_name in [
                    "target_affine",
                    "target_shape",
                    "smoothing_fwhm",
                    "t_r",
                    "memory",
                    "memory_level",
                ]:
                    our_param = getattr(self, param_name)
                    if our_param is None:
                        continue
                    if getattr(self.masker_, param_name) is not None:
                        warn(
                            f"Parameter {param_name} of the masker overridden"
                        )
                    if (
                        isinstance(self.masker_, SurfaceMasker)
                        and param_name not in ["target_affine", "target_shape"]
                    ) or not isinstance(self.masker_, SurfaceMasker):
                        setattr(self.masker_, param_name, our_param)
                self.masker_.fit(run_img)
            else:
                self.masker_ = self.mask_img


def _check_events_file_uses_tab_separators(events_files):
    """Raise a ValueError if provided list of text based data files \
    (.csv, .tsv, etc) do not enforce \
    the :term:`BIDS` convention of using Tabs as separators.

    Only scans their first row.
    Does nothing if:
        - If the separator used is :term:`BIDS` compliant.
        - Paths are invalid.
        - File(s) are not text files.

    Does not flag comma-separated-values-files for compatibility reasons;
    this may change in future as commas are not :term:`BIDS` compliant.

    Parameters
    ----------
    events_files : :obj:`str`, List/Tuple[str]
        A single file's path or a collection of filepaths.
        Files are expected to be text files.
        Non-text files will raise ValueError.

    Returns
    -------
    None

    Raises
    ------
    ValueError:
        If value separators are not Tabs (or commas)

    """
    valid_separators = [",", "\t"]
    if not isinstance(events_files, (list, tuple)):
        events_files = [events_files]
    for events_file_ in events_files:
        if isinstance(events_file_, (pd.DataFrame)):
            continue
        try:
            with Path(events_file_).open() as events_file_obj:
                events_file_sample = events_file_obj.readline()
            # The following errors are not being handled here,
            # as they are handled elsewhere in the calling code.
            # Handling them here will break the calling code,
            # and refactoring is not straightforward.
        except OSError:  # if invalid filepath.
            pass
        else:
            try:
                csv.Sniffer().sniff(
                    sample=events_file_sample,
                    delimiters=valid_separators,
                )
            except csv.Error as e:
                raise ValueError(
                    "The values in the events file "
                    "are not separated by tabs; "
                    "please enforce BIDS conventions",
                    events_file_,
                ) from e


def _check_run_tables(run_imgs, tables_, tables_name):
    """Check fMRI runs and corresponding tables to raise error if necessary."""
    _check_length_match(run_imgs, tables_, "run_imgs", tables_name)
    tables_ = check_and_load_tables(tables_, tables_name)
    return tables_


def _check_length_match(list_1, list_2, var_name_1, var_name_2):
    """Check length match of two given inputs to raise error if necessary."""
    if not isinstance(list_1, list):
        list_1 = [list_1]
    if not isinstance(list_2, list):
        list_2 = [list_2]
    if len(list_1) != len(list_2):
        raise ValueError(
            f"len({var_name_1}) {len(list_1)} does not match "
            f"len({var_name_2}) {len(list_2)}"
        )


def _check_repetition_time(t_r):
    """Check that the repetition time is a positive number."""
    if not isinstance(t_r, (float, int)):
        raise TypeError(
            f"'t_r' must be a float or an integer. Got {type(t_r)} instead."
        )
    if t_r <= 0:
        raise ValueError(f"'t_r' must be positive. Got {t_r} instead.")


def _check_slice_time_ref(slice_time_ref):
    """Check that slice_time_ref is a number between 0 and 1."""
    if not isinstance(slice_time_ref, (float, int)):
        raise TypeError(
            "'slice_time_ref' must be a float or an integer. "
            f"Got {type(slice_time_ref)} instead."
        )
    if slice_time_ref < 0 or slice_time_ref > 1:
        raise ValueError(
            "'slice_time_ref' must be between 0 and 1. "
            f"Got {slice_time_ref} instead."
        )


def first_level_from_bids(
    dataset_path,
    task_label,
    space_label=None,
    sub_labels=None,
    img_filters=None,
    t_r=None,
    slice_time_ref=0.0,
    hrf_model="glover",
    drift_model="cosine",
    high_pass=0.01,
    drift_order=1,
    fir_delays=None,
    min_onset=-24,
    mask_img=None,
    target_affine=None,
    target_shape=None,
    smoothing_fwhm=None,
    memory=None,
    memory_level=1,
    standardize=False,
    signal_scaling=0,
    noise_model="ar1",
    verbose=0,
    n_jobs=1,
    minimize_memory=True,
    derivatives_folder="derivatives",
    **kwargs,
):
    """Create FirstLevelModel objects and fit arguments \
       from a :term:`BIDS` dataset.

    If ``t_r`` is ``None``, this function will attempt
    to load it from a ``bold.json``.
    If ``slice_time_ref`` is  ``None``, this function will attempt
    to infer it from a ``bold.json``.
    Otherwise, ``t_r`` and ``slice_time_ref`` are taken as given,
    but a warning may be raised if they are not consistent with the
    ``bold.json``.

    All parameters not described here are passed to
    :class:`~nilearn.glm.first_level.FirstLevelModel`.

    The subject label of the model will be determined directly
    from the :term:`BIDS` dataset.

    Parameters
    ----------
    dataset_path : :obj:`str` or :obj:`pathlib.Path`
        Directory of the highest level folder of the :term:`BIDS` dataset.
        Should contain subject folders and a derivatives folder.

    task_label : :obj:`str`
        Task_label as specified in the file names like ``_task-<task_label>_``.

    space_label : :obj:`str` or None, default=None
        Specifies the space label of the preprocessed bold.nii images.
        As they are specified in the file names like ``_space-<space_label>_``.
        If "fsaverage5" is passed as a value
        then the GLM will be run on pial surface data.

    sub_labels : :obj:`list` of :obj:`str`, default=None
        Specifies the subset of subject labels to model.
        If ``None``, will model all subjects in the dataset.

        .. versionadded:: 0.10.1

    img_filters : :obj:`list` of :obj:`tuple` (:obj:`str`, :obj:`str`), \
        default=None
        Filters are of the form ``(field, label)``. Only one filter per field
        allowed.
        A file that does not match a filter will be discarded.
        Possible filters are ``'acq'``, ``'ce'``, ``'dir'``, ``'rec'``,
        ``'run'``, ``'echo'``, ``'res'``, ``'den'``, and ``'desc'``.
        Filter examples would be ``('desc', 'preproc')``, ``('dir', 'pa')``
        and ``('run', '10')``.

    slice_time_ref : :obj:`float` between ``0.0`` and ``1.0``, default= ``0.0``
        This parameter indicates the time of the reference slice used in the
        slice timing preprocessing step of the experimental runs. It is
        expressed as a fraction of the ``t_r`` (time repetition), so it can
        have values between ``0.`` and ``1.``

        .. deprecated:: 0.10.1

            The default= ``0`` for ``slice_time_ref`` will be deprecated.
            The default value will change to ``None`` in 0.12.

    derivatives_folder : :obj:`str`, default= ``"derivatives"``.
        derivatives and app folder path containing preprocessed files.
        Like ``"derivatives/FMRIPREP"``.

    kwargs : :obj:`dict`

        Keyword arguments to be passed to functions called within this
        function.

        Kwargs prefixed with ``confounds_``
        will be passed to :func:`~nilearn.interfaces.fmriprep.load_confounds`.
        This allows ``first_level_from_bids`` to return
        a specific set of confounds by relying on confound loading strategies
        defined in :func:`~nilearn.interfaces.fmriprep.load_confounds`.
        If no kwargs are passed, ``first_level_from_bids`` will return
        all the confounds available in the confounds TSV files.

        .. versionadded:: 0.10.3

    Examples
    --------
    If you want to only load
    the rotation and translation motion parameters confounds:

    .. code-block:: python

        models, imgs, events, confounds = first_level_from_bids(
            dataset_path=path_to_a_bids_dataset,
            task_label="TaskName",
            space_label="MNI",
            img_filters=[("desc", "preproc")],
            confounds_strategy=("motion"),
            confounds_motion="basic",
        )

    If you want to load the motion parameters confounds
    with their derivatives:

    .. code-block:: python

        models, imgs, events, confounds = first_level_from_bids(
            dataset_path=path_to_a_bids_dataset,
            task_label="TaskName",
            space_label="MNI",
            img_filters=[("desc", "preproc")],
            confounds_strategy=("motion"),
            confounds_motion="derivatives",
        )

    If you additionally want to load
    the confounds with CSF and white matter signal:

    .. code-block:: python

        models, imgs, events, confounds = first_level_from_bids(
            dataset_path=path_to_a_bids_dataset,
            task_label="TaskName",
            space_label="MNI",
            img_filters=[("desc", "preproc")],
            confounds_strategy=("motion", "wm_csf"),
            confounds_motion="derivatives",
            confounds_wm_csf="basic",
        )

    If you also want to scrub high-motion timepoints:

    .. code-block:: python

        models, imgs, events, confounds = first_level_from_bids(
            dataset_path=path_to_a_bids_dataset,
            task_label="TaskName",
            space_label="MNI",
            img_filters=[("desc", "preproc")],
            confounds_strategy=("motion", "wm_csf", "scrub"),
            confounds_motion="derivatives",
            confounds_wm_csf="basic",
            confounds_scrub=1,
            confounds_fd_threshold=0.2,
            confounds_std_dvars_threshold=0,
        )

    Please refer to the documentation
    of :func:`~nilearn.interfaces.fmriprep.load_confounds`
    for more details on the confounds loading strategies.

    Returns
    -------
    models : list of :class:`~nilearn.glm.first_level.FirstLevelModel` objects
        Each :class:`~nilearn.glm.first_level.FirstLevelModel` object
        corresponds to a subject.
        All runs from different sessions are considered together
        for the same subject to run a fixed effects analysis on them.

    models_run_imgs : :obj:`list` of list of Niimg-like objects,
        Items for the :class:`~nilearn.glm.first_level.FirstLevelModel`
        fit function of their respective model.

    models_events : :obj:`list` of list of pandas DataFrames,
        Items for the :class:`~nilearn.glm.first_level.FirstLevelModel`
        fit function of their respective model.

    models_confounds : :obj:`list` of list of pandas DataFrames or ``None``,
        Items for the :class:`~nilearn.glm.first_level.FirstLevelModel`
        fit function of their respective model.

    """
    if memory is None:
        memory = Memory(None)
    if slice_time_ref == 0:
        warn(
            "Starting in version 0.12, slice_time_ref will default to None.",
            DeprecationWarning,
        )
    if space_label is None:
        space_label = "MNI152NLin2009cAsym"

    sub_labels = sub_labels or []
    img_filters = img_filters or []

    _check_args_first_level_from_bids(
        dataset_path=dataset_path,
        task_label=task_label,
        space_label=space_label,
        sub_labels=sub_labels,
        img_filters=img_filters,
        derivatives_folder=derivatives_folder,
    )

    dataset_path = Path(dataset_path).absolute()

    kwargs_load_confounds, remaining_kwargs = _check_kwargs_load_confounds(
        **kwargs
    )

    if len(remaining_kwargs) > 0:
        raise RuntimeError(
            "Unknown keyword arguments. Keyword arguments should start with "
            f"`confounds_` prefix: {remaining_kwargs}"
        )

    if (
        drift_model is not None
        and kwargs_load_confounds is not None
        and "high_pass" in kwargs_load_confounds.get("strategy")
    ):
        if drift_model == "cosine":
            verb = "duplicate"
        if drift_model == "polynomial":
            verb = "conflict with"

        warn(
            f"""Confounds will contain a high pass filter,
 that may {verb} the {drift_model} one used in the model.
 Remember to visualize your design matrix before fitting your model
 to check that your model is not overspecified.""",
            UserWarning,
        )

    derivatives_path = Path(dataset_path) / derivatives_folder
    derivatives_path = derivatives_path.absolute()

    # Get metadata for models.
    #
    # We do it once and assume all subjects and runs
    # have the same value.

    # Repetition time
    #
    # Try to find a t_r value in the bids datasets
    # If the parameter information is not found in the derivatives folder,
    # a search is done in the raw data folder.
    filters = _make_bids_files_filter(
        task_label=task_label,
        space_label=space_label,
        supported_filters=[
            *bids_entities()["raw"],
            *bids_entities()["derivatives"],
        ],
        extra_filter=img_filters,
        verbose=verbose,
    )
    inferred_t_r = infer_repetition_time_from_dataset(
        bids_path=derivatives_path, filters=filters, verbose=verbose
    )
    if inferred_t_r is None:
        filters = _make_bids_files_filter(
            task_label=task_label,
            supported_filters=[*bids_entities()["raw"]],
            extra_filter=img_filters,
            verbose=verbose,
        )
        inferred_t_r = infer_repetition_time_from_dataset(
            bids_path=dataset_path, filters=filters, verbose=verbose
        )

    if t_r is None and inferred_t_r is not None:
        t_r = inferred_t_r
    if t_r is not None and t_r != inferred_t_r:
        warn(
            f"\n't_r' provided ({t_r}) is different "
            f"from the value found in the BIDS dataset ({inferred_t_r}).\n"
            "Note this may lead to the wrong model specification.",
            stacklevel=2,
        )
    if t_r is not None:
        _check_repetition_time(t_r)
    else:
        warn(
            "\n't_r' not provided and cannot be inferred from BIDS metadata.\n"
            "It will need to be set manually in the list of models, "
            "otherwise their fit will throw an exception.",
            stacklevel=2,
        )

    # Slice time correction reference time
    #
    # Try to infer a slice_time_ref value in the bids derivatives dataset.
    #
    # If no value can be inferred, the default value of 0.0 is used.
    filters = _make_bids_files_filter(
        task_label=task_label,
        space_label=space_label,
        supported_filters=[
            *bids_entities()["raw"],
            *bids_entities()["derivatives"],
        ],
        extra_filter=img_filters,
        verbose=verbose,
    )
    StartTime = infer_slice_timing_start_time_from_dataset(
        bids_path=derivatives_path, filters=filters, verbose=verbose
    )
    if StartTime is not None and t_r is not None:
        assert StartTime < t_r
        inferred_slice_time_ref = StartTime / t_r
    else:
        if slice_time_ref is None:
            warn(
                "'slice_time_ref' not provided "
                "and cannot be inferred from metadata.\n"
                "It will be assumed that the slice timing reference "
                "is 0.0 percent of the repetition time.\n"
                "If it is not the case it will need to "
                "be set manually in the generated list of models."
            )
        inferred_slice_time_ref = 0.0

    if slice_time_ref is None and inferred_slice_time_ref is not None:
        slice_time_ref = inferred_slice_time_ref
    if (
        slice_time_ref is not None
        and slice_time_ref != inferred_slice_time_ref
    ):
        warn(
            f"'slice_time_ref' provided ({slice_time_ref}) is different "
            f"from the value found in the BIDS dataset "
            f"({inferred_slice_time_ref}).\n"
            "Note this may lead to the wrong model specification."
        )
    if slice_time_ref is not None:
        _check_slice_time_ref(slice_time_ref)

    # Build fit_kwargs dictionaries to pass to their respective models fit
    # Events and confounds files must match number of imgs (runs)
    models = []
    models_run_imgs = []
    models_events = []
    models_confounds = []

    sub_labels = _list_valid_subjects(derivatives_path, sub_labels)
    if len(sub_labels) == 0:
        raise RuntimeError(f"\nNo subject found in:\n {derivatives_path}")
    for sub_label_ in sub_labels:
        # Create model
        model = FirstLevelModel(
            t_r=t_r,
            slice_time_ref=slice_time_ref,
            hrf_model=hrf_model,
            drift_model=drift_model,
            high_pass=high_pass,
            drift_order=drift_order,
            fir_delays=fir_delays,
            min_onset=min_onset,
            mask_img=mask_img,
            target_affine=target_affine,
            target_shape=target_shape,
            smoothing_fwhm=smoothing_fwhm,
            memory=memory,
            memory_level=memory_level,
            standardize=standardize,
            signal_scaling=signal_scaling,
            noise_model=noise_model,
            verbose=verbose,
            n_jobs=n_jobs,
            minimize_memory=minimize_memory,
            subject_label=sub_label_,
        )
        models.append(model)

        imgs, files_to_check = _get_processed_imgs(
            derivatives_path=derivatives_path,
            sub_label=sub_label_,
            task_label=task_label,
            space_label=space_label,
            img_filters=img_filters,
            verbose=verbose,
        )
        models_run_imgs.append(imgs)

        events = _get_events_files(
            dataset_path=dataset_path,
            sub_label=sub_label_,
            task_label=task_label,
            img_filters=img_filters,
            imgs=files_to_check,
            verbose=verbose,
        )
        events = [
            pd.read_csv(event, sep="\t", index_col=None) for event in events
        ]
        models_events.append(events)

        confounds = _get_confounds(
            derivatives_path=derivatives_path,
            sub_label=sub_label_,
            task_label=task_label,
            img_filters=img_filters,
            imgs=files_to_check,
            verbose=verbose,
            kwargs_load_confounds=kwargs_load_confounds,
        )
        models_confounds.append(confounds)

    return models, models_run_imgs, models_events, models_confounds


def _list_valid_subjects(derivatives_path, sub_labels):
    """List valid subjects in the dataset.

    - Include all subjects if no subject pre-selection is passed.
    - Exclude subjects that do not exist in the derivatives folder.
    - Remove duplicate subjects.

    Parameters
    ----------
    derivatives_path : :obj:`str` or :obj:`pathlib.Path`
        Path to the BIDS derivatives folder.

    sub_labels : :obj:`list` of :obj:`str`
        List of subject labels to process.
        If None, all subjects in the dataset will be processed.

    Returns
    -------
    sub_labels : :obj:`list` of :obj:`str`
        List of subject labels that will be processed.
    """
    derivatives_path = Path(derivatives_path)
    # Infer subjects in dataset if not provided
    if not sub_labels:
        sub_folders = derivatives_path.glob("sub-*/")
        sub_labels = [s.name.split("-")[1] for s in sub_folders if s.is_dir()]

    # keep only existing subjects
    sub_labels_exist = []
    for sub_label_ in sub_labels:
        if (derivatives_path / f"sub-{sub_label_}").exists():
            sub_labels_exist.append(sub_label_)
        else:
            warn(
                f"\nSubject label '{sub_label_}' is not present "
                "in the following dataset and cannot be processed:\n"
                f" {derivatives_path}",
                stacklevel=3,
            )

    return sorted(set(sub_labels_exist))


def _report_found_files(files, text, sub_label, filters, verbose):
    """Print list of files found for a given subject and filter.

    Parameters
    ----------
    files : :obj:`list` of :obj:`str`
        List of fullpath of files.

    text :  :obj:`str`
        Text description of the file type.

    sub_label : :obj:`str`
        Subject label as specified in the file names like sub-<sub_label>_.

    filters : :obj:`list` of :obj:`tuple` (str, str)
        Filters are of the form (field, label).
        Only one filter per field allowed.

    """
    unordered_list_string = "\n\t- ".join(files)
    logger.log(
        f"\nFound the following {len(files)} {text} files\n"
        f"- for subject {sub_label}\n"
        f"- for filter: {filters}:\n\t"
        f"- {unordered_list_string}\n",
        verbose=verbose,
        stack_level=3,
    )


def _get_processed_imgs(
    derivatives_path, sub_label, task_label, space_label, img_filters, verbose
):
    """Get images for a given subject, task and filters.

    Also checks that there is only one images per run / session.

    Parameters
    ----------
    derivatives_path : :obj:`str`
        Directory of the derivatives BIDS dataset.

    sub_label : :obj:`str`
        Subject label as specified in the file names like sub-<sub_label>_.

    task_label : :obj:`str`
        Task label as specified in the file names like _task-<task_label>_.

    space_label : None or :obj:`str`

    img_filters : :obj:`list` of :obj:`tuple` (str, str)
        Filters are of the form (field, label).
        Only one filter per field allowed.

    verbose : :obj:`integer`
        Indicate the level of verbosity.

    Returns
    -------
    imgs : :obj:`list` of :obj:`str`, \
        or :obj:`list` of :obj:`~nilearn.surface.SurfaceImage`
        List of fullpath to the imgs files
        If fsaverage5 is passed then both hemisphere for each run
        will be loaded into a single SurfaceImage.

    files_to_check : : :obj:`list` of :obj:`str`
        List of fullpath to imgs files.
        Used for validation
        when finding events or confounds associated with images.
    """
    filters = _make_bids_files_filter(
        task_label=task_label,
        space_label=space_label,
        supported_filters=bids_entities()["raw"]
        + bids_entities()["derivatives"],
        extra_filter=img_filters,
        verbose=verbose,
    )

    if space_label is not None and (
        space_label == "" or space_label not in ("fsaverage5")
    ):
        imgs = get_bids_files(
            main_path=derivatives_path,
            modality_folder="func",
            file_tag="bold",
            file_type="nii*",
            sub_label=sub_label,
            filters=filters,
        )
        files_to_report = imgs
        files_to_check = imgs

    else:
        tmp_filter = filters.copy()
        tmp_filter.append(("hemi", "L"))
        imgs_left = get_bids_files(
            main_path=derivatives_path,
            modality_folder="func",
            file_tag="bold",
            file_type="func.gii",
            sub_label=sub_label,
            filters=tmp_filter,
        )
        tmp_filter[-1] = ("hemi", "R")
        imgs_right = get_bids_files(
            main_path=derivatives_path,
            modality_folder="func",
            file_tag="bold",
            file_type="func.gii",
            sub_label=sub_label,
            filters=tmp_filter,
        )

        # Sanity check to make sure we have the same number of files
        # for each hemisphere
        assert len(imgs_left) == len(imgs_right)

        imgs = []
        for data_left, data_right in zip(imgs_left, imgs_right):
            # make sure that filenames only differ by hemisphere
            assert (
                Path(data_left).stem.replace("hemi-L", "hemi-R")
                == Path(data_right).stem
            )
            # Assumption: we are loading the data on the pial surface.
            imgs.append(
                SurfaceImage(
                    mesh=load_fsaverage()["pial"],
                    data={"left": data_left, "right": data_right},
                )
            )

        files_to_report = imgs_left + imgs_right

        # Only check the left files
        # as we know they have a right counterpart.
        files_to_check = imgs_left

    _report_found_files(
        files=files_to_report,
        text="preprocessed BOLD",
        sub_label=sub_label,
        filters=filters,
        verbose=verbose,
    )
    _check_bids_image_list(files_to_check, sub_label, filters)
    return imgs, files_to_check


def _get_events_files(
    dataset_path,
    sub_label,
    task_label,
    img_filters,
    imgs,
    verbose,
):
    """Get events.tsv files for a given subject, task and filters.

    Also checks that the number of events.tsv files
    matches the number of images.

    Parameters
    ----------
    dataset_path : :obj:`str`
        Directory of the derivatives BIDS dataset.

    sub_label : :obj:`str`
        Subject label as specified in the file names like sub-<sub_label>_.

    task_label : :obj:`str`
        Task label as specified in the file names like _task-<task_label>_.

    img_filters : :obj:`list` of :obj:`tuple` (str, str)
        Filters are of the form (field, label).
        Only one filter per field allowed.

    imgs : :obj:`list` of :obj:`str`
        List of fullpath to the preprocessed images

    verbose : :obj:`integer`
        Indicate the level of verbosity.

    Returns
    -------
    events : :obj:`list` of :obj:`str`
        List of fullpath to the events files
    """
    # pop the derivatives filter
    # it would otherwise trigger some meaningless warnings
    # as the derivatives entity are not supported in BIDS raw datasets
    img_filters = [
        x for x in img_filters if x[0] not in bids_entities()["derivatives"]
    ]
    events_filters = _make_bids_files_filter(
        task_label=task_label,
        supported_filters=bids_entities()["raw"],
        extra_filter=img_filters,
        verbose=verbose,
    )
    events = get_bids_files(
        dataset_path,
        modality_folder="func",
        file_tag="events",
        file_type="tsv",
        sub_label=sub_label,
        filters=events_filters,
    )
    _report_found_files(
        files=events,
        text="events",
        sub_label=sub_label,
        filters=events_filters,
        verbose=verbose,
    )
    _check_bids_events_list(
        events=events,
        imgs=imgs,
        sub_label=sub_label,
        task_label=task_label,
        dataset_path=dataset_path,
        events_filters=events_filters,
        verbose=verbose,
    )
    return events


def _get_confounds(
    derivatives_path,
    sub_label,
    task_label,
    img_filters,
    imgs,
    verbose,
    kwargs_load_confounds,
):
    """Get confounds.tsv files for a given subject, task and filters.

    Also checks that the number of confounds.tsv files
    matches the number of images.

    Parameters
    ----------
    derivatives_path : :obj:`str`
        Directory of the derivatives BIDS dataset.

    sub_label : :obj:`str`
        Subject label as specified in the file names like sub-<sub_label>_.

    task_label : :obj:`str`
        Task label as specified in the file names like _task-<task_label>_.

    img_filters : :obj:`list` of :obj:`tuple` (str, str)
        Filters are of the form (field, label).
        Only one filter per field allowed.

    imgs : :obj:`list` of :obj:`str`
        List of fullpath to the preprocessed images

    verbose : :obj:`integer`
        Indicate the level of verbosity.

    Returns
    -------
    confounds : :obj:`list` of :class:`pandas.DataFrame`

    """
    # pop the 'desc' filter
    # it would otherwise trigger some meaningless warnings
    # as desc entity are not supported in BIDS raw datasets
    # and we add a desc-confounds 'filter' later on
    img_filters = [x for x in img_filters if x[0] != "desc"]
    filters = _make_bids_files_filter(
        task_label=task_label,
        supported_filters=bids_entities()["raw"],
        extra_filter=img_filters,
        verbose=verbose,
    )
    confounds = get_bids_files(
        derivatives_path,
        modality_folder="func",
        file_tag="desc-confounds*",
        file_type="tsv",
        sub_label=sub_label,
        filters=filters,
    )
    _report_found_files(
        files=confounds,
        text="confounds",
        sub_label=sub_label,
        filters=filters,
        verbose=verbose,
    )
    _check_confounds_list(confounds=confounds, imgs=imgs)

    if confounds:
        if kwargs_load_confounds is None:
            confounds = [
                pd.read_csv(c, sep="\t", index_col=None) for c in confounds
            ]
            return confounds or None

        confounds, _ = load_confounds(img_files=imgs, **kwargs_load_confounds)

        return confounds


def _check_confounds_list(confounds, imgs):
    """Check the number of confounds.tsv files.

    If no file is found, it will be assumed there are none,
    but if there are any confounds files, there must be one per run.

    Parameters
    ----------
    confounds : :obj:`list` of :obj:`str`
        List of fullpath to the confounds.tsv files

    imgs : :obj:`list` of :obj:`str`
        List of fullpath to the preprocessed images

    """
    if confounds and len(confounds) != len(imgs):
        raise ValueError(
            f"{len(confounds)} confounds.tsv files found "
            f"for {len(imgs)} bold files. "
            "Same number of confound files as "
            "the number of runs is expected"
        )


def _check_args_first_level_from_bids(
    dataset_path,
    task_label,
    space_label,
    sub_labels,
    img_filters,
    derivatives_folder,
):
    """Check type and value of arguments of first_level_from_bids.

    Check that:
        - dataset_path is a string and exists
        - derivatives_path exists
        - task_label and space_label are valid bids labels
        - img_filters is a list of tuples of strings
          and all filters are valid bids entities
          with valid bids labels

    Parameters
    ----------
    dataset_path : :obj:`str`
        Fullpath of the BIDS dataset root folder.

    task_label : :obj:`str`
        Task_label as specified in the file names like _task-<task_label>_.

    space_label : :obj:`str`
        Specifies the space label of the preprocessed bold.nii images.
        As they are specified in the file names like _space-<space_label>_.

    sub_labels : :obj:`list` of :obj:`str`, optional
        Specifies the subset of subject labels to model.
        If 'None', will model all subjects in the dataset.

    img_filters : :obj:`list` of :obj:`tuples` (str, str)
        Filters are of the form (field, label).
        Only one filter per field allowed.

    derivatives_path : :obj:`str`
        Fullpath of the BIDS dataset derivative folder.

    """
    if not isinstance(dataset_path, (str, Path)):
        raise TypeError(
            "'dataset_path' must be a string or pathlike. "
            f"Got {type(dataset_path)} instead."
        )
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise ValueError(f"'dataset_path' does not exist:\n{dataset_path}")

    if not isinstance(derivatives_folder, str):
        raise TypeError(
            "'derivatives_folder' must be a string. "
            f"Got {type(derivatives_folder)} instead."
        )
    derivatives_folder = dataset_path / derivatives_folder
    if not derivatives_folder.exists():
        raise ValueError(
            "derivatives folder not found in given dataset:\n"
            f"{derivatives_folder}"
        )

    check_bids_label(task_label)

    if space_label is not None:
        check_bids_label(space_label)

    if not isinstance(sub_labels, list):
        raise TypeError(
            f"sub_labels must be a list, instead {type(sub_labels)} was given"
        )
    for sub_label_ in sub_labels:
        check_bids_label(sub_label_)

    if not isinstance(img_filters, list):
        raise TypeError(
            f"'img_filters' must be a list. "
            f"Got {type(img_filters)} instead."
        )
    supported_filters = [
        *bids_entities()["raw"],
        *bids_entities()["derivatives"],
    ]
    for filter_ in img_filters:
        if len(filter_) != 2 or not all(isinstance(x, str) for x in filter_):
            raise TypeError(
                "Filters in img_filters must be (str, str). "
                f"Got {filter_} instead."
            )
        if filter_[0] not in supported_filters:
            raise ValueError(
                f"Entity {filter_[0]} for {filter_} is not a possible filter. "
                f"Only {supported_filters} are allowed."
            )
        check_bids_label(filter_[1])


def _check_kwargs_load_confounds(**kwargs):
    # reuse the default from nilearn.interface.fmriprep.load_confounds
    defaults = {
        "strategy": ("motion", "high_pass", "wm_csf"),
        "motion": "full",
        "scrub": 5,
        "fd_threshold": 0.2,
        "std_dvars_threshold": 3,
        "wm_csf": "basic",
        "global_signal": "basic",
        "compcor": "anat_combined",
        "n_compcor": "all",
        "ica_aroma": "full",
        "demean": True,
    }

    if kwargs.get("confounds_strategy") is None:
        return None, kwargs

    remaining_kwargs = kwargs.copy()
    kwargs_load_confounds = {}
    for key in defaults:
        confounds_key = f"confounds_{key}"
        if confounds_key in kwargs:
            kwargs_load_confounds[key] = remaining_kwargs.pop(confounds_key)
        else:
            kwargs_load_confounds[key] = defaults[key]

    return kwargs_load_confounds, remaining_kwargs


def _make_bids_files_filter(
    task_label,
    space_label=None,
    supported_filters=None,
    extra_filter=None,
    verbose=0,
):
    """Return a filter to specific files from a BIDS dataset.

    Parameters
    ----------
    task_label : :obj:`str`
        Task label as specified in the file names like _task-<task_label>_.

    space_label : :obj:`str` or None, optional
        Specifies the space label of the preprocessed bold.nii images.
        As they are specified in the file names like _space-<space_label>_.

    supported_filters : :obj:`list` of :obj:`str` or None, optional
        List of authorized BIDS entities

    extra_filter : :obj:`list` of :obj:`tuple` (str, str) or None, optional
        Filters are of the form (field, label).
        Only one filter per field allowed.

    verbose : :obj:`integer`
        Indicate the level of verbosity.

    Returns
    -------
    Filter to be used by :func:`get_bids_files`: \
        :obj:`list` of :obj:`tuple` (str, str)
        filters

    """
    filters = [("task", task_label)]

    if space_label is not None:
        filters.append(("space", space_label))

    if extra_filter and supported_filters:
        for filter_ in extra_filter:
            if filter_[0] not in supported_filters:
                if verbose:
                    warn(
                        f"The filter {filter_} will be skipped. "
                        f"'{filter_[0]}' is not among the supported filters. "
                        f"Allowed filters include: {supported_filters}",
                        stacklevel=3,
                    )
                continue

            filters.append(filter_)

    return filters


def _check_bids_image_list(imgs, sub_label, filters):
    """Check input BIDS images.

    Check that:
        - some images were found
        - if more than one image was found, check that there is not more than
          one image for a given session / run combination.

    Parameters
    ----------
    imgs : :obj:`list` of :obj:`str` or None
        List of image fullpath filenames.

    sub_label : :obj:`str`
        Subject label as specified in the file names like _sub-<sub_label>_.

    filters : :obj:`list` of :obj:`tuple` (str, str)
        Filters of the form (field, label) used to select the files.
        See :func:`get_bids_files`.

    """
    if not imgs:
        raise ValueError(
            "No BOLD files found "
            f"for subject {sub_label} "
            f"for filter: {filters}"
        )

    if len(imgs) <= 1:
        return

    msg_start = (
        "Too many images found\n "
        f"for subject: '{sub_label}'\n"
        f"for filters: {filters}\n"
    )
    msg_end = (
        "Please specify it further by setting, "
        "for example, some required task_label, "
        "space_label or img_filters"
    )

    run_check_list: list = []

    for img_ in imgs:
        parsed_filename = parse_bids_filename(img_)
        session = parsed_filename.get("ses")
        run = parsed_filename.get("run")

        if session and run:
            if (session, run) in set(run_check_list):
                raise ValueError(
                    f"{msg_start}"
                    f"for the same run {run} and session {session}. "
                    f"{msg_end}"
                )
            run_check_list.append((session, run))

        elif session:
            if session in set(run_check_list):
                raise ValueError(
                    f"{msg_start}"
                    f"for the same session {session}, "
                    "while no additional run specification present. "
                    f"{msg_end}"
                )
            run_check_list.append(session)

        elif run:
            if run in set(run_check_list):
                raise ValueError(
                    f"{msg_start}for the same run {run}. {msg_end}"
                )
            run_check_list.append(run)


def _check_bids_events_list(
    events, imgs, sub_label, task_label, dataset_path, events_filters, verbose
):
    """Check input BIDS events.

    Check that:
        - some events.tsv files were found
        - as many events.tsv were found as images
        - there is only one events.tsv per image and that they have the same
          raw entities.

    Parameters
    ----------
    events : :obj:`list` of :obj:`str` or None
        List of events.tsv fullpath filenames.

    imgs : :obj:`list` of :obj:`str`
        List of image fullpath filenames.

    sub_label : :obj:`str`
        Subject label as specified in the file names like sub-<sub_label>_.

    task_label : :obj:`str`
        Task label as specified in the file names like _task-<task_label>_.

    dataset_path : :obj:`str`
        Fullpath to the BIDS dataset.

    events_filters : :obj:`list` of :obj:`tuple` (str, str)
        Filters of the form (field, label) used to select the files.
        See :func:`get_bids_files`.

    """
    if not events:
        raise ValueError(
            "No events.tsv files found "
            f"for subject {sub_label} "
            f"for filter: {events_filters}."
        )
    if len(events) != len(imgs):
        raise ValueError(
            f"{len(events)} events.tsv files found"
            f" for {len(imgs)} bold files. "
            "Same number of event files "
            "as the number of runs is expected."
        )
    _check_trial_type(events=events)

    supported_filters = [
        "sub",
        "ses",
        "task",
        *bids_entities()["raw"],
    ]
    for this_img in imgs:
        parsed_filename = parse_bids_filename(this_img)
        extra_filter = [
            (key, parsed_filename[key])
            for key in parsed_filename
            if key in supported_filters
        ]
        filters = _make_bids_files_filter(
            task_label=task_label,
            space_label=None,
            supported_filters=supported_filters,
            extra_filter=extra_filter,
            verbose=verbose,
        )
        this_event = get_bids_files(
            dataset_path,
            modality_folder="func",
            file_tag="events",
            file_type="tsv",
            sub_label=sub_label,
            filters=filters,
        )
        msg_suffix = (
            f"bold file:\n{this_img}\nfilter:\n{filters})\n"
            "Found all the following events files "
            f"for filter:\n{events}\n"
        )
        if len(this_event) == 0:
            raise ValueError(
                f"No events.tsv files corresponding to {msg_suffix}"
            )
        if len(this_event) > 1:
            raise ValueError(
                f"More than 1 events.tsv files "
                f"corresponding to {msg_suffix}"
            )
        if this_event[0] not in events:
            raise ValueError(
                f"\n{this_event} not in {events}.\n"
                "No corresponding events.tsv files found "
                f"for {msg_suffix}"
            )

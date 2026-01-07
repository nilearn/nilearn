"""Transformer used to apply basic transformations on MRI data."""

import inspect
import warnings
from copy import copy as copy_object

import numpy as np
from joblib import Memory
from sklearn.base import ClassNamePrefixFeaturesOutMixin
from sklearn.utils.estimator_checks import check_is_fitted

from nilearn._utils.class_inspect import get_params
from nilearn._utils.docs import fill_doc
from nilearn._utils.helpers import is_matplotlib_installed
from nilearn._utils.logger import find_stack_level
from nilearn._utils.niimg import img_data_dtype
from nilearn._utils.param_validation import check_params
from nilearn.image import check_niimg, crop_img, resample_img
from nilearn.image.image import check_same_fov
from nilearn.maskers._utils import compute_middle_image
from nilearn.maskers.base_masker import (
    BaseMasker,
    filter_and_extract,
    mask_logger,
)
from nilearn.masking import (
    apply_mask,
    compute_background_mask,
    compute_brain_mask,
    compute_epi_mask,
    compute_multi_brain_mask,
    load_mask_img,
)


class _ExtractionFunctor:
    func_name = "nifti_masker_extractor"

    def __init__(self, mask_img_):
        self.mask_img_ = mask_img_

    def __call__(self, imgs):
        return (
            apply_mask(
                imgs,
                self.mask_img_,
                dtype=img_data_dtype(imgs),
            ),
            imgs.affine,
        )


def _get_mask_strategy(strategy: str):
    """Return the mask computing method based on a provided strategy."""
    if strategy == "background":
        return compute_background_mask
    elif strategy == "epi":
        return compute_epi_mask
    elif strategy == "whole-brain-template":
        return make_brain_mask_func("whole-brain")
    elif strategy == "gm-template":
        return make_brain_mask_func("gm")
    elif strategy == "wm-template":
        return make_brain_mask_func("wm")
    else:
        raise ValueError(
            f"Unknown value of mask_strategy '{strategy}'. "
            "Acceptable values are 'background', "
            "'epi', 'whole-brain-template', "
            "'gm-template', and "
            "'wm-template'."
        )


def make_brain_mask_func(mask_type: str, multi: bool = False):
    """Generate a compute_brain_mask function adapted for each mask.

    This is done instead of using functools.partial because
    joblib does not play well with partials.

    See: https://github.com/nilearn/nilearn/issues/5527

    Parameters
    ----------
    mask_type : str
        Type of masking function to return.

    multi : bool
        Whether to return functions for multimasker or not.
    """

    def _compute(
        target_img,
        threshold=0.5,
        connected=True,
        opening=2,
        memory=None,
        verbose=0,
    ):
        if multi:
            return compute_multi_brain_mask(
                target_img,
                threshold,
                connected,
                opening,
                memory,
                verbose,
                mask_type=mask_type,
            )

        return compute_brain_mask(
            target_img,
            threshold,
            connected,
            opening,
            memory,
            verbose,
            mask_type=mask_type,
        )

    return _compute


def filter_and_mask(
    imgs,
    mask_img_,
    parameters,
    memory_level=0,
    memory=None,
    verbose=0,
    confounds=None,
    sample_mask=None,
    copy=True,
    dtype=None,
    sklearn_output_config=None,
):
    """Extract representative time series using given mask.

    Parameters
    ----------
    imgs : 3D/4D Niimg-like object
        Images to be masked. Can be 3-dimensional or 4-dimensional.

    For all other parameters refer to NiftiMasker documentation.

    Returns
    -------
    signals : 2D numpy array
        Signals extracted using the provided mask. It is a scikit-learn
        friendly 2D array with shape n_sample x n_features.

    """
    if memory is None:
        memory = Memory(location=None)
    # Convert input to niimg to check shape.
    # This must be repeated after the shape check because check_niimg will
    # coerce 5D data to 4D, which we don't want.
    temp_imgs = check_niimg(imgs)

    imgs = check_niimg(imgs, atleast_4d=True, ensure_ndim=4)

    # Check whether resampling is truly necessary. If so, crop mask
    # as small as possible in order to speed up the process

    if not check_same_fov(imgs, mask_img_):
        warnings.warn(
            "imgs are being resampled to the mask_img resolution. "
            "This process is memory intensive. You might want to provide "
            "a target_affine that is equal to the affine of the imgs "
            "or resample the mask beforehand "
            "to save memory and computation time.",
            UserWarning,
            stacklevel=find_stack_level(),
        )
        parameters = copy_object(parameters)
        # now we can crop
        mask_img_ = crop_img(mask_img_, copy=False)
        parameters["target_shape"] = mask_img_.shape
        parameters["target_affine"] = mask_img_.affine

    data, _ = filter_and_extract(
        imgs,
        _ExtractionFunctor(mask_img_),
        parameters,
        memory_level=memory_level,
        memory=memory,
        verbose=verbose,
        confounds=confounds,
        sample_mask=sample_mask,
        copy=copy,
        dtype=dtype,
    )
    # For _later_: missing value removal or imputing of missing data
    # (i.e. we want to get rid of NaNs, if smoothing must be done
    # earlier)
    # Optionally: 'doctor_nan', remove voxels with NaNs, other option
    # for later: some form of imputation

    # if we need to output to numpy and input was a 3D img
    # we return 1D array
    if temp_imgs.ndim == 3 and sklearn_output_config is None:
        data = data.squeeze()
    return data


@fill_doc
class NiftiMasker(ClassNamePrefixFeaturesOutMixin, BaseMasker):
    """Applying a mask to extract time-series from Niimg-like objects.

    NiftiMasker is useful when preprocessing (detrending, standardization,
    resampling, etc.) of in-mask :term:`voxels<voxel>` is necessary.

    Use case:
    working with time series of :term:`resting-state` or task maps.

    Parameters
    ----------
    mask_img : Niimg-like object, optional
        See :ref:`extracting_data`.
        Mask for the data. If not given, a mask is computed in the fit step.
        Optional parameters (mask_args and mask_strategy) can be set to
        fine tune the mask extraction.
        If the mask and the images have different resolutions, the images
        are resampled to the mask resolution.
        If target_shape and/or target_affine are provided, the mask is
        resampled first. After this, the images are resampled to the
        resampled mask.

    runs : :obj:`numpy.ndarray`, optional
        Add a run level to the preprocessing. Each run will be
        detrended independently. Must be a 1D array of n_samples elements.

    %(smoothing_fwhm)s

    %(standardize_false)s

    %(standardize_confounds)s

    %(detrend)s

    high_variance_confounds : :obj:`bool`, default=False
        If True, high variance confounds are computed on provided image with
        :func:`nilearn.image.high_variance_confounds` and default parameters
        and regressed out.

    %(low_pass)s

    %(high_pass)s

    %(t_r)s

    %(target_affine)s

        .. note::
            This parameter is passed to :func:`nilearn.image.resample_img`.

    %(target_shape)s

        .. note::
            This parameter is passed to :func:`nilearn.image.resample_img`.

    %(mask_strategy)s

        .. note::
            Depending on this value, the mask will be computed from
            :func:`nilearn.masking.compute_background_mask`,
            :func:`nilearn.masking.compute_epi_mask`, or
            :func:`nilearn.masking.compute_brain_mask`.

        Default='background'.

    mask_args : :obj:`dict`, optional
        If mask is None, these are additional parameters passed to
        :func:`nilearn.masking.compute_background_mask`,
        or :func:`nilearn.masking.compute_epi_mask`
        to fine-tune mask computation.
        Please see the related documentation for details.

    %(dtype)s

    %(memory)s

    %(memory_level1)s

    %(verbose0)s

    reports : :obj:`bool`, default=True
        If set to True, data is saved in order to produce a report.

    %(cmap)s
        default="gray"
        Only relevant for the report figures.

    %(clean_args)s

        .. nilearn_versionadded:: 0.12.0


    Attributes
    ----------
    affine_ : 4x4 :obj:`numpy.ndarray`
        Affine of the transformed image.

    %(clean_args_)s

    %(nifti_mask_img_)s

    memory_ : joblib memory cache

    n_elements_ : :obj:`int`
        The number of voxels in the mask.

        .. nilearn_versionadded:: 0.9.2

    See Also
    --------
    nilearn.masking.compute_background_mask
    nilearn.masking.compute_epi_mask
    nilearn.image.resample_img
    nilearn.image.high_variance_confounds
    nilearn.masking.apply_mask
    nilearn.signal.clean

    """

    def __init__(
        self,
        mask_img=None,
        runs=None,
        smoothing_fwhm=None,
        standardize=False,
        standardize_confounds=True,
        detrend=False,
        high_variance_confounds=False,
        low_pass=None,
        high_pass=None,
        t_r=None,
        target_affine=None,
        target_shape=None,
        mask_strategy="background",
        mask_args=None,
        dtype=None,
        memory=None,
        memory_level=1,
        verbose=0,
        reports=True,
        cmap="gray",
        clean_args=None,
    ):
        # Mask is provided or computed
        self.mask_img = mask_img
        self.runs = runs
        self.smoothing_fwhm = smoothing_fwhm
        self.standardize = standardize
        self.standardize_confounds = standardize_confounds
        self.high_variance_confounds = high_variance_confounds
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.mask_strategy = mask_strategy
        self.mask_args = mask_args
        self.dtype = dtype
        self.memory = memory
        self.memory_level = memory_level
        self.verbose = verbose
        self.reports = reports
        self.cmap = cmap
        self.clean_args = clean_args

        self._report_content = {
            "description": (
                "This report shows the input Nifti image overlaid "
                "with the outlines of the mask (in green). We "
                "recommend to inspect the report for the overlap "
                "between the mask and its input image. "
            ),
            "n_elements": 0,
            "coverage": 0,
            "summary": {},
            "warning_messages": [],
        }

    def generate_report(self, title: str | None = None):
        """Generate an HTML report for the current object.

        Parameters
        ----------
        title : :obj:`str` or None, default=None
            title for the report. If None, title will be the class name.

        Returns
        -------
        report : `nilearn.reporting.html_report.HTMLReport`
            HTML report for the masker.
        """
        from nilearn.reporting.html_report import generate_report

        self._report_content["title"] = title

        if self._has_report_data():
            img = self._reporting_data["images"]

            if img is None:  # images were not provided to fit
                msg = (
                    f"No image provided to fit in {self.__class__.__name__}. "
                    "Setting image to mask for reporting."
                )
                self._report_content["warning_messages"].append(msg)

            elif self._reporting_data["dim"] == 5:
                msg = (
                    "A list of 4D subject images were provided to fit. "
                    "Only first subject is shown in the report."
                )
                self._report_content["warning_messages"].append(msg)

        return generate_report(self)

    def _reporting(self):
        """Load displays needed for report.

        Returns
        -------
        displays : List of :class:`~matplotlib.figure.Figure`
            A list of all displays to be rendered.
            Returns None when masker is not fitted
        """
        # Handle the edge case where this function is
        # called with a masker having report capabilities disabled
        if not self._has_report_data():
            self._report_content["overlay"] = None
            return None

        return self._create_figure_for_report()

    def _create_figure_for_report(self):
        """Generate figure to include in the report.

        Returns
        -------
        list of :class:`~matplotlib.figure.Figure` or None
        """
        if not is_matplotlib_installed():
            self._report_content["overlay"] = None
            return None

        import matplotlib.pyplot as plt

        from nilearn.plotting import plot_img

        img = self._reporting_data["images"]
        mask = self._reporting_data["mask"]

        if img is None:  # images were not provided to fit
            img = mask

        # create display of retained input mask, image
        # for visual comparison
        init_display = plot_img(
            img,
            black_bg=False,
            cmap=self.cmap,
        )
        plt.close()

        if mask is not None:
            init_display.add_contours(
                mask,
                levels=[0.5],
                colors="g",
                linewidths=2.5,
            )

        overlay = None
        resampled_img = None
        resampled_mask = None
        # if resampling was performed
        if "transform" in self._reporting_data:
            self._report_content["description"] += (
                "\n To see the input Nifti image before resampling, "
                "hover over the displayed image."
            )

            # create display of resampled NiftiImage and mask
            resampled_img, resampled_mask = self._reporting_data["transform"]
            if resampled_img is None:  # images were not provided to fit
                resampled_img = resampled_mask

            overlay = plot_img(
                resampled_img,
                black_bg=False,
                cmap=self.cmap,
            )
            plt.close()
            overlay.add_contours(
                resampled_mask,
                levels=[0.5],
                colors="g",
                linewidths=2.5,
            )

        self._report_content["overlay"] = overlay

        return init_display

    def __sklearn_is_fitted__(self):
        return hasattr(self, "mask_img_")

    @fill_doc
    def fit(self, imgs=None, y=None):
        """Compute the mask corresponding to the data.

        Parameters
        ----------
        imgs : :obj:`list` of Niimg-like objects or None, default=None
            See :ref:`extracting_data`.
            Data on which the mask must be calculated. If this is a list,
            the affine is considered the same for all.

        %(y_dummy)s
        """
        del y
        check_params(self.__dict__)

        # Reset warning message
        # in case where the masker was previously fitted
        self._report_content["warning_messages"] = []

        self.clean_args_ = {} if self.clean_args is None else self.clean_args

        self._fit_cache()

        # Load data (if filenames are given, load them)
        mask_logger("load_data", img=imgs, verbose=self.verbose)

        self.mask_img_ = self._load_mask(imgs)

        # Compute the mask if not given by the user
        if self.mask_img_ is None:
            if imgs is None:
                raise ValueError(
                    "Parameter 'imgs' must be provided to "
                    f"{self.__class__.__name__}.fit() "
                    "if no mask is passed to mask_img."
                )

            mask_logger("compute_mask", verbose=self.verbose)

            compute_mask = _get_mask_strategy(self.mask_strategy)

            # add extra argument to pass
            # to the mask computing function
            # depending if they are supported.
            sig = dict(**inspect.signature(compute_mask).parameters)
            mask_args = {}
            if self.mask_args:
                skipped_args = []
                for arg in self.mask_args:
                    if arg in sig:
                        mask_args[arg] = self.mask_args.get(arg)
                    else:
                        skipped_args.append(arg)
                if skipped_args:
                    warnings.warn(
                        (
                            "The following arguments are not supported by"
                            f"the masking strategy '{self.mask_strategy}': "
                            f"{skipped_args}"
                        ),
                        UserWarning,
                        stacklevel=find_stack_level(),
                    )

            verbose = self.verbose
            if verbose:
                verbose = 1
            elif not verbose:
                verbose = 0

            self.mask_img_ = self._cache(compute_mask, ignore=["verbose"])(
                imgs, verbose=max(0, self.verbose - 1), **mask_args
            )
        elif imgs is not None:
            warnings.warn(
                f"[{self.__class__.__name__}.fit] "
                "Generation of a mask has been requested (imgs != None) "
                "while a mask was given at masker creation. "
                "Given mask will be used.",
                stacklevel=find_stack_level(),
            )

        self._report_content["reports_at_fit_time"] = self.reports
        if self.reports:  # save inputs for reporting
            self._reporting_data = {
                "mask": self.mask_img_,
                "dim": None,
                "images": imgs,
            }
            if imgs is not None:
                imgs, dims = compute_middle_image(imgs)
                self._reporting_data["images"] = imgs
                self._reporting_data["dim"] = dims

        # TODO add if block to only run when resampling is needed
        # If resampling is requested, resample also the mask
        # Resampling: allows the user to change the affine, the shape or both
        mask_logger("resample_mask", verbose=self.verbose)

        self.mask_img_ = self._cache(resample_img)(
            self.mask_img_,
            target_affine=self.target_affine,
            target_shape=self.target_shape,
            copy=False,
            interpolation="nearest",
        )

        if self.target_affine is not None:  # resample image to target affine
            self.affine_ = self.target_affine
        else:  # resample image to mask affine
            self.affine_ = self.mask_img_.affine

        # Load data in memory, while also checking that mask is binary/valid
        data, _ = load_mask_img(self.mask_img_, allow_empty=False)

        # Infer the number of elements (voxels) in the mask
        self.n_elements_ = int(data.sum())
        self._report_content["n_elements"] = self.n_elements_
        self._report_content["coverage"] = (
            self.n_elements_ / np.prod(data.shape) * 100
        )

        if (self.target_shape is not None) or (
            (self.target_affine is not None) and self.reports
        ):
            if imgs is not None:
                resampl_imgs = self._cache(resample_img)(
                    imgs,
                    target_affine=self.affine_,
                    copy=False,
                    interpolation="nearest",
                )
                resampl_imgs, _ = compute_middle_image(resampl_imgs)
            else:  # imgs not provided to fit
                resampl_imgs = None

            self._reporting_data["transform"] = [resampl_imgs, self.mask_img_]

        mask_logger("fit_done", verbose=self.verbose)

        return self

    @fill_doc
    def transform_single_imgs(
        self,
        imgs,
        confounds=None,
        sample_mask=None,
        copy=True,
    ):
        """Apply mask, spatial and temporal preprocessing.

        Parameters
        ----------
        imgs : 3D/4D Niimg-like object
            See :ref:`extracting_data`.
            Images to process.

        %(confounds)s

        %(sample_mask)s

        copy : :obj:`bool`, default=True
            Indicates whether a copy is returned or not.

        Returns
        -------
        %(signals_transform_nifti)s

        """
        check_is_fitted(self)

        # Ignore the mask-computing params: they are not useful and will
        # just invalid the cache for no good reason
        # target_shape and target_affine are conveyed implicitly in mask_img
        params = get_params(
            self.__class__,
            self,
            ignore=[
                "mask_img",
                "mask_args",
                "mask_strategy",
                "_sample_mask",
                "sample_mask",
            ],
        )
        params["clean_kwargs"] = self.clean_args_

        sklearn_output_config = getattr(self, "_sklearn_output_config", None)

        data = self._cache(
            filter_and_mask,
            ignore=[
                "verbose",
                "memory",
                "memory_level",
                "copy",
            ],
            shelve=self._shelving,
        )(
            imgs,
            self.mask_img_,
            params,
            memory_level=self.memory_level,
            memory=self.memory_,
            verbose=self.verbose,
            confounds=confounds,
            sample_mask=sample_mask,
            copy=copy,
            dtype=self.dtype,
            sklearn_output_config=sklearn_output_config,
        )

        return data

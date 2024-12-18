"""Transformer used to apply basic transformations on MRI data."""

# Author: Gael Varoquaux, Alexandre Abraham

import warnings
from copy import copy as copy_object
from functools import partial

from joblib import Memory

from nilearn import _utils, image, masking
from nilearn._utils import logger
from nilearn._utils.helpers import is_matplotlib_installed
from nilearn.maskers._utils import compute_middle_image
from nilearn.maskers.base_masker import BaseMasker, _filter_and_extract


class _ExtractionFunctor:
    func_name = "nifti_masker_extractor"

    def __init__(self, mask_img_):
        self.mask_img_ = mask_img_

    def __call__(self, imgs):
        return (
            masking.apply_mask(
                imgs,
                self.mask_img_,
                dtype=_utils.niimg.img_data_dtype(imgs),
            ),
            imgs.affine,
        )


def _get_mask_strategy(strategy):
    """Return the mask computing method based on a provided strategy."""
    if strategy == "background":
        return masking.compute_background_mask
    elif strategy == "epi":
        return masking.compute_epi_mask
    elif strategy == "whole-brain-template":
        return partial(masking.compute_brain_mask, mask_type="whole-brain")
    elif strategy == "gm-template":
        return partial(masking.compute_brain_mask, mask_type="gm")
    elif strategy == "wm-template":
        return partial(masking.compute_brain_mask, mask_type="wm")
    elif strategy == "template":
        warnings.warn(
            "Masking strategy 'template' is deprecated."
            "Please use 'whole-brain-template' instead."
        )
        return partial(masking.compute_brain_mask, mask_type="whole-brain")
    else:
        raise ValueError(
            f"Unknown value of mask_strategy '{strategy}'. "
            "Acceptable values are 'background', "
            "'epi', 'whole-brain-template', "
            "'gm-template', and "
            "'wm-template'."
        )


def _filter_and_mask(
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
    temp_imgs = _utils.check_niimg(imgs)

    # Raise warning if a 3D niimg is provided.
    if temp_imgs.ndim == 3:
        warnings.warn(
            "Starting in version 0.12, 3D images will be transformed to "
            "1D arrays. "
            "Until then, 3D images will be coerced to 2D arrays, with a "
            "singleton first dimension representing time.",
            DeprecationWarning,
        )

    imgs = _utils.check_niimg(imgs, atleast_4d=True, ensure_ndim=4)

    # Check whether resampling is truly necessary. If so, crop mask
    # as small as possible in order to speed up the process

    if not _utils.niimg_conversions.check_same_fov(imgs, mask_img_):
        warnings.warn(
            "imgs are being resampled to the mask_img resolution. "
            "This process is memory intensive. You might want to provide "
            "a target_affine that is equal to the affine of the imgs "
            "or resample the mask beforehand "
            "to save memory and computation time.",
            UserWarning,
        )
        parameters = copy_object(parameters)
        # now we can crop
        mask_img_ = image.crop_img(mask_img_, copy=False, copy_header=True)
        parameters["target_shape"] = mask_img_.shape
        parameters["target_affine"] = mask_img_.affine

    data, affine = _filter_and_extract(
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
    return data


@_utils.fill_doc
class NiftiMasker(BaseMasker):
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

    %(standardize_maskers)s

    %(standardize_confounds)s

    high_variance_confounds : :obj:`bool`, default=False
        If True, high variance confounds are computed on provided image with
        :func:`nilearn.image.high_variance_confounds` and default parameters
        and regressed out.

    %(detrend)s

    %(low_pass)s

    %(high_pass)s

    %(t_r)s

    target_affine : 3x3 or 4x4 :obj:`numpy.ndarray`, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    target_shape : 3-:obj:`tuple` of :obj:`int`, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

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

    dtype : {dtype, "auto"}, optional
        Data type toward which the data should be converted. If "auto", the
        data will be converted to int32 if dtype is discrete and float32 if it
        is continuous.

    %(memory)s

    %(memory_level1)s

    %(verbose0)s

    reports : :obj:`bool`, default=True
        If set to True, data is saved in order to produce a report.

    %(cmap)s
        default="CMRmap_r"
        Only relevant for the report figures.

    %(masker_kwargs)s

    Attributes
    ----------
    mask_img_ : :obj:`nibabel.nifti1.Nifti1Image`
        The mask of the data, or the computed one.

    affine_ : 4x4 :obj:`numpy.ndarray`
        Affine of the transformed image.

    n_elements_ : :obj:`int`
        The number of voxels in the mask.

        .. versionadded:: 0.9.2

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
        memory_level=1,
        memory=None,
        verbose=0,
        reports=True,
        cmap="CMRmap_r",
        **kwargs,
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
        self.clean_kwargs = kwargs

    def generate_report(self):
        """Generate a report of the masker."""
        if not is_matplotlib_installed():
            with warnings.catch_warnings():
                mpl_unavail_msg = (
                    "Matplotlib is not imported! "
                    "No reports will be generated."
                )
                warnings.filterwarnings("always", message=mpl_unavail_msg)
                warnings.warn(category=ImportWarning, message=mpl_unavail_msg)
                return [None]

        from nilearn.reporting.html_report import generate_report

        return generate_report(self)

    def _reporting(self):
        """Load displays needed for report.

        Returns
        -------
        displays : list
            A list of all displays to be rendered.

        """
        import matplotlib.pyplot as plt

        from nilearn import plotting

        # Handle the edge case where this function is
        # called with a masker having report capabilities disabled
        if self._reporting_data is None:
            return [None]

        img = self._reporting_data["images"]
        mask = self._reporting_data["mask"]

        if img is None:  # images were not provided to fit
            msg = (
                "No image provided to fit in NiftiMasker. "
                "Setting image to mask for reporting."
            )
            warnings.warn(msg, stacklevel=6)
            self._report_content["warning_message"] = msg
            img = mask
        if self._reporting_data["dim"] == 5:
            msg = (
                "A list of 4D subject images were provided to fit. "
                "Only first subject is shown in the report."
            )
            warnings.warn(msg, stacklevel=6)
            self._report_content["warning_message"] = msg
        # create display of retained input mask, image
        # for visual comparison
        init_display = plotting.plot_img(
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

        if "transform" not in self._reporting_data:
            return [init_display]

        # if resampling was performed
        self._report_content["description"] += self._overlay_text

        # create display of resampled NiftiImage and mask
        resampl_img, resampl_mask = self._reporting_data["transform"]
        if resampl_img is None:  # images were not provided to fit
            resampl_img = resampl_mask

        final_display = plotting.plot_img(
            resampl_img,
            black_bg=False,
            cmap=self.cmap,
        )
        plt.close()
        final_display.add_contours(
            resampl_mask,
            levels=[0.5],
            colors="g",
            linewidths=2.5,
        )

        return [init_display, final_display]

    def _check_fitted(self):
        if not hasattr(self, "mask_img_"):
            raise ValueError(
                f"It seems that {self.__class__.__name__} has not been "
                "fitted. "
                "You must call fit() before calling transform()."
            )

    def fit(
        self,
        imgs=None,
        y=None,  # noqa: ARG002
    ):
        """Compute the mask corresponding to the data.

        Parameters
        ----------
        imgs : :obj:`list` of Niimg-like objects
            See :ref:`extracting_data`.
            Data on which the mask must be calculated. If this is a list,
            the affine is considered the same for all.

        y : None
            This parameter is unused. It is solely included for scikit-learn
            compatibility.

        """
        self._report_content = {
            "description": (
                "This report shows the input Nifti image overlaid "
                "with the outlines of the mask (in green). We "
                "recommend to inspect the report for the overlap "
                "between the mask and its input image. "
            ),
            "warning_message": None,
        }
        self._overlay_text = (
            "\n To see the input Nifti image before resampling, "
            "hover over the displayed image."
        )

        if getattr(self, "_shelving", None) is None:
            self._shelving = False

        if self.memory is None:
            self.memory = Memory(location=None)

        self.clean_kwargs = {
            k[7:]: v
            for k, v in self.clean_kwargs.items()
            if k.startswith("clean__")
        }

        # Load data (if filenames are given, load them)
        logger.log(
            f"Loading data from {_utils.repr_niimgs(imgs, shorten=False)}",
            verbose=self.verbose,
        )

        # Compute the mask if not given by the user
        if self.mask_img is None:
            if imgs is None:
                raise ValueError(
                    "Parameter 'imgs' must be provided to "
                    f"{self.__class__.__name__}.fit() "
                    "if no mask is passed to mask_img."
                )
            mask_args = self.mask_args if self.mask_args is not None else {}
            compute_mask = _get_mask_strategy(self.mask_strategy)

            logger.log("Computing the mask", verbose=self.verbose)
            self.mask_img_ = self._cache(compute_mask, ignore=["verbose"])(
                imgs, verbose=max(0, self.verbose - 1), **mask_args
            )
        else:
            self.mask_img_ = _utils.check_niimg_3d(self.mask_img)

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
        else:
            self._reporting_data = None

        # If resampling is requested, resample also the mask
        # Resampling: allows the user to change the affine, the shape or both
        logger.log("Resampling mask", verbose=self.verbose)

        # TODO switch to force_resample=True
        # when bumping to version > 0.13
        self.mask_img_ = self._cache(image.resample_img)(
            self.mask_img_,
            target_affine=self.target_affine,
            target_shape=self.target_shape,
            copy=False,
            interpolation="nearest",
            copy_header=True,
            force_resample=False,
        )

        if self.target_affine is not None:  # resample image to target affine
            self.affine_ = self.target_affine
        else:  # resample image to mask affine
            self.affine_ = self.mask_img_.affine

        # Load data in memory, while also checking that mask is binary/valid
        data, _ = masking.load_mask_img(self.mask_img_, allow_empty=True)

        # Infer the number of elements (voxels) in the mask
        self.n_elements_ = int(data.sum())

        logger.log("Finished fit", verbose=self.verbose)

        if (self.target_shape is not None) or (
            (self.target_affine is not None) and self.reports
        ):
            if imgs is not None:
                # TODO switch to force_resample=True
                # when bumping to version > 0.13
                resampl_imgs = self._cache(image.resample_img)(
                    imgs,
                    target_affine=self.affine_,
                    copy=False,
                    interpolation="nearest",
                    copy_header=True,
                    force_resample=False,
                )
                resampl_imgs, _ = compute_middle_image(resampl_imgs)
            else:  # imgs not provided to fit
                resampl_imgs = None

            self._reporting_data["transform"] = [resampl_imgs, self.mask_img_]

        return self

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
            If a 3D niimg is provided, a singleton dimension will be added to
            the output to represent the single scan in the niimg.

        confounds : CSV file or array-like or :obj:`pandas.DataFrame`, optional
            This parameter is passed to signal.clean. Please see the related
            documentation for details: :func:`nilearn.signal.clean`.
            shape: (number of scans, number of confounds)

        sample_mask : Any type compatible with numpy-array indexing, optional
            shape: (number of scans - number of volumes removed, )
            Masks the niimgs along time/fourth dimension to perform scrubbing
            (remove volumes with high motion) and/or non-steady-state volumes.
            This parameter is passed to signal.clean.

        copy : :obj:`bool`, default=True
            Indicates whether a copy is returned or not.

        Returns
        -------
        region_signals : 2D :obj:`numpy.ndarray`
            Signal for each :term:`voxel` inside the mask.
            shape: (number of scans, number of voxels)

        Warns
        -----
        DeprecationWarning
            If a 3D niimg input is provided, the current behavior
            (adding a singleton dimension to produce a 2D array) is deprecated.
            Starting in version 0.12, a 1D array will be returned for 3D
            inputs.

        """
        # Ignore the mask-computing params: they are not useful and will
        # just invalid the cache for no good reason
        # target_shape and target_affine are conveyed implicitly in mask_img
        params = _utils.class_inspect.get_params(
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
        params["clean_kwargs"] = self.clean_kwargs

        data = self._cache(
            _filter_and_mask,
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
            memory=self.memory,
            verbose=self.verbose,
            confounds=confounds,
            sample_mask=sample_mask,
            copy=copy,
            dtype=self.dtype,
        )

        return data

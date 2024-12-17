"""Transformer used to apply basic transformations \
on multi subject MRI data.
"""

# Author: Gael Varoquaux, Alexandre Abraham

import collections.abc
import itertools
import warnings
from functools import partial

from joblib import Parallel, delayed

from nilearn import image, masking
from nilearn._utils import (
    CacheMixin,
    check_niimg_3d,
    fill_doc,
    logger,
    repr_niimgs,
    stringify_path,
)
from nilearn._utils.class_inspect import (
    get_params,
)
from nilearn._utils.niimg_conversions import iter_check_niimg
from nilearn.maskers._utils import compute_middle_image
from nilearn.maskers.nifti_masker import NiftiMasker, _filter_and_mask


def _get_mask_strategy(strategy):
    """Return the mask computing method based on a provided strategy."""
    if strategy == "background":
        return masking.compute_multi_background_mask
    elif strategy == "epi":
        return masking.compute_multi_epi_mask
    elif strategy == "whole-brain-template":
        return partial(
            masking.compute_multi_brain_mask, mask_type="whole-brain"
        )
    elif strategy == "gm-template":
        return partial(masking.compute_multi_brain_mask, mask_type="gm")
    elif strategy == "wm-template":
        return partial(masking.compute_multi_brain_mask, mask_type="wm")
    elif strategy == "template":
        warnings.warn(
            "Masking strategy 'template' is deprecated. "
            "Please use 'whole-brain-template' instead."
        )
        return partial(
            masking.compute_multi_brain_mask, mask_type="whole-brain"
        )
    else:
        raise ValueError(
            f"Unknown value of mask_strategy '{strategy}'. "
            "Acceptable values are 'background', "
            "'epi', 'whole-brain-template', "
            "'gm-template', and 'wm-template'."
        )


@fill_doc
class MultiNiftiMasker(NiftiMasker, CacheMixin):
    """Applying a mask to extract time-series from multiple Niimg-like objects.

    MultiNiftiMasker is useful when dealing with image sets from multiple
    subjects.

    Use case:
    integrates well with decomposition by MultiPCA and CanICA
    (multi-subject models)

    Parameters
    ----------
    mask_img : Niimg-like object
        See :ref:`extracting_data`.
        Mask of the data. If not given, a mask is computed in the fit step.
        Optional parameters can be set using mask_args and mask_strategy to
        fine tune the mask extraction.

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
            :func:`nilearn.masking.compute_multi_background_mask`,
            :func:`nilearn.masking.compute_multi_epi_mask`, or
            :func:`nilearn.masking.compute_multi_brain_mask`.

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

    %(memory_level)s

    %(n_jobs)s

    %(verbose0)s

    %(masker_kwargs)s

    Attributes
    ----------
    mask_img_ : :obj:`nibabel.nifti1.Nifti1Image`
        The mask of the data.

    affine_ : 4x4 :obj:`numpy.ndarray`
        Affine of the transformed image.

    n_elements_ : :obj:`int`
        The number of voxels in the mask.

        .. versionadded:: 0.9.2

    See Also
    --------
    nilearn.image.resample_img: image resampling
    nilearn.masking.compute_epi_mask: mask computation
    nilearn.masking.apply_mask: mask application on image
    nilearn.signal.clean: confounds removal and general filtering of signals

    """

    def __init__(
        self,
        mask_img=None,
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
        memory_level=0,
        n_jobs=1,
        verbose=0,
        cmap="CMRmap_r",
        **kwargs,
    ):
        super().__init__(
            # Mask is provided or computed
            mask_img=mask_img,
            smoothing_fwhm=smoothing_fwhm,
            standardize=standardize,
            standardize_confounds=standardize_confounds,
            high_variance_confounds=high_variance_confounds,
            detrend=detrend,
            low_pass=low_pass,
            high_pass=high_pass,
            t_r=t_r,
            target_affine=target_affine,
            target_shape=target_shape,
            mask_strategy=mask_strategy,
            mask_args=mask_args,
            dtype=dtype,
            memory=memory,
            memory_level=memory_level,
            verbose=verbose,
            cmap=cmap,
            **kwargs,
        )
        self.n_jobs = n_jobs

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
        if getattr(self, "_shelving", None) is None:
            self._shelving = False

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

        # Load data (if filenames are given, load them)
        logger.log(
            f"Loading data from {repr_niimgs(imgs, shorten=False)}.",
            self.verbose,
        )

        # Compute the mask if not given by the user
        if self.mask_img is None:
            logger.log("Computing mask", self.verbose)

            imgs = stringify_path(imgs)
            if not isinstance(imgs, collections.abc.Iterable) or isinstance(
                imgs, str
            ):
                raise ValueError(
                    f"[{self.__class__.__name__}.fit] "
                    "For multiple processing, you should  provide a list of "
                    "data (e.g. Nifti1Image objects or filenames). "
                    f"{imgs} is an invalid input."
                )

            mask_args = self.mask_args if self.mask_args is not None else {}
            compute_mask = _get_mask_strategy(self.mask_strategy)
            self.mask_img_ = self._cache(
                compute_mask,
                ignore=["n_jobs", "verbose", "memory"],
            )(
                imgs,
                target_affine=self.target_affine,
                target_shape=self.target_shape,
                n_jobs=self.n_jobs,
                memory=self.memory,
                verbose=max(0, self.verbose - 1),
                **mask_args,
            )
        else:
            if imgs is not None:
                warnings.warn(
                    f"[{self.__class__.__name__}.fit] "
                    "Generation of a mask has been requested (imgs != None) "
                    "while a mask has been provided at masker creation. "
                    "Given mask will be used.",
                    stacklevel=2,
                )

            self.mask_img_ = check_niimg_3d(self.mask_img)

        self._reporting_data = None
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

        # If resampling is requested, resample the mask as well.
        # Resampling: allows the user to change the affine, the shape or both.
        logger.log("Resampling mask")

        # TODO switch to force_resample=True
        # when bumping to version > 0.13
        self.mask_img_ = self._cache(image.resample_img)(
            self.mask_img_,
            target_affine=self.target_affine,
            target_shape=self.target_shape,
            interpolation="nearest",
            copy=False,
            copy_header=True,
            force_resample=False,
        )

        if self.target_affine is not None:
            self.affine_ = self.target_affine
        else:
            self.affine_ = self.mask_img_.affine

        # Load data in memory, while also checking that mask is binary/valid
        data, _ = masking.load_mask_img(self.mask_img_, allow_empty=True)

        # Infer the number of elements (voxels) in the mask
        self.n_elements_ = int(data.sum())

        if (self.target_shape is not None) or (
            (self.target_affine is not None) and self.reports
        ):
            resampl_imgs = None
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

            self._reporting_data["transform"] = [resampl_imgs, self.mask_img_]

        return self

    def transform_imgs(
        self, imgs_list, confounds=None, sample_mask=None, copy=True, n_jobs=1
    ):
        """Prepare multi subject data in parallel.

        Parameters
        ----------
        imgs_list : :obj:`list` of Niimg-like objects
            See :ref:`extracting_data`.
            List of imgs file to prepare. One item per subject.

        confounds : :obj:`list` of confounds, optional
            List of confounds (2D arrays or filenames pointing to CSV
            files or pandas DataFrames). Must be of same length than imgs_list.

        sample_mask : :obj:`list` of sample_mask, optional
            List of sample_mask (1D arrays) if scrubbing motion outliers.
            Must be of same length than imgs_list.

                .. versionadded:: 0.8.0

        copy : :obj:`bool`, default=True
            If True, guarantees that output array has no memory in common with
            input array.

        n_jobs : :obj:`int`, default=1
            The number of cpus to use to do the computation. -1 means
            'all cpus'.

        Returns
        -------
        region_signals : :obj:`list` of 2D :obj:`numpy.ndarray`
            List of signal for each element per subject.
            shape: list of (number of scans, number of elements)

        Warns
        -----
        DeprecationWarning
            If a 3D niimg input is provided, the current behavior
            (adding a singleton dimension to produce a 2D array) is deprecated.
            Starting in version 0.12, a 1D array will be returned for 3D
            inputs.

        """
        if not hasattr(self, "mask_img_"):
            raise ValueError(
                f"It seems that {self.__class__.__name__} has not been "
                "fitted. "
                "You must call fit() before calling transform()."
            )
        target_fov = None
        if self.target_affine is None:
            # Force resampling on first image
            target_fov = "first"

        niimg_iter = iter_check_niimg(
            imgs_list,
            ensure_ndim=None,
            atleast_4d=False,
            target_fov=target_fov,
            memory=self.memory,
            memory_level=self.memory_level,
        )

        if confounds is None:
            confounds = itertools.repeat(None, len(imgs_list))

        if sample_mask is None:
            sample_mask = itertools.repeat(None, len(imgs_list))

        # Ignore the mask-computing params: they are not useful and will
        # just invalidate the cache for no good reason
        # target_shape and target_affine are conveyed implicitly in mask_img
        params = get_params(
            self.__class__,
            self,
            ignore=[
                "mask_img",
                "mask_args",
                "mask_strategy",
                "copy",
            ],
        )
        params["clean_kwargs"] = self.clean_kwargs

        func = self._cache(
            _filter_and_mask,
            ignore=[
                "verbose",
                "memory",
                "memory_level",
                "copy",
            ],
            shelve=self._shelving,
        )
        data = Parallel(n_jobs=n_jobs)(
            delayed(func)(
                imgs,
                self.mask_img_,
                params,
                memory_level=self.memory_level,
                memory=self.memory,
                verbose=self.verbose,
                confounds=cfs,
                copy=copy,
                dtype=self.dtype,
                sample_mask=sms,
            )
            for imgs, cfs, sms in zip(niimg_iter, confounds, sample_mask)
        )
        return data

    def transform(self, imgs, confounds=None, sample_mask=None):
        """Apply mask, spatial and temporal preprocessing.

        Parameters
        ----------
        imgs : :obj:`list` of Niimg-like objects
            See :ref:`extracting_data`.
            Data to be preprocessed

        confounds : CSV file or 2D :obj:`numpy.ndarray` or \
                :obj:`pandas.DataFrame`, optional
            This parameter is passed to signal.clean. Please see the
            corresponding documentation for details.

        sample_mask : :obj:`list` of 1D :obj:`numpy.ndarray`, optional
            List of sample_mask (1D arrays) if scrubbing motion outliers.
            Must be of same length than imgs_list.

                .. versionadded:: 0.8.0

        Returns
        -------
        data : :obj:`list` of :obj:`numpy.ndarray`
            preprocessed images

        Warns
        -----
        DeprecationWarning
            If 3D niimg inputs are provided, the current behavior
            (adding a singleton dimension to produce 2D arrays) is deprecated.
            Starting in version 0.12, 1D arrays will be returned for 3D
            inputs.

        """
        self._check_fitted()
        if not hasattr(imgs, "__iter__") or isinstance(imgs, str):
            return self.transform_single_imgs(imgs)

        return self.transform_imgs(
            imgs,
            confounds=confounds,
            sample_mask=sample_mask,
            n_jobs=self.n_jobs,
        )

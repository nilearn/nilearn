"""Transformer used to apply basic transformations \
on multi subject MRI data.
"""

import collections.abc
import inspect
import itertools
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.utils.estimator_checks import check_is_fitted

from nilearn._utils.class_inspect import get_params
from nilearn._utils.docs import fill_doc
from nilearn._utils.helpers import stringify_path
from nilearn._utils.logger import find_stack_level
from nilearn._utils.niimg_conversions import iter_check_niimg
from nilearn._utils.param_validation import check_params
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn.image import resample_img
from nilearn.maskers._utils import compute_middle_image
from nilearn.maskers.base_masker import (
    mask_logger,
    prepare_confounds_multimaskers,
)
from nilearn.maskers.nifti_masker import (
    NiftiMasker,
    _make_brain_mask_func,
    filter_and_mask,
)
from nilearn.masking import (
    compute_multi_background_mask,
    compute_multi_epi_mask,
    load_mask_img,
)
from nilearn.typing import NiimgLike


def _get_mask_strategy(strategy: str):
    """Return the mask computing method based on a provided strategy."""
    if strategy == "background":
        return compute_multi_background_mask
    elif strategy == "epi":
        return compute_multi_epi_mask
    elif strategy == "whole-brain-template":
        return _make_brain_mask_func("whole-brain", multi=True)
    elif strategy == "gm-template":
        return _make_brain_mask_func("gm", multi=True)
    elif strategy == "wm-template":
        return _make_brain_mask_func("wm", multi=True)
    elif strategy == "template":
        warnings.warn(
            "Masking strategy 'template' is deprecated. "
            "Please use 'whole-brain-template' instead.",
            stacklevel=find_stack_level(),
        )
        return _make_brain_mask_func("whole-brain")
    else:
        raise ValueError(
            f"Unknown value of mask_strategy '{strategy}'. "
            "Acceptable values are 'background', "
            "'epi', 'whole-brain-template', "
            "'gm-template', and 'wm-template'."
        )


@fill_doc
class MultiNiftiMasker(NiftiMasker):
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

    runs : :obj:`numpy.ndarray`, optional
        Add a run level to the preprocessing. Each run will be
        detrended independently. Must be a 1D array of n_samples elements.

    %(smoothing_fwhm)s

    %(standardize_maskers)s

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

    %(dtype)s

    %(memory)s

    %(memory_level)s

    %(n_jobs)s

    %(verbose0)s

    reports : :obj:`bool`, default=True
        If set to True, data is saved in order to produce a report.

    %(cmap)s
        default="gray"
        Only relevant for the report figures.

    %(clean_args)s

    %(masker_kwargs)s

    Attributes
    ----------
    affine_ : 4x4 :obj:`numpy.ndarray`
        Affine of the transformed image.

    %(clean_args_)s

    %(masker_kwargs_)s

    mask_img_ : A 3D binary :obj:`nibabel.nifti1.Nifti1Image`
        The mask of the data, or the one computed from ``imgs`` passed to fit.
        If a ``mask_img`` is passed at masker construction,
        then ``mask_img_`` is the resulting binarized version of it
        where each voxel is ``True`` if all values across samples
        (for example across timepoints) is finite value different from 0.

    memory_ : joblib memory cache

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
        memory_level=0,
        n_jobs=1,
        verbose=0,
        reports=True,
        cmap="gray",
        clean_args=None,
        **kwargs,  # TODO (nilearn >= 0.13.0) remove
    ):
        super().__init__(
            # Mask is provided or computed
            mask_img=mask_img,
            runs=runs,
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
            reports=reports,
            cmap=cmap,
            clean_args=clean_args,
            # TODO (nilearn >= 0.13.0) remove
            **kwargs,
        )
        self.n_jobs = n_jobs

    def __sklearn_tags__(self):
        """Return estimator tags.

        See the sklearn documentation for more details on tags
        https://scikit-learn.org/1.6/developers/develop.html#estimator-tags
        """
        # TODO (sklearn  >= 1.6.0) remove if block
        if SKLEARN_LT_1_6:
            from nilearn._utils.tags import tags

            return tags(masker=True, multi_masker=True)

        from nilearn._utils.tags import InputTags

        tags = super().__sklearn_tags__()
        tags.input_tags = InputTags(masker=True, multi_masker=True)
        return tags

    @fill_doc
    def fit(
        self,
        imgs=None,
        y=None,
    ):
        """Compute the mask corresponding to the data.

        Parameters
        ----------
        imgs : Niimg-like objects, :obj:`list` of Niimg-like objects or None, \
            default=None
            See :ref:`extracting_data`.
            Data on which the mask must be calculated.
            If this is a list, the affine is considered the same for all.

        %(y_dummy)s

        """
        del y
        check_params(self.__dict__)

        self._report_content = {
            "description": (
                "This report shows the input Nifti image overlaid "
                "with the outlines of the mask (in green). We "
                "recommend to inspect the report for the overlap "
                "between the mask and its input image. "
            ),
            "warning_message": None,
            "n_elements": 0,
            "coverage": 0,
        }
        self._overlay_text = (
            "\n To see the input Nifti image before resampling, "
            "hover over the displayed image."
        )

        self._sanitize_cleaning_parameters()
        self.clean_args_ = {} if self.clean_args is None else self.clean_args

        self.mask_img_ = self._load_mask(imgs)

        self._fit_cache()

        mask_logger("load_data", img=imgs, verbose=self.verbose)

        # Compute the mask if not given by the user
        if self.mask_img_ is None:
            if imgs is None:
                raise ValueError(
                    "Parameter 'imgs' must be provided to "
                    f"{self.__class__.__name__}.fit() "
                    "if no mask is passed to mask_img."
                )

            mask_logger("compute_mask", verbose=self.verbose)

            imgs = stringify_path(imgs)
            if not isinstance(imgs, collections.abc.Iterable) or isinstance(
                imgs, str
            ):
                imgs = [imgs]

            compute_mask = _get_mask_strategy(self.mask_strategy)

            # add extra argument to pass
            # to the mask computing function
            # depending if they are supported.
            signature = dict(**inspect.signature(compute_mask).parameters)
            mask_args = {}
            for arg in ["n_jobs", "target_shape", "target_affine"]:
                if arg in signature and getattr(self, arg) is not None:
                    mask_args[arg] = getattr(self, arg)
            if self.mask_args:
                skipped_args = []
                for arg in self.mask_args:
                    if arg in signature:
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

            self.mask_img_ = self._cache(compute_mask, ignore=["verbose"])(
                imgs,
                memory=self.memory_,
                verbose=max(0, self.verbose - 1),
                **mask_args,
            )
        elif imgs is not None:
            warnings.warn(
                f"[{self.__class__.__name__}.fit] "
                "Generation of a mask has been requested (imgs != None) "
                "while a mask was given at masker creation. "
                "Given mask will be used.",
                stacklevel=find_stack_level(),
            )

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

        # TODO add if block to only run when resampling is needed
        # If resampling is requested, resample the mask as well.
        # Resampling: allows the user to change the affine, the shape or both.
        mask_logger("resample_mask", verbose=self.verbose)

        # TODO (nilearn >= 0.13.0) force_resample=True
        self.mask_img_ = self._cache(resample_img)(
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
        data, _ = load_mask_img(self.mask_img_, allow_empty=True)

        # Infer the number of elements (voxels) in the mask
        self.n_elements_ = int(data.sum())
        self._report_content["n_elements"] = self.n_elements_
        self._report_content["coverage"] = (
            self.n_elements_ / np.prod(data.shape) * 100
        )

        if (self.target_shape is not None) or (
            (self.target_affine is not None) and self.reports
        ):
            resampl_imgs = None
            if imgs is not None:
                # TODO (nilearn >= 0.13.0) force_resample=True
                resampl_imgs = self._cache(resample_img)(
                    imgs,
                    target_affine=self.affine_,
                    copy=False,
                    interpolation="nearest",
                    copy_header=True,
                    force_resample=False,
                )

            self._reporting_data["transform"] = [resampl_imgs, self.mask_img_]

        mask_logger("fit_done", verbose=self.verbose)

        return self

    @fill_doc
    def transform_imgs(
        self, imgs_list, confounds=None, sample_mask=None, copy=True, n_jobs=1
    ):
        """Prepare multi subject data in parallel.

        Parameters
        ----------
        %(imgs)s
            Images to process.

        %(confounds_multi)s

        %(sample_mask_multi)s

            .. versionadded:: 0.8.0

        copy : :obj:`bool`, default=True
            If True, guarantees that output array has no memory in common with
            input array.

        %(n_jobs)s

        Returns
        -------
        %(signals_transform_imgs_multi_nifti)s

        """
        check_is_fitted(self)

        target_fov = "first" if self.target_affine is None else None
        niimg_iter = iter_check_niimg(
            imgs_list,
            ensure_ndim=None,
            atleast_4d=False,
            target_fov=target_fov,
            memory=self.memory_,
            memory_level=self.memory_level,
        )

        confounds = prepare_confounds_multimaskers(self, imgs_list, confounds)

        if sample_mask is None:
            sample_mask = itertools.repeat(None, len(imgs_list))
        elif len(sample_mask) != len(imgs_list):
            raise ValueError(
                f"number of sample_mask ({len(sample_mask)}) unequal to "
                f"number of images ({len(imgs_list)})."
            )

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
        params["clean_kwargs"] = self.clean_args_
        # TODO (nilearn  >= 0.13.0) remove
        if self.clean_kwargs:
            params["clean_kwargs"] = self.clean_kwargs_

        func = self._cache(
            filter_and_mask,
            ignore=[
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
                memory=self.memory_,
                verbose=self.verbose,
                confounds=cfs,
                copy=copy,
                dtype=self.dtype,
                sample_mask=sms,
            )
            for imgs, cfs, sms in zip(niimg_iter, confounds, sample_mask)
        )
        return data

    @fill_doc
    def transform(self, imgs, confounds=None, sample_mask=None):
        """Apply mask, spatial and temporal preprocessing.

        Parameters
        ----------
        imgs : Niimg-like object, or a :obj:`list` of Niimg-like objects
            See :ref:`extracting_data`.
            Data to be preprocessed

        %(confounds_multi)s

        %(sample_mask_multi)s

            .. versionadded:: 0.8.0

        Returns
        -------
        %(signals_transform_multi_nifti)s

        """
        check_is_fitted(self)

        if not (confounds is None or isinstance(confounds, list)):
            raise TypeError(
                "'confounds' must be a None or a list. "
                f"Got {confounds.__class__.__name__}."
            )
        if not (sample_mask is None or isinstance(sample_mask, list)):
            raise TypeError(
                "'sample_mask' must be a None or a list. "
                f"Got {sample_mask.__class__.__name__}."
            )
        if isinstance(imgs, NiimgLike):
            if isinstance(confounds, list):
                confounds = confounds[0]
            if isinstance(sample_mask, list):
                sample_mask = sample_mask[0]
            return super().transform(
                imgs, confounds=confounds, sample_mask=sample_mask
            )

        return self.transform_imgs(
            imgs,
            confounds=confounds,
            sample_mask=sample_mask,
            n_jobs=self.n_jobs,
        )

    @fill_doc
    def fit_transform(self, imgs, y=None, confounds=None, sample_mask=None):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        imgs : Niimg-like object, or a :obj:`list` of Niimg-like objects
            See :ref:`extracting_data`.
            Data to be preprocessed

        y : None
            This parameter is unused. It is solely included for scikit-learn
            compatibility.

        %(confounds_multi)s

        %(sample_mask_multi)s

            .. versionadded:: 0.8.0

        Returns
        -------
        %(signals_transform_multi_nifti)s
        """
        return self.fit(imgs, y=y).transform(
            imgs, confounds=confounds, sample_mask=sample_mask
        )

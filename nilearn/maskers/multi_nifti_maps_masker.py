"""Transformer for computing ROI signals of multiple 4D images."""

import itertools

from joblib import Parallel, delayed
from sklearn.utils.estimator_checks import check_is_fitted

from nilearn._utils.docs import fill_doc
from nilearn._utils.niimg_conversions import iter_check_niimg
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn.maskers.base_masker import prepare_confounds_multimaskers
from nilearn.maskers.nifti_maps_masker import NiftiMapsMasker
from nilearn.typing import NiimgLike


@fill_doc
class MultiNiftiMapsMasker(NiftiMapsMasker):
    """Class for extracting data from multiple Niimg-like objects \
       using maps of potentially overlapping brain regions.

    MultiNiftiMapsMasker is useful when data from overlapping volumes
    and from different subjects should be extracted (contrary to
    :class:`nilearn.maskers.NiftiMapsMasker`).

    Use case:
    summarize brain signals from several subjects
    from large-scale networks obtained by prior PCA or :term:`ICA`.

    .. note::
        Inf or NaN present in the given input images are automatically
        put to zero rather than considered as missing data.

    For more details on the definitions of maps in Nilearn,
    see the :ref:`region` section.

    Parameters
    ----------
    maps_img : 4D niimg-like object or None, default=None
        See :ref:`extracting_data`.
        Set of continuous maps. One representative time course per map is
        extracted using least square regression.

    mask_img : 3D niimg-like object, optional
        See :ref:`extracting_data`.
        Mask to apply to regions before extracting signals.

    allow_overlap : :obj:`bool`, default=True
        If False, an error is raised if the maps overlaps (ie at least two
        maps have a non-zero value for the same voxel).

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

    %(dtype)s

    resampling_target : {"data", "mask", "maps", None}, default="data"
        Defines which image gives the final shape/size:

        - ``"data"`` means that the atlas is resampled
          to the shape of the data if needed
        - ``"mask"`` means that the ``maps_img`` and images provided
          to ``fit()`` are
          resampled to the shape and affine of ``mask_img``
        - ``"maps"`` means the ``mask_img`` and images provided
          to ``fit()`` are
          resampled to the shape and affine of ``maps_img``
        - ``None`` means no resampling: if shapes and affines do not match,
          a :obj:`ValueError` is raised.

    %(keep_masked_maps)s

    %(memory)s

    %(memory_level)s

    %(verbose0)s

    reports : :obj:`bool`, default=True
        If set to True, data is saved in order to produce a report.

    %(cmap)s
        default="CMRmap_r"
        Only relevant for the report figures.

    %(n_jobs)s

    %(clean_args)s

    %(masker_kwargs)s

    Attributes
    ----------
    %(clean_args_)s

    %(masker_kwargs_)s

    maps_img_ : :obj:`nibabel.nifti1.Nifti1Image`
        The maps mask of the data.

    %(nifti_mask_img_)s

    memory_ : joblib memory cache

    n_elements_ : :obj:`int`
        The number of overlapping maps in the mask.
        This is equivalent to the number of volumes in the mask image.

        .. versionadded:: 0.9.2

    Notes
    -----
    If resampling_target is set to "maps", every 3D image processed by
    transform() will be resampled to the shape of maps_img. It may lead to a
    very large memory consumption if the voxel number in maps_img is large.

    See Also
    --------
    nilearn.maskers.NiftiMasker
    nilearn.maskers.NiftiLabelsMasker
    nilearn.maskers.NiftiMapsMasker

    """

    # memory and memory_level are used by CacheMixin.

    def __init__(
        self,
        maps_img=None,
        mask_img=None,
        allow_overlap=True,
        smoothing_fwhm=None,
        standardize=False,
        standardize_confounds=True,
        high_variance_confounds=False,
        detrend=False,
        low_pass=None,
        high_pass=None,
        t_r=None,
        dtype=None,
        resampling_target="data",
        keep_masked_maps=True,
        memory=None,
        memory_level=0,
        verbose=0,
        reports=True,
        cmap="CMRmap_r",
        n_jobs=1,
        clean_args=None,
        **kwargs,
    ):
        self.n_jobs = n_jobs
        super().__init__(
            maps_img,
            mask_img=mask_img,
            allow_overlap=allow_overlap,
            smoothing_fwhm=smoothing_fwhm,
            standardize=standardize,
            standardize_confounds=standardize_confounds,
            high_variance_confounds=high_variance_confounds,
            detrend=detrend,
            low_pass=low_pass,
            high_pass=high_pass,
            t_r=t_r,
            dtype=dtype,
            resampling_target=resampling_target,
            memory=memory,
            memory_level=memory_level,
            verbose=verbose,
            keep_masked_maps=keep_masked_maps,
            reports=reports,
            cmap=cmap,
            clean_args=clean_args,
            **kwargs,
        )

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
    def transform_imgs(
        self, imgs_list, confounds=None, n_jobs=1, sample_mask=None
    ):
        """Extract signals from a list of 4D niimgs.

        Parameters
        ----------
        %(imgs)s
            Images to process.

        %(confounds_multi)s

        %(n_jobs)s

        %(sample_mask_multi)s

        Returns
        -------
        %(signals_transform_imgs_multi_nifti)s

        """
        # We handle the resampling of maps and mask separately because the
        # affine of the maps and mask images should not impact the extraction
        # of the signal.

        check_is_fitted(self)

        niimg_iter = iter_check_niimg(
            imgs_list,
            ensure_ndim=None,
            atleast_4d=False,
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

        func = self._cache(self.transform_single_imgs)

        region_signals = Parallel(n_jobs=n_jobs)(
            delayed(func)(imgs=imgs, confounds=cfs, sample_mask=sms)
            for imgs, cfs, sms in zip(niimg_iter, confounds, sample_mask)
        )
        return region_signals

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

"""Transformer for computing ROI signals of multiple 4D images."""

import itertools

from joblib import Parallel, delayed
from sklearn.utils.estimator_checks import check_is_fitted

from nilearn._utils.docs import fill_doc
from nilearn._utils.niimg_conversions import iter_check_niimg
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn.maskers.base_masker import prepare_confounds_multimaskers
from nilearn.maskers.nifti_labels_masker import NiftiLabelsMasker
from nilearn.typing import NiimgLike


@fill_doc
class MultiNiftiLabelsMasker(NiftiLabelsMasker):
    """Class for extracting data from multiple Niimg-like objects \
       using labels of non-overlapping brain regions.

    MultiNiftiLabelsMasker is useful when data from non-overlapping volumes
    and from different subjects should be extracted (contrary to
    :class:`nilearn.maskers.NiftiLabelsMasker`).

    For more details on the definitions of labels in Nilearn,
    see the :ref:`region` section.

    Parameters
    ----------
    labels_img : Niimg-like object or None, default=None
        See :ref:`extracting_data`.
        Region definitions, as one image of labels.

    labels : :obj:`list` of :obj:`str`, optional
        Full labels corresponding to the labels image. This is used
        to improve reporting quality if provided.

        .. warning::
            The labels must be consistent with the label
            values provided through `labels_img`.

    %(masker_lut)s

    background_label : :obj:`int` or :obj:`float`, default=0
        Label used in labels_img to represent background.

        .. warning::

            This value must be consistent with label values and image provided.

    mask_img : Niimg-like object, optional
        See :ref:`extracting_data`.
        Mask to apply to regions before extracting signals.

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

    resampling_target : {"data", "labels", None}, default="data"
        Defines which image gives the final shape/size:

        - ``"data"`` means the atlas is resampled
          to the shape of the data if needed.
        - ``"labels"`` means that the ``mask_img`` and images provided
          to ``fit()`` are resampled to the shape and affine of ``labels_img``.
        - ``"None"`` means no resampling:
          if shapes and affines do not match, a :obj:`ValueError` is raised.

    %(memory)s

    %(memory_level1)s

    %(verbose0)s

    %(strategy)s

    %(keep_masked_labels)s

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

    labels_img_ : :obj:`nibabel.nifti1.Nifti1Image`
        The labels image.

    lut_ : :obj:`pandas.DataFrame`
        Look-up table derived from the ``labels`` or ``lut``
        or from the values of the label image.

    %(nifti_mask_img_)s

    memory_ : joblib memory cache

    See Also
    --------
    nilearn.maskers.NiftiMasker
    nilearn.maskers.NiftiLabelsMasker

    """

    def __init__(
        self,
        labels_img=None,
        labels=None,
        lut=None,
        background_label=0,
        mask_img=None,
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
        memory=None,
        memory_level=1,
        verbose=0,
        strategy="mean",
        keep_masked_labels=True,
        reports=True,
        cmap="CMRmap_r",
        n_jobs=1,
        clean_args=None,
        **kwargs,
    ):
        self.n_jobs = n_jobs
        super().__init__(
            labels_img,
            labels=labels,
            lut=lut,
            background_label=background_label,
            mask_img=mask_img,
            smoothing_fwhm=smoothing_fwhm,
            standardize=standardize,
            standardize_confounds=standardize_confounds,
            high_variance_confounds=high_variance_confounds,
            low_pass=low_pass,
            high_pass=high_pass,
            detrend=detrend,
            t_r=t_r,
            dtype=dtype,
            resampling_target=resampling_target,
            memory=memory,
            memory_level=memory_level,
            verbose=verbose,
            strategy=strategy,
            reports=reports,
            cmap=cmap,
            clean_args=clean_args,
            keep_masked_labels=keep_masked_labels,
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
        check_is_fitted(self)

        # We handle the resampling of labels separately because the affine of
        # the labels image should not impact the extraction of the signal.

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

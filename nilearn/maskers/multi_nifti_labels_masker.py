"""Transformer for computing ROI signals of multiple 4D images."""

import itertools

from joblib import Memory, Parallel, delayed

from .._utils import fill_doc
from .._utils.niimg_conversions import iter_check_niimg
from .nifti_labels_masker import NiftiLabelsMasker


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
    labels_img : Niimg-like object
        See :ref:`extracting_data`.
        Region definitions, as one image of labels.

    labels : :obj:`list` of :obj:`str`, optional
        Full labels corresponding to the labels image. This is used
        to improve reporting quality if provided.

        .. warning::
            The labels must be consistent with the label
            values provided through `labels_img`.

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
    dtype : {dtype, "auto"}
        Data type toward which the data should be converted. If "auto", the
        data will be converted to int32 if dtype is discrete and float32 if it
        is continuous.

    resampling_target : {"data", "labels", None}, default="data"
        Gives which image gives the final shape/size:

            - "data" means the atlas is resampled to the
              shape of the data if needed
            - "labels" means en mask_img and images provided to fit() are
              resampled to the shape and affine of maps_img
            - None means no resampling: if shapes and affines do not match, a
              ValueError is raised

    %(memory)s
    %(memory_level1)s
    %(n_jobs)s
    %(verbose0)s
    strategy : :obj:`str`, default='mean'
        The name of a valid function to reduce the region with.
        Must be one of: sum, mean, median, minimum, maximum, variance,
        standard_deviation.

    reports : :obj:`bool`, default=True
        If set to True, data is saved in order to produce a report.

    %(masker_kwargs)s

    Attributes
    ----------
    mask_img_ : :obj:`nibabel.nifti1.Nifti1Image`
        The mask of the data, or the computed one.

    labels_img_ : :obj:`nibabel.nifti1.Nifti1Image`
        The labels image.

    n_elements_ : :obj:`int`
        The number of discrete values in the mask.
        This is equivalent to the number of unique values in the mask image,
        ignoring the background value.

        .. versionadded:: 0.9.2

    See Also
    --------
    nilearn.maskers.NiftiMasker
    nilearn.maskers.NiftiLabelsMasker

    """

    def __init__(
        self,
        labels_img,
        labels=None,
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
        reports=True,
        n_jobs=1,
        **kwargs,
    ):
        if memory is None:
            memory = Memory(location=None, verbose=0)
        self.n_jobs = n_jobs
        super().__init__(
            labels_img,
            labels=labels,
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
            **kwargs,
        )

    @fill_doc
    def transform_imgs(
        self, imgs_list, confounds=None, n_jobs=1, sample_mask=None
    ):
        """Extract signals from a list of 4D niimgs.

        Parameters
        ----------
        %(imgs)s
            Images to process. Each element of the list is a 4D image.
        %(confounds)s
        %(sample_mask)s

        Returns
        -------
        region_signals: list of 2D :obj:`numpy.ndarray`
            List of signals for each label per subject.
            shape: list of (number of scans, number of labels)

        """
        # We handle the resampling of labels separately because the affine of
        # the labels image should not impact the extraction of the signal.

        self._check_fitted()
        niimg_iter = iter_check_niimg(
            imgs_list,
            ensure_ndim=None,
            atleast_4d=False,
            memory=self.memory,
            memory_level=self.memory_level,
        )

        if confounds is None:
            confounds = itertools.repeat(None, len(imgs_list))

        if sample_mask is None:
            sample_mask = itertools.repeat(None, len(imgs_list))

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
        %(imgs)s
            Images to process. Each element of the list is a 4D image.
        %(confounds)s
        %(sample_mask)s

        Returns
        -------
        region_signals : list of 2D :obj:`numpy.ndarray`
            List of signals for each label per subject.
            shape: list of (number of scans, number of labels)

        """
        self._check_fitted()
        if not hasattr(imgs, "__iter__") or isinstance(imgs, str):
            return self.transform_single_imgs(imgs)
        return self.transform_imgs(
            imgs, confounds, n_jobs=self.n_jobs, sample_mask=sample_mask
        )

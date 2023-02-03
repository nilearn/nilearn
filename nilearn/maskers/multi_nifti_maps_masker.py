"""
Transformer for computing ROI signals of multiple 4D images
"""

import itertools

from joblib import Memory, Parallel, delayed

from .._utils import fill_doc
from .._utils.niimg_conversions import _iter_check_niimg
from .nifti_maps_masker import NiftiMapsMasker


@fill_doc
class MultiNiftiMapsMasker(NiftiMapsMasker):
    """Class for masking of Niimg-like objects.

    MultiNiftiMapsMasker is useful when data from overlapping volumes
    and from different subjects should be extracted (contrary to
    :class:`nilearn.maskers.NiftiMapsMasker`).

    .. note::
        Inf or NaN present in the given input images are automatically
        put to zero rather than considered as missing data.

    Parameters
    ----------
    maps_img : 4D niimg-like object
        See :ref:`extracting_data`.
        Set of continuous maps. One representative time course per map is
        extracted using least square regression.

    mask_img : 3D niimg-like object, optional
        See :ref:`extracting_data`.
        Mask to apply to regions before extracting signals.

    allow_overlap : :obj:`bool`, optional
        If False, an error is raised if the maps overlaps (ie at least two
        maps have a non-zero value for the same voxel). Default=True.
    %(smoothing_fwhm)s
    standardize : {False, True, 'zscore', 'psc'}, optional
        Strategy to standardize the signal.

            - 'zscore': the signal is z-scored. Timeseries are shifted
              to zero mean and scaled to unit variance.
            - 'psc':  Timeseries are shifted to zero mean value and scaled
              to percent signal change (as compared to original mean signal).
            - True : the signal is z-scored. Timeseries are shifted
              to zero mean and scaled to unit variance.
            - False : Do not standardize the data.

        Default=False.

    %(standardize_confounds)s
    high_variance_confounds : :obj:`bool`, optional
        If True, high variance confounds are computed on provided image with
        :func:`nilearn.image.high_variance_confounds` and default parameters
        and regressed out. Default=False.
    %(detrend)s
    %(low_pass)s
    %(high_pass)s
    %(t_r)s
    dtype : {dtype, "auto"}, optional
        Data type toward which the data should be converted. If "auto", the
        data will be converted to int32 if dtype is discrete and float32 if it
        is continuous.

    resampling_target : {"data", "mask", "maps", None}, optional.
        Gives which image gives the final shape/size:

            - "data" means the atlas is resampled to the shape of the data if
              needed
            - "mask" means the maps_img and images provided to fit() are
              resampled to the shape and affine of mask_img
            - "maps" means the mask_img and images provided to fit() are
              resampled to the shape and affine of maps_img
            - None means no resampling: if shapes and affines do not match,
              a ValueError is raised.

        Default="data".

    %(memory)s
    %(memory_level)s
    %(n_jobs)s
    %(verbose0)s
    reports : :obj:`bool`, optional
        If set to True, data is saved in order to produce a report.
        Default=True.

    %(masker_kwargs)s

    Notes
    -----
    If resampling_target is set to "maps", every 3D image processed by
    transform() will be resampled to the shape of maps_img. It may lead to a
    very large memory consumption if the voxel number in maps_img is large.

    See also
    --------
    nilearn.maskers.NiftiMasker
    nilearn.maskers.NiftiLabelsMasker
    nilearn.maskers.NiftiMapsMasker

    """
    # memory and memory_level are used by CacheMixin.

    def __init__(
        self,
        maps_img,
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
        memory=Memory(location=None, verbose=0),
        memory_level=0,
        verbose=0,
        reports=True,
        n_jobs=1,
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
            reports=reports,
            **kwargs,
        )

    def transform_imgs(self, imgs_list, confounds=None, n_jobs=1,
                       sample_mask=None):
        """Extract signals from a list of 4D niimgs.

        Parameters
        ----------
        %(imgs)s
            Images to process. Each element of the list is a 4D image.
        %(confounds)s
        %(sample_mask)s

        Returns
        -------
        region_signals : list of 2D :obj:`numpy.ndarray`
            List of signals for each map per subject.
            shape: list of (number of scans, number of maps)

        """
        # We handle the resampling of maps and mask separately because the
        # affine of the maps and mask images should not impact the extraction
        # of the signal.

        self._check_fitted()

        niimg_iter = _iter_check_niimg(imgs_list, ensure_ndim=None,
                                       atleast_4d=False,
                                       memory=self.memory,
                                       memory_level=self.memory_level,
                                       verbose=self.verbose)

        if confounds is None:
            confounds = itertools.repeat(None, len(imgs_list))

        func = self._cache(self.transform_single_imgs)

        region_signals = Parallel(n_jobs=n_jobs)(
            delayed(func)(imgs=imgs, confounds=cfs, sample_mask=sample_mask)
            for imgs, cfs in zip(niimg_iter, confounds))
        return region_signals

    def transform(self, imgs, confounds=None, sample_mask=None):
        """ Apply mask, spatial and temporal preprocessing

        Parameters
        ----------
        %(imgs)s
            Images to process. Each element of the list is a 4D image.
        %(confounds)s
        %(sample_mask)s

        Returns
        -------
        region_signals : list of 2D :obj:`numpy.ndarray`
            List of signals for each map per subject.
            shape: list of (number of scans, number of maps)

        """

        self._check_fitted()
        if (not hasattr(imgs, '__iter__')
                or isinstance(imgs, str)):
            return self.transform_single_imgs(imgs)
        return self.transform_imgs(imgs, confounds, n_jobs=self.n_jobs,
                                   sample_mask=sample_mask)

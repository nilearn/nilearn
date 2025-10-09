"""Transformer for computing ROI signals of multiple 4D images."""

from nilearn._utils.docs import fill_doc
from nilearn.maskers._mixin import _MultiMixin
from nilearn.maskers.nifti_maps_masker import NiftiMapsMasker


@fill_doc
class MultiNiftiMapsMasker(_MultiMixin, NiftiMapsMasker):
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

    %(standardize_false)s

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

    Attributes
    ----------
    %(clean_args_)s

    maps_img_ : :obj:`nibabel.nifti1.Nifti1Image`
        The maps mask of the data.

    %(nifti_mask_img_)s

    memory_ : joblib memory cache

    n_elements_ : :obj:`int`
        The number of overlapping maps in the mask.
        This is equivalent to the number of volumes in the mask image.

        .. nilearn_versionadded:: 0.9.2

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
        keep_masked_maps=False,
        memory=None,
        memory_level=0,
        verbose=0,
        reports=True,
        cmap="CMRmap_r",
        n_jobs=1,
        clean_args=None,
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
        )

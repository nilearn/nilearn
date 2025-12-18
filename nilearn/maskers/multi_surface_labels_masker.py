"""Extract data from multiple 2D surface objects."""

from nilearn import DEFAULT_SEQUENTIAL_CMAP
from nilearn._utils.docs import fill_doc
from nilearn._utils.param_validation import check_params
from nilearn.maskers._mixin import _MultiMixin
from nilearn.maskers.surface_labels_masker import SurfaceLabelsMasker
from nilearn.surface.surface import check_surf_img


@fill_doc
class MultiSurfaceLabelsMasker(_MultiMixin, SurfaceLabelsMasker):
    """Extract time-series from multiple SurfaceImage objects.

    MultiSurfaceMasker is useful when dealing with image sets from multiple
    subjects.

    .. versionadded:: 0.13.0dev

    Parameters
    ----------
    labels_img : :obj:`~nilearn.surface.SurfaceImage` object
        Region definitions, as one image of labels.
        The data for each hemisphere
        is of shape (n_vertices_per_hemisphere, n_regions).

    labels : :obj:`list` of :obj:`str`, default=None
        Mutually exclusive with ``lut``.
        Labels corresponding to the labels image.
        This is used to improve reporting quality if provided.

        "Background" can be included in this list of labels
        to denote which values in the image should be considered
        background value.

        .. warning::
            If the labels are not be consistent with the label values
            provided through ``labels_img``,
            excess labels will be dropped,
            and missing labels will be labeled ``'unknown'``.

    %(masker_lut)s

    background_label : :obj:`int` or :obj:`float`, default=0
        Label used in labels_img to represent background.

        .. warning::

            This value must be consistent with label values
            and image provided.

    mask_img : :obj:`~nilearn.surface.SurfaceImage` object, optional
        Mask to apply to labels_img before extracting signals. Defines the \
        overall area of the brain to consider. The data for each \
        hemisphere is of shape (n_vertices_per_hemisphere, n_regions).

    %(smoothing_fwhm)s
        This parameter is not implemented yet.

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

    %(memory)s

    %(memory_level1)s

    %(n_jobs)s

    %(verbose0)s

    %(strategy)s

    reports : :obj:`bool`, default=True
        If set to True, data is saved in order to produce a report.

    %(cmap)s
        default="inferno"
        Only relevant for the report figures.

    %(clean_args)s

    Attributes
    ----------
    %(clean_args_)s

    labels_img_ : :obj:`nibabel.nifti1.Nifti1Image`
        The labels image after fitting.
        If a mask_img was used,
        then masked vertices will have the background value.

    lut_ : :obj:`pandas.DataFrame`
        Look-up table derived from the ``labels`` or ``lut``
        or from the values of the label image.

    mask_img_ : A 1D binary :obj:`~nilearn.surface.SurfaceImage` or None.
        The mask of the data.
        If no ``mask_img`` was passed at masker construction,
        then ``mask_img_`` is ``None``, otherwise
        is the resulting binarized version of ``mask_img``
        where each vertex is ``True`` if all values across samples
        (for example across timepoints) is finite value different from 0.

    memory_ : joblib memory cache

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
        detrend=False,
        high_variance_confounds=False,
        low_pass=None,
        high_pass=None,
        t_r=None,
        memory=None,
        memory_level=1,
        n_jobs=1,
        verbose=0,
        strategy="mean",
        reports=True,
        cmap=DEFAULT_SEQUENTIAL_CMAP,
        clean_args=None,
    ):
        self.labels_img = labels_img
        self.labels = labels
        self.lut = lut
        self.background_label = background_label
        self.mask_img = mask_img
        self.smoothing_fwhm = smoothing_fwhm
        self.standardize = standardize
        self.standardize_confounds = standardize_confounds
        self.high_variance_confounds = high_variance_confounds
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.memory = memory
        self.memory_level = memory_level
        self.verbose = verbose
        self.strategy = strategy
        self.reports = reports
        self.cmap = cmap
        self.clean_args = clean_args
        super().__init__(
            labels_img=labels_img,
            labels=labels,
            lut=lut,
            background_label=background_label,
            mask_img=mask_img,
            smoothing_fwhm=smoothing_fwhm,
            standardize=standardize,
            standardize_confounds=standardize_confounds,
            high_variance_confounds=high_variance_confounds,
            detrend=detrend,
            low_pass=low_pass,
            high_pass=high_pass,
            t_r=t_r,
            memory=memory,
            memory_level=memory_level,
            verbose=verbose,
            strategy=strategy,
            reports=reports,
            cmap=cmap,
            clean_args=clean_args,
        )
        self.n_jobs = n_jobs

    @fill_doc
    def fit(self, imgs=None, y=None):
        """Prepare signal extraction from regions.

        Parameters
        ----------
        imgs : :obj:`~nilearn.surface.SurfaceImage` or \
              :obj:`list` of :obj:`~nilearn.surface.SurfaceImage` or \
              :obj:`tuple` of :obj:`~nilearn.surface.SurfaceImage` or None, \
              default = None
            Mesh and data for both hemispheres.

        %(y_dummy)s

        Returns
        -------
        SurfaceMasker object
        """
        del y
        check_params(self.__dict__)

        # Reset warning message
        # in case where the masker was previously fitted
        self._report_content["warning_messages"] = []

        if imgs is not None:
            self._check_imgs(imgs)
            if not hasattr(imgs, "__iter__"):
                imgs = [imgs]
            for x in imgs:
                check_surf_img(x)

        return self._fit(imgs)

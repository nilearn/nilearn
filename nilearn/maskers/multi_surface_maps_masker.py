"""Extract data from multiple 2D surface objects."""

from nilearn import DEFAULT_SEQUENTIAL_CMAP
from nilearn._utils.docs import fill_doc
from nilearn._utils.param_validation import check_params
from nilearn.maskers._mixin import _MultiMixin
from nilearn.maskers.surface_maps_masker import SurfaceMapsMasker


@fill_doc
class MultiSurfaceMapsMasker(_MultiMixin, SurfaceMapsMasker):
    """Extract time-series from multiple SurfaceImage objects.

    MultiSurfaceMasker is useful when dealing with image sets from multiple
    subjects.

    .. versionadded:: 0.13.0dev

    Parameters
    ----------
    maps_img : :obj:`~nilearn.surface.SurfaceImage`
        Set of maps that define the regions. A representative time course \
        per map is extracted using least square regression. The data for \
        each hemisphere is of shape (n_vertices_per_hemisphere, n_regions).

    mask_img : :obj:`~nilearn.surface.SurfaceImage`, optional, default=None
        Mask to apply to regions before extracting signals. Defines the \
        overall area of the brain to consider. The data for each \
        hemisphere is of shape (n_vertices_per_hemisphere, n_regions).

    allow_overlap : :obj:`bool`, default=True
        If False, an error is raised if the maps overlaps (ie at least two
        maps have a non-zero value for the same voxel).

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

    reports : :obj:`bool`, default=True
        If set to True, data is saved in order to produce a report.

    %(cmap)s
        default="inferno"
        Only relevant for the report figures.

    %(clean_args)s

    Attributes
    ----------
    %(clean_args_)s

    maps_img_ : :obj:`~nilearn.surface.SurfaceImage`
        The same as the input `maps_img`, kept solely for consistency
        across maskers.

    mask_img_ : A 1D binary :obj:`~nilearn.surface.SurfaceImage` or None.
        The mask of the data.
        If no ``mask_img`` was passed at masker construction,
        then ``mask_img_`` is ``None``, otherwise
        is the resulting binarized version of ``mask_img``
        where each vertex is ``True`` if all values across samples
        (for example across timepoints) is finite value different from 0.

    memory_ : joblib memory cache

    n_elements_ : :obj:`int` or None
        number of vertices included in mask

    """

    def __init__(
        self,
        maps_img=None,
        mask_img=None,
        allow_overlap=True,
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
        reports=True,
        cmap=DEFAULT_SEQUENTIAL_CMAP,
        clean_args=None,
    ):
        self.maps_img = maps_img
        self.mask_img = mask_img
        self.allow_overlap = allow_overlap
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
        self.reports = reports
        self.cmap = cmap
        self.clean_args = clean_args
        super().__init__(
            maps_img=maps_img,
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
            memory=memory,
            memory_level=memory_level,
            verbose=verbose,
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

        return self._fit(imgs)

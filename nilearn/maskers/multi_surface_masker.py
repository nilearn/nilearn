"""Extract data from multiple 2D surface objects."""

from nilearn import DEFAULT_SEQUENTIAL_CMAP
from nilearn._utils.docs import fill_doc
from nilearn._utils.param_validation import check_params
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn.maskers import SurfaceMasker
from nilearn.maskers._mixin import _MultiMixin
from nilearn.surface.surface import SurfaceImage


@fill_doc
class MultiSurfaceMasker(_MultiMixin, SurfaceMasker):
    """Extract time-series from multiple SurfaceImage objects.

    MultiSurfaceMasker is useful when dealing with image sets from multiple
    subjects.
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
        memory=None,
        memory_level=1,
        verbose=0,
        reports=True,
        cmap=DEFAULT_SEQUENTIAL_CMAP,
        clean_args=None,
    ):
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
        self.reports = reports
        self.cmap = cmap
        self.clean_args = clean_args
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
            memory=memory,
            memory_level=memory_level,
            verbose=verbose,
            reports=reports,
            cmap=cmap,
            clean_args=clean_args,
        )

    def _more_tags(self):
        """Return estimator tags.

        TODO (sklearn >= 1.6.0) remove
        """
        return self.__sklearn_tags__()

    def __sklearn_tags__(self):
        """Return estimator tags.

        See the sklearn documentation for more details on tags
        https://scikit-learn.org/1.6/developers/develop.html#estimator-tags
        """
        # TODO (sklearn  >= 1.6.0) remove if block
        if SKLEARN_LT_1_6:
            from nilearn._utils.tags import tags

            return tags(surf_img=True, niimg_like=False, masker=True)

        from nilearn._utils.tags import InputTags

        tags = super().__sklearn_tags__()
        tags.input_tags = InputTags(
            surf_img=True, niimg_like=False, masker=True
        )
        return tags

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
        if imgs is not None:
            self._check_imgs(imgs)

            if isinstance(imgs, SurfaceImage) and any(
                x.ndim == 1 for x in imgs.data.parts.values()
            ):
                raise ValueError("should only be SurfaceImage should >=2D.")

        return self._fit(imgs)

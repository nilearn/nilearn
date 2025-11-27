"""Extract data from multiple 2D surface objects."""

import numpy as np
from sklearn.utils.estimator_checks import check_is_fitted

from nilearn import DEFAULT_SEQUENTIAL_CMAP, signal
from nilearn._utils.class_inspect import get_params
from nilearn._utils.docs import fill_doc
from nilearn._utils.masker_validation import (
    check_compatibility_mask_and_images,
)
from nilearn._utils.param_validation import check_params
from nilearn.maskers._mixin import _MultiMixin
from nilearn.maskers.base_masker import mask_logger
from nilearn.maskers.surface_masker import SurfaceMasker
from nilearn.surface.surface import SurfaceImage
from nilearn.surface.utils import check_polymesh_equal


@fill_doc
class MultiSurfaceMasker(_MultiMixin, SurfaceMasker):
    """Extract time-series from multiple SurfaceImage objects.

    MultiSurfaceMasker is useful when dealing with image sets from multiple
    subjects.

    .. versionadded:: 0.13.0dev

    Parameters
    ----------
    mask_img : :obj:`~nilearn.surface.SurfaceImage` or None, default=None

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

    mask_img_ : A 1D binary :obj:`~nilearn.surface.SurfaceImage`
        The mask of the data, or the one computed from ``imgs`` passed to fit.
        If a ``mask_img`` is passed at masker construction,
        then ``mask_img_`` is the resulting binarized version of it
        where each vertex is ``True`` if all values across samples
        (for example across timepoints) is finite value different from 0.

    memory_ : joblib memory cache

    n_elements_ : :obj:`int` or None
        number of vertices included in mask

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
        n_jobs=1,
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

    @fill_doc
    def transform_single_imgs(
        self,
        imgs,
        confounds=None,
        sample_mask=None,
    ):
        """Extract signals from fitted surface object.

        Parameters
        ----------
        imgs : imgs : :obj:`~nilearn.surface.SurfaceImage` object or \
              iterable of :obj:`~nilearn.surface.SurfaceImage`
            Images to process.
            Mesh and data for both hemispheres/parts.

        %(confounds)s

        %(sample_mask)s

        Returns
        -------
        %(signals_transform_surface)s

        """
        check_is_fitted(self)

        check_compatibility_mask_and_images(self.mask_img_, imgs)
        check_polymesh_equal(self.mask_img_.mesh, imgs.mesh)

        if isinstance(imgs, SurfaceImage) and any(
            hemi.ndim > 2 for hemi in imgs.data.parts.values()
        ):
            raise ValueError("should only be SurfaceImage should 1D or 2D.")
        elif hasattr(imgs, "__iter__"):
            for i, x in enumerate(imgs):
                x.data._check_n_samples(1, f"imgs[{i}]")

        if self.reports:
            self._reporting_data["images"] = imgs

        mask_logger("extracting", verbose=self.verbose)

        output = np.empty((1, self.n_elements_))
        if len(imgs.shape) == 2:
            output = np.empty((imgs.shape[1], self.n_elements_))
        for part_name, (start, stop) in self._slices.items():
            mask = self.mask_img_.data.parts[part_name].ravel()
            output[:, start:stop] = imgs.data.parts[part_name][mask].T

        mask_logger("cleaning", verbose=self.verbose)

        parameters = get_params(self.__class__, self, ignore=["mask_img"])

        parameters["clean_args"] = self.clean_args_

        # signal cleaning here
        output = self._cache(signal.clean, func_memory_level=2)(
            output,
            detrend=parameters["detrend"],
            standardize=parameters["standardize"],
            standardize_confounds=parameters["standardize_confounds"],
            t_r=parameters["t_r"],
            low_pass=parameters["low_pass"],
            high_pass=parameters["high_pass"],
            confounds=confounds,
            sample_mask=sample_mask,
            **parameters["clean_args"],
        )

        return output

"""Extract data from a SurfaceImage, using maps of potentially overlapping
brain regions.
"""

import warnings

import numpy as np
from joblib import Memory
from scipy import linalg

from nilearn import signal
from nilearn._utils import fill_doc, logger
from nilearn._utils.cache_mixin import cache
from nilearn._utils.class_inspect import get_params
from nilearn.maskers._utils import (
    check_same_n_vertices,
    check_surface_data_ndims,
    concat_extract_surface_data_parts,
    concatenate_surface_images,
)
from nilearn.maskers.base_masker import _BaseSurfaceMasker
from nilearn.surface import SurfaceImage


@fill_doc
class SurfaceMapsMasker(_BaseSurfaceMasker):
    """Extract data from a SurfaceImage, using maps of potentially overlapping
    brain regions.

    .. versionadded:: 0.11.1

    Parameters
    ----------
    maps_img : :obj:`~nilearn.surface.SurfaceImage`
        Set of maps that define the regions. representative time course \
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

    %(standardize_maskers)s

    %(standardize_confounds)s

    %(detrend)s

    high_variance_confounds : :obj:`bool`, default=False
        If True, high variance confounds are computed on provided image \
        with :func:`nilearn.image.high_variance_confounds` and default \
        parameters and regressed out.

    %(low_pass)s

    %(high_pass)s

    %(t_r)s

    %(memory)s

    %(memory_level1)s

    %(verbose0)s

    reports : :obj:`bool`, default=True
        If set to True, data is saved in order to produce a report.

    %(cmap)s
        default="inferno"
        Only relevant for the report figures.

    clean_args : :obj:`dict` or None, default=None
        Keyword arguments to be passed
        to :func:`nilearn.signal.clean`
        called within the masker.

    Attributes
    ----------
    maps_img_ : :obj:`~nilearn.surface.SurfaceImage`
        The same as the input `maps_img`, kept solely for consistency
        across maskers.

    n_elements_ : :obj:`int`
        The number of regions in the maps image.

    See Also
    --------
        nilearn.maskers.SurfaceMasker
        nilearn.maskers.SurfaceLabelsMasker

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
        verbose=0,
        reports=True,
        cmap="inferno",
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

    def fit(self, img=None, y=None):
        """Prepare signal extraction from regions.

        Parameters
        ----------
        img : :obj:`~nilearn.surface.SurfaceImage` object or None, default=None
            This parameter is currently unused.

        y : None
            This parameter is unused.
            It is solely included for scikit-learn compatibility.

        Returns
        -------
        SurfaceMapsMasker object
        """
        del img, y

        if self.maps_img is None:
            raise ValueError(
                "Please provide a maps_img during initialization. "
                "For example, masker = SurfaceMapsMasker(maps_img=maps_img)"
            )

        logger.log(
            msg=f"loading regions from {self.maps_img.__repr__()}",
            verbose=self.verbose,
        )
        # check maps_img data is 2D
        check_surface_data_ndims(self.maps_img, 2, "maps_img")
        self.maps_img_ = self.maps_img

        self.n_elements_ = self.maps_img.shape[1]

        if self.mask_img is not None:
            logger.log(
                msg=f"loading regions from {self.mask_img.__repr__()}",
                verbose=self.verbose,
            )
            check_same_n_vertices(self.maps_img.mesh, self.mask_img.mesh)
            # squeeze the mask data if it is 2D and has a single column
            for part in self.mask_img.data.parts:
                if (
                    self.mask_img.data.parts[part].ndim == 2
                    and self.mask_img.data.parts[part].shape[1] == 1
                ):
                    self.mask_img.data.parts[part] = np.squeeze(
                        self.mask_img.data.parts[part], axis=1
                    )
            check_surface_data_ndims(self.mask_img, 1, "mask_img")

        self._shelving = False
        # content to inject in the HTML template
        self._report_content = (
            {
                "description": (
                    "This report shows the input surface image "
                    "(if provided via img) overlaid with the regions provided "
                    "via maps_img."
                ),
                "n_vertices": {},
                "number_of_regions": 0,
                "summary": {},
            },
        )
        return self

    def __sklearn_is_fitted__(self):
        return hasattr(self, "n_elements_")

    def _check_fitted(self):
        if not self.__sklearn_is_fitted__():
            raise ValueError(
                f"It seems that {self.__class__.__name__} "
                "has not been fitted."
            )

    def transform(self, img, confounds=None, sample_mask=None):
        """Extract signals from surface object.

        Parameters
        ----------
        img : :obj:`~nilearn.surface.SurfaceImage` object or \
              :obj:`list` of :obj:`~nilearn.surface.SurfaceImage` or \
              :obj:`tuple` of :obj:`~nilearn.surface.SurfaceImage`
            Mesh and data for both hemispheres/parts. The data for each \
            hemisphere is of shape (n_vertices_per_hemisphere, n_timepoints).

        confounds : :class:`numpy.ndarray`, :obj:`str`,\
                    :class:`pathlib.Path`, \
                    :class:`pandas.DataFrame` \
                    or :obj:`list` of confounds timeseries, default=None
            Confounds to pass to :func:`nilearn.signal.clean`.

        sample_mask : None, Any type compatible with numpy-array indexing, \
                  or :obj:`list` of \
                  shape: (number of scans - number of volumes removed) \
                  for explicit index, or (number of scans) for binary mask, \
                  default=None
            sample_mask to pass to :func:`nilearn.signal.clean`.


        Returns
        -------
        region_signals: :obj:`numpy.ndarray`
            Signal for each region as provided in the maps (via `maps_img`).
            shape: (n_timepoints, n_regions)
        """
        self._check_fitted()

        # if img is a single image, convert it to a list
        # to be able to concatenate it
        if not isinstance(img, list):
            img = [img]
        img = concatenate_surface_images(img)
        # check img data is 2D
        check_surface_data_ndims(img, 2, "img")
        check_same_n_vertices(self.maps_img.mesh, img.mesh)
        img_data = np.concatenate(
            list(img.data.parts.values()), axis=0
        ).astype(np.float32)

        # get concatenated hemispheres/parts data from maps_img and mask_img
        maps_data = concat_extract_surface_data_parts(self.maps_img)
        if self.mask_img is not None:
            mask_data = concat_extract_surface_data_parts(self.mask_img)
        else:
            mask_data = None

        if self.smoothing_fwhm is not None:
            warnings.warn(
                "Parameter smoothing_fwhm "
                "is not yet supported for surface data",
                UserWarning,
                stacklevel=2,
            )
            self.smoothing_fwhm = None

        parameters = get_params(
            self.__class__,
            self,
        )
        if self.clean_args is None:
            self.clean_args = {}
        parameters["clean_args"] = self.clean_args

        if self.memory is None:
            self.memory = Memory(location=None)

        # apply mask if provided
        # and then extract signal via least square regression
        if mask_data is not None:
            region_signals = cache(
                linalg.lstsq,
                memory=self.memory,
                func_memory_level=2,
                memory_level=self.memory_level,
                shelve=self._shelving,
            )(
                maps_data[mask_data.flatten(), :],
                img_data[mask_data.flatten(), :],
            )[0].T
        # if no mask, directly extract signal
        else:
            region_signals = cache(
                linalg.lstsq,
                memory=self.memory,
                func_memory_level=2,
                memory_level=self.memory_level,
                shelve=self._shelving,
            )(maps_data, img_data)[0].T

        # signal cleaning here
        region_signals = cache(
            signal.clean,
            memory=self.memory,
            func_memory_level=2,
            memory_level=self.memory_level,
            shelve=self._shelving,
        )(
            region_signals,
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

        return region_signals

    def fit_transform(self, img, y=None, confounds=None, sample_mask=None):
        """Prepare and perform signal extraction from regions.

        Parameters
        ----------
        img : :obj:`~nilearn.surface.SurfaceImage` object or \
              :obj:`list` of :obj:`~nilearn.surface.SurfaceImage` or \
              :obj:`tuple` of :obj:`~nilearn.surface.SurfaceImage`
            Mesh and data for both hemispheres. The data for each hemisphere \
            is of shape (n_vertices_per_hemisphere, n_timepoints).

        y : None
            This parameter is unused.
            It is solely included for scikit-learn compatibility.

        confounds : :class:`numpy.ndarray`, :obj:`str`,\
                    :class:`pathlib.Path`, \
                    :class:`pandas.DataFrame` \
                    or :obj:`list` of confounds timeseries, default=None
            Confounds to pass to :func:`nilearn.signal.clean`.

        sample_mask : None, Any type compatible with numpy-array indexing, \
                  or :obj:`list` of \
                  shape: (number of scans - number of volumes removed) \
                  for explicit index, or (number of scans) for binary mask, \
                  default=None
            sample_mask to pass to :func:`nilearn.signal.clean`.


        Returns
        -------
        region_signals: :obj:`numpy.ndarray`
            Signal for each region as provided in the maps (via `maps_img`).
            shape: (n_timepoints, n_regions)
        """
        del y
        return self.fit().transform(img, confounds, sample_mask)

    def inverse_transform(self, region_signals):
        """Compute :term:`vertex` signals from region signals.

        Parameters
        ----------
        region_signals: :obj:`numpy.ndarray`
            Signal for each region as provided in the maps (via `maps_img`).
            shape: (n_timepoints, n_regions)

        Returns
        -------
        vertex_signals: :obj:`~nilearn.surface.SurfaceImage`
            Signal for each vertex projected on the mesh of the `maps_img`.
            The data for each hemisphere is of shape
            (n_vertices_per_hemisphere, n_timepoints).
        """
        self._check_fitted()

        # get concatenated hemispheres/parts data from maps_img and mask_img
        maps_data = concat_extract_surface_data_parts(self.maps_img)
        if self.mask_img is not None:
            mask_data = concat_extract_surface_data_parts(self.mask_img)
        else:
            mask_data = None

        if region_signals.shape[1] != self.n_elements_:
            raise ValueError(
                f"Expected {self.n_elements_} regions, "
                f"but got {region_signals.shape[1]}."
            )
        logger.log("computing image from signals", verbose=self.verbose)
        # project region signals back to vertices
        if mask_data is not None:
            # vertices that are not in the mask will have a signal of 0
            # so we initialize the vertex signals with 0
            # and shape (n_timepoints, n_vertices)
            vertex_signals = np.zeros(
                (region_signals.shape[0], self.maps_img.mesh.n_vertices)
            )
            # dot product between (n_timepoints, n_regions) and
            # (n_regions, n_vertices)
            vertex_signals[:, mask_data.flatten()] = np.dot(
                region_signals, maps_data[mask_data.flatten(), :].T
            )
        else:
            vertex_signals = np.dot(region_signals, maps_data.T)

        # we need the data to be of shape (n_vertices, n_timepoints)
        # because the SurfaceImage object expects it
        vertex_signals = vertex_signals.T

        # split the signal into hemispheres
        vertex_signals = {
            "left": vertex_signals[
                : self.maps_img.data.parts["left"].shape[0], :
            ],
            "right": vertex_signals[
                self.maps_img.data.parts["left"].shape[0] :, :
            ],
        }

        return SurfaceImage(mesh=self.maps_img.mesh, data=vertex_signals)

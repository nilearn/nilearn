"""Extract data from a SurfaceImage, averaging over atlas regions."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.utils.estimator_checks import check_is_fitted

from nilearn import DEFAULT_SEQUENTIAL_CMAP, signal
from nilearn._utils.bids import (
    generate_atlas_look_up_table,
    sanitize_look_up_table,
)
from nilearn._utils.cache_mixin import cache
from nilearn._utils.class_inspect import get_params
from nilearn._utils.docs import fill_doc
from nilearn._utils.helpers import (
    constrained_layout_kwargs,
)
from nilearn._utils.logger import find_stack_level
from nilearn._utils.masker_validation import (
    check_compatibility_mask_and_images,
)
from nilearn._utils.param_validation import (
    check_params,
    check_reduction_strategy,
)
from nilearn.image import concat_imgs, mean_img
from nilearn.maskers.base_masker import _BaseSurfaceMasker
from nilearn.surface.surface import (
    SurfaceImage,
    at_least_2d,
    check_same_n_vertices,
)


def _apply_surf_mask_on_labels(mask_data, labels_data, background_label=0):
    """Apply mask to labels data.

    Ensures that we only get the data back
    according to the mask that was applied.
    So if some labels were removed,
    we will only get the data for the remaining labels,
    the vertices that were masked out will be set to the background label.
    """
    labels_before_mask = {int(label) for label in np.unique(labels_data)}
    labels_data[np.logical_not(mask_data.flatten())] = background_label
    labels_after_mask = {int(label) for label in np.unique(labels_data)}
    labels_diff = labels_before_mask - labels_after_mask
    if labels_diff:
        warnings.warn(
            "After applying mask to the labels image, "
            "the following labels were "
            f"removed: {labels_diff}. "
            f"Out of {len(labels_before_mask)} labels, the "
            "masked labels image only contains "
            f"{len(labels_after_mask)} labels "
            "(including background).",
            stacklevel=find_stack_level(),
        )
    labels = np.unique(labels_data)
    labels = labels[labels != background_label]

    return labels_data, labels


def signals_to_surf_img_labels(
    signals,
    labels,
    labels_img,
    mask_img,
    background_label=0,
):
    """Transform signals to surface image labels."""
    if mask_img is not None:
        mask_data = np.concatenate(list(mask_img.data.parts.values()), axis=0)
        labels_data = np.concatenate(
            list(labels_img.data.parts.values()), axis=0
        )
        _, labels = _apply_surf_mask_on_labels(
            mask_data, labels_data, background_label
        )

    data = {}
    for part_name, labels_part in labels_img.data.parts.items():
        data[part_name] = np.zeros(
            (labels_part.shape[0], signals.shape[0]),
            dtype=signals.dtype,
        )
        for label_idx, label in enumerate(labels):
            data[part_name][labels_part == label] = signals[:, label_idx].T
    return SurfaceImage(mesh=labels_img.mesh, data=data)


@fill_doc
class SurfaceLabelsMasker(_BaseSurfaceMasker):
    """Extract data from a SurfaceImage, averaging over atlas regions.

    .. versionadded:: 0.11.0

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

        .. warning::
            If the labels are not be consistent with the label values
            provided through ``labels_img``,
            excess labels will be dropped,
            and missing labels will be labeled ``'unknown'``.

    'lut' : :obj:`pandas.DataFrame` or :obj:`str` \
            or :obj:`pathlib.Path` to a TSV file or None, default=None
        Mutually exclusive with ``labels``.
        Act as a look up table (lut)
        with at least columns 'index' and 'name'.
        Formatted according to 'dseg.tsv' format from
        `BIDS <https://bids-specification.readthedocs.io/en/latest/derivatives/imaging.html#common-image-derived-labels>`_.

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

    %(standardize_maskers)s

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

    %(verbose0)s

    reports : :obj:`bool`, default=True
        If set to True, data is saved in order to produce a report.

    %(cmap)s
        default="inferno"
        Only relevant for the report figures.

    %(clean_args)s

    Attributes
    ----------
    n_elements_ : :obj:`int`
        The number of discrete values in the mask.
        This is equivalent to the number of unique values in the mask image,
        ignoring the background value.

    lut_ : :obj:`pandas.DataFrame`
        Look-up table derived from the ``labels`` or ``lut``
        or from the values of the label image.
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
        self.reports = reports
        self.strategy = strategy
        self.cmap = cmap
        self.clean_args = clean_args

    @property
    def _labels_data(self):
        """Return data of label image concatenated over hemispheres."""
        all_labels = [x.ravel() for x in self.labels_img.data.parts.values()]
        return np.concatenate(all_labels)

    def fit(self, img=None, y=None):
        """Prepare signal extraction from regions.

        Parameters
        ----------
        img : :obj:`~nilearn.surface.SurfaceImage` object or None, default=None

        y : None
            This parameter is unused.
            It is solely included for scikit-learn compatibility.

        Returns
        -------
        SurfaceLabelsMasker object
        """
        check_params(self.__dict__)
        del y

        check_reduction_strategy(self.strategy)

        if self.labels_img is None:
            raise ValueError(
                "Please provide a labels_img to the masker. For example, "
                "masker = SurfaceLabelsMasker(labels_img=labels_img)"
            )

        if self.labels and self.lut is not None:
            raise ValueError(
                "Pass either labels or a lookup table (lut) to the masker, "
                "but not both."
            )

        self._shelving = False

        all_labels = set(self._labels_data.ravel())
        all_labels.discard(self.background_label)
        self._labels_ = list(all_labels)

        self.n_elements_ = len(self._labels_)

        # generate a look up table if one was not provided
        if self.lut is not None:
            if isinstance(self.lut, (str, Path)):
                lut = pd.read_table(self.lut, sep=None)
            else:
                lut = self.lut
        elif self.labels:
            lut = generate_atlas_look_up_table(
                function=None,
                name=self.labels,
                index=self.labels_img,
            )
        else:
            lut = generate_atlas_look_up_table(
                function=None, index=self.labels_img
            )

        self.lut_ = sanitize_look_up_table(lut, atlas=self.labels_img)

        self.label_names_ = self.lut_.name.to_list()

        if self.mask_img is not None:
            if img is not None:
                check_compatibility_mask_and_images(self.mask_img, img)
            check_same_n_vertices(self.labels_img.mesh, self.mask_img.mesh)
        self.mask_img_ = self.mask_img

        self._shelving = False

        if not self.reports:
            self._reporting_data = None
            return self

        # content to inject in the HTML template
        self._report_content = {
            "description": (
                "This report shows the input surface image overlaid "
                "with the outlines of the mask. "
                "We recommend to inspect the report for the overlap "
                "between the mask and its input image. "
            ),
            "n_vertices": {},
            "number_of_regions": self.n_elements_,
            "summary": {},
            "warning_message": None,
        }

        for part in self.labels_img.data.parts:
            self._report_content["n_vertices"][part] = (
                self.labels_img.mesh.parts[part].n_vertices
            )

        self._reporting_data = self._generate_reporting_data()

        return self

    def _generate_reporting_data(self):
        for part in self.labels_img.data.parts:
            size = []
            relative_size = []

            table = self.lut_.copy()

            for _, row in table.iterrows():
                n_vertices = self.labels_img.data.parts[part] == row["index"]
                size.append(n_vertices.sum())
                tmp = (
                    n_vertices.sum()
                    / self.labels_img.mesh.parts[part].n_vertices
                    * 100
                )
                relative_size.append(f"{tmp:.2}")

            table["size"] = size
            table["relative size"] = relative_size

            self._report_content["summary"][part] = table

        return {
            "labels_image": self.labels_img,
            "images": None,
        }

    def __sklearn_is_fitted__(self):
        return (
            hasattr(self, "n_elements_")
            and hasattr(self, "lut_")
            and hasattr(self, "mask_img_")
        )

    def transform(self, img, confounds=None, sample_mask=None):
        """Extract signals from surface object.

        Parameters
        ----------
        img : :obj:`~nilearn.surface.SurfaceImage` object or \
              :obj:`list` of :obj:`~nilearn.surface.SurfaceImage` or \
              :obj:`tuple` of :obj:`~nilearn.surface.SurfaceImage`
            Mesh and data for both hemispheres.

        confounds : :class:`numpy.ndarray`, :obj:`str`,\
                    :class:`pathlib.Path`, \
                    :class:`pandas.DataFrame` \
                    or :obj:`list` of confounds timeseries, default=None
            Confounds to pass to :func:`nilearn.signal.clean`.

        sample_mask : None, Any type compatible with numpy-array indexing, \
                  or :obj:`list` of \
                  shape: (total number of scans - number of scans removed) \
                  for explicit index, or (number of scans) for binary mask, \
                  default=None
            sample_mask to pass to :func:`nilearn.signal.clean`.

        Returns
        -------
        region_signals : 2D :obj:`numpy.ndarray`
            Signal for each element.
            shape: (img data shape, total number of vertices)
        """
        check_is_fitted(self)

        # if img is a single image, convert it to a list
        # to be able to concatenate it
        if not isinstance(img, list):
            img = [img]
        img = concat_imgs(img)
        check_compatibility_mask_and_images(self.labels_img, img)
        check_same_n_vertices(self.labels_img.mesh, img.mesh)
        img = at_least_2d(img)
        # concatenate data over hemispheres
        img_data = np.concatenate(list(img.data.parts.values()), axis=0)

        labels_data = self._labels_data
        labels = self._labels_
        if self.mask_img_ is not None:
            check_compatibility_mask_and_images(self.mask_img_, img)
            mask_data = np.concatenate(
                list(self.mask_img.data.parts.values()), axis=0
            )
            labels_data, labels = _apply_surf_mask_on_labels(
                mask_data,
                self._labels_data,
                self.background_label,
            )

        if self.smoothing_fwhm is not None:
            warnings.warn(
                "Parameter smoothing_fwhm "
                "is not yet supported for surface data",
                UserWarning,
                stacklevel=find_stack_level(),
            )
            self.smoothing_fwhm = None

        if self.reports:
            self._reporting_data["images"] = img

        parameters = get_params(
            self.__class__,
            self,
            ignore=[
                "mask_img",
            ],
        )
        if self.clean_args is None:
            self.clean_args = {}
        parameters["clean_args"] = self.clean_args

        target_datatype = (
            np.float32 if img_data.dtype == np.float32 else np.float64
        )
        img_data = img_data.astype(target_datatype)

        n_time_points = 1 if len(img_data.shape) == 1 else img_data.shape[1]
        region_signals = np.ndarray(
            (n_time_points, len(labels)), dtype=target_datatype
        )

        # adapted from nilearn.regions.signal_extraction.img_to_signals_labels
        # iterate over time points and apply reduction function over labels.
        reduction_function = getattr(ndimage, self.strategy)
        for n, sample in enumerate(np.rollaxis(img_data, -1)):
            region_signals[n] = np.asarray(
                reduction_function(sample, labels=labels_data, index=labels)
            )

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
            Mesh and data for both hemispheres.

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
                  shape: (total number of scans - number of scans removed) \
                  for explicit index, or (number of scans) for binary mask, \
                  default=None
            sample_mask to pass to :func:`nilearn.signal.clean`.

        Returns
        -------
        :obj:`numpy.ndarray`
            Signal for each element.
            shape: (img data shape, total number of vertices)
        """
        del y
        return self.fit().transform(img, confounds, sample_mask)

    def inverse_transform(self, signals):
        """Transform extracted signal back to surface image.

        Parameters
        ----------
        signals : :obj:`numpy.ndarray`
            Extracted signal for each region.
            If a 1D array is provided, then the shape of each hemisphere's data
            should be (number of elements,) in the returned surface image.
            If a 2D array is provided, then it would be
            (number of scans, number of elements).


        Returns
        -------
        :obj:`~nilearn.surface.SurfaceImage` object
            Mesh and data for both hemispheres.
        """
        check_is_fitted(self)

        return signals_to_surf_img_labels(
            signals,
            self._labels_,
            self.labels_img,
            self.mask_img,
            self.background_label,
        )

    def generate_report(self):
        """Generate a report."""
        from nilearn.reporting.html_report import generate_report

        return generate_report(self)

    def _reporting(self):
        """Load displays needed for report.

        Returns
        -------
        displays : list
            A list of all displays to be rendered.
        """
        import matplotlib.pyplot as plt

        from nilearn.reporting.utils import figure_to_png_base64

        # Handle the edge case where this function is
        # called with a masker having report capabilities disabled
        if self._reporting_data is None:
            return [None]

        fig = self._create_figure_for_report()

        plt.close()

        init_display = figure_to_png_base64(fig)

        return [init_display]

    def _create_figure_for_report(self):
        """Create a figure of the contours of label image.

        If transform() was applied to an image,
        this image is used as background
        on which the contours are drawn.
        """
        import matplotlib.pyplot as plt

        from nilearn.plotting import plot_surf, plot_surf_contours

        labels_img = self._reporting_data["labels_image"]

        img = self._reporting_data["images"]
        if img:
            img = mean_img(img)
            vmin, vmax = img.data._get_min_max()

        # TODO: possibly allow to generate a report with other views
        views = ["lateral", "medial"]
        hemispheres = ["left", "right"]

        fig, axes = plt.subplots(
            len(views),
            len(hemispheres),
            subplot_kw={"projection": "3d"},
            figsize=(20, 20),
            **constrained_layout_kwargs(),
        )
        axes = np.atleast_2d(axes)

        for ax_row, view in zip(axes, views):
            for ax, hemi in zip(ax_row, hemispheres):
                if img:
                    plot_surf(
                        surf_map=img,
                        hemi=hemi,
                        view=view,
                        figure=fig,
                        axes=ax,
                        cmap=self.cmap,
                        vmin=vmin,
                        vmax=vmax,
                    )
                plot_surf_contours(
                    roi_map=labels_img,
                    hemi=hemi,
                    view=view,
                    figure=fig,
                    axes=ax,
                )

        return fig

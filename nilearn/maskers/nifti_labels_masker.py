"""Transformer for computing ROI signals."""

import warnings

import numpy as np
from joblib import Memory
from nibabel import Nifti1Image

from nilearn import _utils, image, masking
from nilearn._utils import logger
from nilearn._utils.helpers import is_matplotlib_installed
from nilearn.maskers._utils import compute_middle_image
from nilearn.maskers.base_masker import BaseMasker, _filter_and_extract


class _ExtractionFunctor:
    func_name = "nifti_labels_masker_extractor"

    def __init__(
        self,
        _resampled_labels_img_,
        background_label,
        strategy,
        keep_masked_labels,
        mask_img,
    ):
        self._resampled_labels_img_ = _resampled_labels_img_
        self.background_label = background_label
        self.strategy = strategy
        self.keep_masked_labels = keep_masked_labels
        self.mask_img = mask_img

    def __call__(self, imgs):
        from ..regions.signal_extraction import img_to_signals_labels

        signals, labels, masked_labels_img = img_to_signals_labels(
            imgs,
            self._resampled_labels_img_,
            background_label=self.background_label,
            strategy=self.strategy,
            keep_masked_labels=self.keep_masked_labels,
            mask_img=self.mask_img,
            return_masked_atlas=True,
        )
        return signals, (labels, masked_labels_img)


@_utils.fill_doc
class NiftiLabelsMasker(BaseMasker):
    """Class for extracting data from Niimg-like objects \
       using labels of non-overlapping brain regions.

    NiftiLabelsMasker is useful when data from non-overlapping volumes should
    be extracted (contrarily to :class:`nilearn.maskers.NiftiMapsMasker`).

    Use case:
    summarize brain signals from clusters that were obtained by prior
    K-means or Ward clustering.

    For more details on the definitions of labels in Nilearn,
    see the :ref:`region` section.

    Parameters
    ----------
    labels_img : Niimg-like object
        See :ref:`extracting_data`.
        Region definitions, as one image of labels.

    labels : :obj:`list` of :obj:`str`, optional
        Full labels corresponding to the labels image.
        This is used to improve reporting quality if provided.

        .. warning::
            The labels must be consistent with the label values
            provided through ``labels_img``.

    background_label : :obj:`int` or :obj:`float`, default=0
        Label used in labels_img to represent background.

        .. warning:::

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
    dtype : {dtype, "auto"}, optional
        Data type toward which the data should be converted. If "auto", the
        data will be converted to int32 if dtype is discrete and float32 if it
        is continuous.

    resampling_target : {"data", "labels", None}, default="data"
        Gives which image gives the final shape/size.
        For example, if ``resampling_target`` is ``"data"``,
        the atlas is resampled to the shape of the data if needed.
        If it is ``"labels"`` then mask_img and images provided to fit()
        are resampled to the shape and affine of maps_img.
        ``"None"`` means no resampling:
        if shapes and affines do not match, a ValueError is raised.

    %(memory)s
    %(memory_level1)s
    %(verbose0)s
    strategy : :obj:`str`, default='mean'
        The name of a valid function to reduce the region with.
        Must be one of: sum, mean, median, minimum, maximum, variance,
        standard_deviation.
    %(keep_masked_labels)s
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

    region_ids_ : dict[str | int, int]
        A dictionary containing the region ids corresponding
        to each column in the ``region_signal``
        returned by `fit_transform`.
        The region id corresponding to ``region_signal[:,i]``
        is ``region_ids_[i]``.
        ``region_ids_['background']`` is the background label.

        .. versionadded:: 0.10.3

    region_names_ : dict[int, str]
        A dictionary containing the region names corresponding
        to each column in the ``region_signal``
        returned by `fit_transform`.
        The region names correspond to the labels provided
        in labels in input.
        The region name corresponding to ``region_signal[:,i]``
        is ``region_names_[i]``.

        .. versionadded:: 0.10.3

    region_atlas_ : Niimg-like object
        Regions definition as labels.
        The labels correspond to the indices in ``region_ids_``.
        The region in ``region_atlas_`` that takes the value ``region_ids_[i]``
        is used to compute the signal in ``region_signal[:,i]``.

        .. versionadded:: 0.10.3

    See Also
    --------
    nilearn.maskers.NiftiMasker

    """

    # memory and memory_level are used by _utils.CacheMixin.

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
        keep_masked_labels=True,
        reports=True,
        **kwargs,
    ):
        if memory is None:
            memory = Memory(location=None, verbose=0)
        self.labels_img = labels_img

        self.background_label = background_label
        self._original_region_ids = self._get_labels_values(self.labels_img)
        self.labels = self._sanitize_labels(labels)
        self._check_mismatch_labels_regions(
            self._original_region_ids, tolerant=True
        )

        self.mask_img = mask_img

        # Parameters for smooth_array
        self.smoothing_fwhm = smoothing_fwhm

        # Parameters for clean()
        self.standardize = standardize
        self.standardize_confounds = standardize_confounds
        self.high_variance_confounds = high_variance_confounds
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.dtype = dtype
        self.clean_kwargs = {
            k[7:]: v for k, v in kwargs.items() if k.startswith("clean__")
        }

        # Parameters for resampling
        self.resampling_target = resampling_target

        # Parameters for joblib
        self.memory = memory
        self.memory_level = memory_level
        self.verbose = verbose
        self.reports = reports
        self._report_content = {
            "description": (
                "This reports shows the regions "
                "defined by the labels of the mask."
            ),
            "warning_message": None,
        }

        available_reduction_strategies = {
            "mean",
            "median",
            "sum",
            "minimum",
            "maximum",
            "standard_deviation",
            "variance",
        }

        if strategy not in available_reduction_strategies:
            raise ValueError(
                f"Invalid strategy '{strategy}'. "
                f"Valid strategies are {available_reduction_strategies}."
            )
        self.strategy = strategy

        if resampling_target not in ("labels", "data", None):
            raise ValueError(
                "invalid value for 'resampling_target' "
                f"parameter: {resampling_target}"
            )

        self.keep_masked_labels = keep_masked_labels

        self.cmap = kwargs.get("cmap", "CMRmap_r")

    def _get_labels_values(self, labels_image):
        labels_image = image.load_img(labels_image, dtype="int32")
        labels_image_data = image.get_data(labels_image)
        return np.unique(labels_image_data)

    def _sanitize_labels(self, labels):
        """Check and clean labels.

        - checks that labels is a list of strings.
        - cast all items of the list into strings if they are bytestrings.
        """
        if labels is not None:
            if not isinstance(labels, list):
                warnings.warn(
                    f"'labels' must be a list. Got: {type(labels)}",
                    stacklevel=3,
                )
            if not all(isinstance(x, str) for x in labels):
                types_labels = {type(x) for x in labels}
                warnings.warn(
                    "All elements of 'labels' must be a string.\n"
                    f"Got a list of {types_labels}",
                    stacklevel=3,
                )
            labels = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in labels
            ]
        return labels

    def _check_mismatch_labels_regions(
        self, region_ids, tolerant=True, resampling_done=False
    ):
        """Check we have as many labels as regions (plus background).

        Parameters
        ----------
        region_ids : :obj:`list` or numpy.array

        tolerant : :obj:`bool`, default=True
                  If set to `True` this function will throw a warning,
                  and will throw an error otherwise.

        resampling_done : :obj:`bool`, default=False
                         Used to mention if this check is done
                         before or after the resampling has been done,
                         to adapt the message accordingly.
        """
        if (
            self.labels is not None
            and len(self.labels) != self._number_of_regions(region_ids) + 1
        ):
            msg = (
                "Mismatch between the number of provided labels "
                f"({len(self.labels)}) and the number of regions in "
                "provided label image "
                f"({self._number_of_regions(region_ids) + 1})."
            )
            if (
                getattr(self, "resampling_target", None) == "data"
                and resampling_done
            ):
                msg += (
                    "\nNote that this may be due to some regions "
                    "being dropped from the label image "
                    "after resampling."
                )
            if tolerant:
                warnings.warn(msg, UserWarning, stacklevel=3)
            else:
                raise ValueError(msg)

    def _number_of_regions(self, region_ids):
        """Compute number of regions excluding the background.

        Parameters
        ----------
        region_ids : :obj:`list` or numpy.array
        """
        if isinstance(region_ids, list):
            region_ids = np.array(region_ids)
        return np.sum(region_ids != self.background_label)

    def _post_masking_atlas(self, visualize=False):
        """
        Find the masked atlas before transform and return it.

        Also return the removed region ids and names.
        if visualize is True, plot the masked atlas.
        """
        labels_data = _utils.niimg.safe_get_data(
            self._resampled_labels_img_, ensure_finite=True
        )
        labels_data = labels_data.copy()
        mask_data = _utils.niimg.safe_get_data(
            self.mask_img_, ensure_finite=True
        )
        mask_data = mask_data.copy()
        region_ids_before_masking = np.unique(labels_data).tolist()
        # apply the mask to the atlas
        labels_data[np.logical_not(mask_data)] = self.background_label
        region_ids_after_masking = np.unique(labels_data).tolist()
        masked_atlas = Nifti1Image(
            labels_data.astype(np.int8), self._resampled_labels_img_.affine
        )
        removed_region_ids = [
            region_id
            for region_id in region_ids_before_masking
            if region_id not in region_ids_after_masking
        ]
        removed_region_names = [
            self._region_id_name[region_id]
            for region_id in removed_region_ids
            if region_id != self.background_label
        ]
        display = None
        if visualize:
            from nilearn.plotting import plot_roi

            display = plot_roi(masked_atlas, title="Masked atlas")

        return masked_atlas, removed_region_ids, removed_region_names, display

    def generate_report(self):
        """Generate a report."""
        if not is_matplotlib_installed():
            with warnings.catch_warnings():
                mpl_unavail_msg = (
                    "Matplotlib is not imported! "
                    "No reports will be generated."
                )
                warnings.filterwarnings("always", message=mpl_unavail_msg)
                warnings.warn(category=ImportWarning, message=mpl_unavail_msg)
                return [None]

        from nilearn.reporting.html_report import generate_report

        return generate_report(self)

    def _reporting(self):
        """Return a list of all displays to be rendered.

        Returns
        -------
        displays : list
            A list of all displays to be rendered.

        """
        import matplotlib.pyplot as plt

        from nilearn import plotting

        if self._reporting_data is not None:
            labels_image = self._reporting_data["labels_image"]
        else:
            labels_image = None

        if labels_image is not None:
            # Remove warning message in case where the masker was
            # previously fitted with no func image and is re-fitted
            if "warning_message" in self._report_content:
                self._report_content["warning_message"] = None

            label_values = self._get_labels_values(labels_image)

            self._check_mismatch_labels_regions(label_values, tolerant=False)

            self._report_content["number_of_regions"] = (
                self._number_of_regions(label_values)
            )

            label_values = label_values[label_values != self.background_label]
            columns = [
                "label value",
                "region name",
                "size (in mm^3)",
                "relative size (in %)",
            ]

            if self.labels is None:
                columns.remove("region name")

            labels_image = image.load_img(labels_image, dtype="int32")
            labels_image_data = image.get_data(labels_image)
            labels_image_affine = labels_image.affine

            regions_summary = {c: [] for c in columns}
            for label in label_values:
                regions_summary["label value"].append(label)
                if self.labels is not None:
                    regions_summary["region name"].append(self.labels[label])

                size = len(labels_image_data[labels_image_data == label])
                voxel_volume = np.abs(
                    np.linalg.det(labels_image_affine[:3, :3])
                )
                regions_summary["size (in mm^3)"].append(
                    round(size * voxel_volume)
                )
                regions_summary["relative size (in %)"].append(
                    round(
                        size
                        / len(labels_image_data[labels_image_data != 0])
                        * 100,
                        2,
                    )
                )

            self._report_content["summary"] = regions_summary

            img = self._reporting_data["img"]

            # compute the cut coordinates on the label image in case
            # we have a functional image
            cut_coords = plotting.find_xyz_cut_coords(
                labels_image, activation_threshold=0.5
            )

            # If we have a func image to show in the report, use it
            if img is not None:
                if self._reporting_data["dim"] == 5:
                    msg = (
                        "A list of 4D subject images were provided to fit. "
                        "Only first subject is shown in the report."
                    )
                    warnings.warn(msg, stacklevel=6)
                    self._report_content["warning_message"] = msg
                display = plotting.plot_img(
                    img,
                    cut_coords=cut_coords,
                    black_bg=False,
                    cmap=self.cmap,
                )
                plt.close()
                display.add_contours(labels_image, filled=False, linewidths=3)

            # Otherwise, simply plot the ROI of the label image
            # and give a warning to the user
            else:
                msg = (
                    "No image provided to fit in NiftiLabelsMasker. "
                    "Plotting ROIs of label image on the "
                    "MNI152Template for reporting."
                )
                warnings.warn(msg, stacklevel=6)
                self._report_content["warning_message"] = msg
                display = plotting.plot_roi(labels_image)
                plt.close()

            # If we have a mask, show its contours
            if self._reporting_data["mask"] is not None:
                display.add_contours(
                    self._reporting_data["mask"],
                    filled=False,
                    colors="g",
                    linewidths=3,
                )
        else:
            self._report_content["summary"] = None
            display = None

        return [display]

    def fit(
        self,
        imgs=None,
        y=None,  # noqa: ARG002
    ):
        """Prepare signal extraction from regions.

        Parameters
        ----------
        imgs : :obj:`list` of Niimg-like objects
            See :ref:`extracting_data`.
            Image data passed to the reporter.

        y : None
            This parameter is unused. It is solely included for scikit-learn
            compatibility.
        """
        repr = _utils.repr_niimgs(self.labels_img, shorten=(not self.verbose))
        msg = f"loading data from {repr}"
        logger.log(msg=msg, verbose=self.verbose)
        self.labels_img_ = _utils.check_niimg_3d(self.labels_img)

        # create _region_id_name dictionary
        # this dictionary will be used to store region names and
        # the corresponding region ids as keys
        self._region_id_name = None
        if self.labels is not None:
            known_backgrounds = {"background", "Background"}
            initial_region_ids = [
                region_id
                for region_id in np.unique(
                    _utils.niimg.safe_get_data(self.labels_img_)
                )
                if region_id != self.background_label
            ]
            initial_region_names = [
                region_name
                for region_name in self.labels
                if region_name not in known_backgrounds
            ]

            if len(initial_region_ids) != len(initial_region_names):
                warnings.warn(
                    "Number of regions in the labels image "
                    "does not match the number of labels provided.",
                    stacklevel=2,
                )
            # if number of regions in the labels image is more
            # than the number of labels provided, then we cannot
            # create _region_id_name dictionary
            if len(initial_region_ids) <= len(initial_region_names):
                self._region_id_name = {
                    region_id: initial_region_names[i]
                    for i, region_id in enumerate(initial_region_ids)
                }

        if self.mask_img is not None:
            repr = _utils.repr_niimgs(
                self.mask_img, shorten=(not self.verbose)
            )
            msg = f"loading data from {repr}"
            logger.log(msg=msg, verbose=self.verbose)
            self.mask_img_ = _utils.check_niimg_3d(self.mask_img)

        else:
            self.mask_img_ = None

        # Check shapes and affines or resample.
        if self.mask_img_ is not None:
            if self.resampling_target == "data":
                # resampling will be done at transform time
                pass

            elif self.resampling_target is None:
                if self.mask_img_.shape != self.labels_img_.shape[:3]:
                    raise ValueError(
                        _utils.compose_err_msg(
                            "Regions and mask do not have the same shape",
                            mask_img=self.mask_img,
                            labels_img=self.labels_img,
                        )
                    )

                if not np.allclose(
                    self.mask_img_.affine,
                    self.labels_img_.affine,
                ):
                    raise ValueError(
                        _utils.compose_err_msg(
                            "Regions and mask do not have the same affine.",
                            mask_img=self.mask_img,
                            labels_img=self.labels_img,
                        ),
                    )

            elif self.resampling_target == "labels":
                logger.log("resampling the mask", verbose=self.verbose)
                # TODO switch to force_resample=True
                # when bumping to version > 0.13
                self.mask_img_ = image.resample_img(
                    self.mask_img_,
                    target_affine=self.labels_img_.affine,
                    target_shape=self.labels_img_.shape[:3],
                    interpolation="nearest",
                    copy=True,
                    copy_header=True,
                    force_resample=False,
                )

            else:
                raise ValueError(
                    "Invalid value for "
                    f"resampling_target: {self.resampling_target}"
                )

            # Just check that the mask is valid
            masking.load_mask_img(self.mask_img_)

        if not hasattr(self, "_resampled_labels_img_"):
            # obviates need to run .transform() before .inverse_transform()
            self._resampled_labels_img_ = self.labels_img_

        if self.reports:
            self._reporting_data = {
                "labels_image": self._resampled_labels_img_,
                "mask": self.mask_img_,
                "dim": None,
                "img": imgs,
            }
            if imgs is not None:
                imgs, dims = compute_middle_image(imgs)
                self._reporting_data["img"] = imgs
                self._reporting_data["dim"] = dims
        else:
            self._reporting_data = None

        # Infer the number of elements in the mask
        # This is equal to the number of unique values in the label image,
        # minus the background value.
        self.n_elements_ = (
            np.unique(image.get_data(self._resampled_labels_img_)).size - 1
        )

        return self

    def fit_transform(self, imgs, confounds=None, sample_mask=None):
        """Prepare and perform signal extraction from regions.

        Parameters
        ----------
        imgs : 3D/4D Niimg-like object
            See :ref:`extracting_data`.
            Images to process.
            If a 3D niimg is provided, a singleton dimension will be added to
            the output to represent the single scan in the niimg.

        confounds : CSV file or array-like or :obj:`pandas.DataFrame`, optional
            This parameter is passed to signal.clean. Please see the related
            documentation for details.
            shape: (number of scans, number of confounds)

        sample_mask : Any type compatible with numpy-array indexing, optional
            shape: (number of scans - number of volumes removed, )
            Masks the niimgs along time/fourth dimension to perform scrubbing
            (remove volumes with high motion) and/or non-steady-state volumes.
            This parameter is passed to signal.clean.

                .. versionadded:: 0.8.0

        Returns
        -------
        region_signals : 2D :obj:`numpy.ndarray`
            Signal for each label.
            shape: (number of scans, number of labels)

        """
        return self.fit(imgs).transform(
            imgs, confounds=confounds, sample_mask=sample_mask
        )

    def _check_fitted(self):
        if not hasattr(self, "labels_img_"):
            raise ValueError(
                f"It seems that {self.__class__.__name__} has not been "
                "fitted. "
                "You must call fit() before calling transform()."
            )

    def transform_single_imgs(self, imgs, confounds=None, sample_mask=None):
        """Extract signals from a single 4D niimg.

        Parameters
        ----------
        imgs : 3D/4D Niimg-like object
            See :ref:`extracting_data`.
            Images to process.
            If a 3D niimg is provided, a singleton dimension will be added to
            the output to represent the single scan in the niimg.

        confounds : CSV file or array-like or :obj:`pandas.DataFrame`, optional
            This parameter is passed to signal.clean. Please see the related
            documentation for details.
            shape: (number of scans, number of confounds)

        sample_mask : Any type compatible with numpy-array indexing, optional
            shape: (number of scans - number of volumes removed, )
            Masks the niimgs along time/fourth dimension to perform scrubbing
            (remove volumes with high motion) and/or non-steady-state volumes.
            This parameter is passed to signal.clean.

                .. versionadded:: 0.8.0

        Returns
        -------
        region_signals : 2D numpy.ndarray
            Signal for each label.
            shape: (number of scans, number of labels)

        Warns
        -----
        DeprecationWarning
            If a 3D niimg input is provided, the current behavior
            (adding a singleton dimension to produce a 2D array) is deprecated.
            Starting in version 0.12, a 1D array will be returned for 3D
            inputs.

        """
        # We handle the resampling of labels separately because the affine of
        # the labels image should not impact the extraction of the signal.

        if not hasattr(self, "_resampled_labels_img_"):
            self._resampled_labels_img_ = self.labels_img_

        if not hasattr(self, "_resampled_mask_img"):
            self._resampled_mask_img = self.mask_img_

        if self.resampling_target == "data":
            imgs_ = _utils.check_niimg(imgs, atleast_4d=True)
            if not _utils.niimg_conversions.check_same_fov(
                imgs_,
                self._resampled_labels_img_,
            ):
                self._resample_labels(imgs_)

            if (self.mask_img is not None) and (
                not _utils.niimg_conversions.check_same_fov(
                    imgs_,
                    self._resampled_mask_img,
                )
            ):
                logger.log("Resampling mask", self.verbose)
                self._resampled_mask_img = self._cache(
                    image.resample_img, func_memory_level=2
                )(
                    self.mask_img_,
                    interpolation="nearest",
                    target_shape=imgs_.shape[:3],
                    target_affine=imgs_.affine,
                    copy_header=True,
                    force_resample=False,
                )

            # Remove imgs_ from memory before loading the same image
            # in filter_and_extract.
            del imgs_

        target_shape = None
        target_affine = None
        if self.resampling_target == "labels":
            target_shape = self._resampled_labels_img_.shape[:3]
            target_affine = self._resampled_labels_img_.affine

        params = _utils.class_inspect.get_params(
            NiftiLabelsMasker,
            self,
            ignore=["resampling_target"],
        )
        params["target_shape"] = target_shape
        params["target_affine"] = target_affine
        params["clean_kwargs"] = self.clean_kwargs

        region_signals, (ids, masked_atlas) = self._cache(
            _filter_and_extract,
            ignore=["verbose", "memory", "memory_level"],
        )(
            # Images
            imgs,
            _ExtractionFunctor(
                self._resampled_labels_img_,
                self.background_label,
                self.strategy,
                self.keep_masked_labels,
                self._resampled_mask_img,
            ),
            # Pre-processing
            params,
            confounds=confounds,
            sample_mask=sample_mask,
            dtype=self.dtype,
            # Caching
            memory=self.memory,
            memory_level=self.memory_level,
            verbose=self.verbose,
        )

        self.labels_ = ids

        # defining a dictionary containing regions ids
        region_ids = {"background": self.background_label}
        for i in range(region_signals.shape[1]):
            # ids does not include background label
            region_ids[i] = ids[i]

        self.region_names_ = None

        self._check_mismatch_labels_regions(
            self.labels_, tolerant=True, resampling_done=True
        )

        if self._region_id_name is not None:
            self.region_names_ = {
                key: self._region_id_name[region_id]
                for key, region_id in region_ids.items()
                if region_id != self.background_label
            }

        self.region_ids_ = region_ids
        self.region_atlas_ = masked_atlas

        return region_signals

    def _resample_labels(self, imgs_):
        logger.log("Resampling labels", self.verbose, stack_level=2)
        labels_before_resampling = set(
            np.unique(_utils.niimg.safe_get_data(self._resampled_labels_img_))
        )
        self._resampled_labels_img_ = self._cache(
            image.resample_img, func_memory_level=2
        )(
            self.labels_img_,
            interpolation="nearest",
            target_shape=imgs_.shape[:3],
            target_affine=imgs_.affine,
            copy_header=True,
            force_resample=False,
        )
        labels_after_resampling = set(
            np.unique(_utils.niimg.safe_get_data(self._resampled_labels_img_))
        )
        if labels_diff := labels_before_resampling.difference(
            labels_after_resampling
        ):
            warnings.warn(
                "After resampling the label image to the data image, "
                f"the following labels were removed: {labels_diff}. "
                "Label image only contains "
                f"{len(labels_after_resampling)} labels "
                "(including background)."
            )

        return self

    def inverse_transform(self, signals):
        """Compute :term:`voxel` signals from region signals.

        Any mask given at initialization is taken into account.

        .. versionchanged:: 0.9.2

            This method now supports 1D arrays, which will produce 3D images.

        Parameters
        ----------
        signals : 1D/2D :obj:`numpy.ndarray`
            Signal for each region.
            If a 1D array is provided, then the shape should be
            (number of elements,), and a 3D img will be returned.
            If a 2D array is provided, then the shape should be
            (number of scans, number of elements), and a 4D img will be
            returned.

        Returns
        -------
        img : :obj:`nibabel.nifti1.Nifti1Image`
            Signal for each voxel
            shape: (X, Y, Z, number of scans)

        """
        from ..regions import signal_extraction

        self._check_fitted()

        logger.log("computing image from signals", verbose=self.verbose)
        return signal_extraction.signals_to_img_labels(
            signals,
            self._resampled_labels_img_,
            self.mask_img_,
            background_label=self.background_label,
        )

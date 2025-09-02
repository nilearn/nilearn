"""Transformer for computing ROI signals."""

import warnings
from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
from nibabel import Nifti1Image
from sklearn.utils.estimator_checks import check_is_fitted

from nilearn._utils.bids import (
    sanitize_look_up_table,
)
from nilearn._utils.class_inspect import get_params
from nilearn._utils.docs import fill_doc
from nilearn._utils.logger import find_stack_level
from nilearn._utils.niimg import safe_get_data
from nilearn._utils.niimg_conversions import (
    check_niimg,
    check_niimg_3d,
    check_same_fov,
)
from nilearn._utils.param_validation import (
    check_params,
    check_reduction_strategy,
)
from nilearn.image import get_data, load_img, resample_img
from nilearn.maskers._utils import compute_middle_image
from nilearn.maskers.base_masker import (
    BaseMasker,
    filter_and_extract,
    generate_lut,
    mask_logger,
)
from nilearn.masking import load_mask_img


class _ExtractionFunctor:
    func_name = "nifti_labels_masker_extractor"

    def __init__(
        self,
        labels_img,
        background_label,
        strategy,
        keep_masked_labels,
        mask_img,
    ):
        self.labels_img = labels_img
        self.background_label = background_label
        self.strategy = strategy
        self.keep_masked_labels = keep_masked_labels
        self.mask_img = mask_img

    def __call__(self, imgs):
        from nilearn.regions.signal_extraction import img_to_signals_labels

        signals, labels, masked_labels_img = img_to_signals_labels(
            imgs,
            self.labels_img,
            background_label=self.background_label,
            strategy=self.strategy,
            keep_masked_labels=self.keep_masked_labels,
            mask_img=self.mask_img,
            return_masked_atlas=True,
        )
        return signals, (labels, masked_labels_img)


@fill_doc
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
    labels_img : Niimg-like object or None, default=None
        See :ref:`extracting_data`.
        Region definitions, as one image of labels.

    labels : :obj:`list` of :obj:`str`, optional
        Full labels corresponding to the labels image.
        This is used to improve reporting quality if provided.
        Mutually exclusive with ``lut``.

        "Background" can be included in this list of labels
        to denote which values in the image should be considered
        background value.

        .. warning::
            The labels must be consistent with the label values
            provided through ``labels_img``.
            If too many labels are passed,
            a warning is thrown and extra labels are dropped.
            If too few labels are passed,
            extra regions will get the 'unknown' label.

    %(masker_lut)s


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

    %(dtype)s

    resampling_target : {"data", "labels", None}, default="data"
        Defines which image gives the final shape/size.

        - ``"data"`` means the atlas is resampled
          to the shape of the data if needed.
        - ``"labels"`` means that the ``mask_img`` and images provided
          to ``fit()`` are resampled to the shape and affine of ``labels_img``.
        - ``"None"`` means no resampling:
          if shapes and affines do not match, a :obj:`ValueError` is raised.

    %(memory)s

    %(memory_level1)s

    %(verbose0)s

    %(strategy)s

    %(keep_masked_labels)s

    reports : :obj:`bool`, default=True
        If set to True, data is saved in order to produce a report.

    %(cmap)s
        default="CMRmap_r"
        Only relevant for the report figures.

    %(clean_args)s
        .. versionadded:: 0.12.0

    %(masker_kwargs)s

    Attributes
    ----------
    %(clean_args_)s

    %(masker_kwargs_)s

    labels_img_ : :obj:`nibabel.nifti1.Nifti1Image`
        The labels image.

    lut_ : :obj:`pandas.DataFrame`
        Look-up table derived from the ``labels`` or ``lut``
        or from the values of the label image.

    %(nifti_mask_img_)s

    memory_ : joblib memory cache


    See Also
    --------
    nilearn.maskers.NiftiMasker

    """

    # memory and memory_level are used by _utils.CacheMixin.

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
        cmap="CMRmap_r",
        clean_args=None,
        **kwargs,  # TODO (nilearn >= 0.13.0) remove
    ):
        self.labels_img = labels_img
        self.background_label = background_label

        self.labels = labels
        self.lut = lut

        self.mask_img = mask_img
        self.keep_masked_labels = keep_masked_labels

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
        self.clean_args = clean_args

        # TODO (nilearn >= 0.13.0) remove
        self.clean_kwargs = kwargs

        # Parameters for resampling
        self.resampling_target = resampling_target

        # Parameters for joblib
        self.memory = memory
        self.memory_level = memory_level
        self.verbose = verbose

        # Parameters for reports
        self.reports = reports
        self.cmap = cmap

        self.strategy = strategy

    @property
    def _region_id_name(self):
        """Return dictionary used to store region names and
        the corresponding region ids as keys.
        """
        check_is_fitted(self)
        lut = self.lut_
        return (
            lut.loc[lut["name"] != "Background"]
            .set_index("index")["name"]
            .to_dict()
        )

    @property
    def labels_(self) -> list[Union[int, float]]:
        """Return list of labels of the regions.

        The background label is included if present in the image.
        """
        check_is_fitted(self)
        lut = self.lut_
        if hasattr(self, "_lut_"):
            lut = self._lut_
        return lut["index"].to_list()

    @property
    def region_names_(self) -> dict[int, str]:
        """Return a dictionary containing the region names corresponding \
            to each column in the array returned by `transform`.

        The region names correspond to the labels provided
        in labels in input.
        The region name corresponding to ``region_signal[:,i]``
        is ``region_names_[i]``.

        .. versionadded:: 0.10.3
        """
        check_is_fitted(self)

        index = self.labels_
        valid_ids = [id for id in index if id != self.background_label]

        sub_df = self.lut_[self.lut_["index"].isin(valid_ids)]

        return sub_df["name"].reset_index(drop=True).to_dict()

    @property
    def region_ids_(self) -> dict[Union[str, int], Union[int, float]]:
        """Return dictionary containing the region ids corresponding \
           to each column in the array \
           returned by `transform`.

        The region id corresponding to ``region_signal[:,i]``
        is ``region_ids_[i]``.
        ``region_ids_['background']`` is the background label.

        .. versionadded:: 0.10.3
        """
        check_is_fitted(self)

        index = self.labels_

        region_ids_: dict[Union[str, int], Union[int, float]] = {}
        if self.background_label in index:
            index.pop(index.index(self.background_label))
            region_ids_["background"] = self.background_label
        for i, id in enumerate(index):
            region_ids_[i] = id  # noqa : PERF403

        return region_ids_

    @property
    def n_elements_(self) -> int:
        """Return number of regions.

        This is equal to the number of unique values
        in the fitted label image,
        minus the background value.

        .. versionadded:: 0.9.2
        """
        check_is_fitted(self)
        lut = self.lut_
        if hasattr(self, "_lut_"):
            lut = self._lut_
        return len(lut[lut["index"] != self.background_label])

    def _post_masking_atlas(self, visualize=False):
        """
        Find the masked atlas before transform and return it.

        Also return the removed region ids and names.
        if visualize is True, plot the masked atlas.
        """
        labels_data = safe_get_data(self.labels_img_, ensure_finite=True)
        labels_data = labels_data.copy()
        mask_data = safe_get_data(self.mask_img_, ensure_finite=True)
        mask_data = mask_data.copy()
        region_ids_before_masking = np.unique(labels_data).tolist()
        # apply the mask to the atlas
        labels_data[np.logical_not(mask_data)] = self.background_label
        region_ids_after_masking = np.unique(labels_data).tolist()
        masked_atlas = Nifti1Image(
            labels_data.astype(np.int8), self.labels_img_.affine
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

        labels_image = None
        if self._reporting_data is not None:
            labels_image = self._reporting_data["labels_image"]

        if (
            labels_image is None
            or not self.__sklearn_is_fitted__
            or not self.reports
        ):
            self._report_content["summary"] = None
            return [None]

        # Remove warning message in case where the masker was
        # previously fitted with no func image and is re-fitted
        if "warning_message" in self._report_content:
            self._report_content["warning_message"] = None

        table = self.lut_.copy()
        if hasattr(self, "_lut_"):
            table = self._lut_.copy()

        table = table[["index", "name"]]

        table["index"] = table["index"].astype(int)

        table = table.rename(
            columns={"name": "region name", "index": "label value"}
        )

        labels_image = load_img(labels_image, dtype="int32")
        labels_image_data = get_data(labels_image)
        labels_image_affine = labels_image.affine

        voxel_volume = np.abs(np.linalg.det(labels_image_affine[:3, :3]))

        new_columns = {"size (in mm^3)": [], "relative size (in %)": []}
        for label in table["label value"].to_list():
            size = len(labels_image_data[labels_image_data == label])
            new_columns["size (in mm^3)"].append(round(size * voxel_volume))

            new_columns["relative size (in %)"].append(
                round(
                    size
                    / len(
                        labels_image_data[
                            labels_image_data != self.background_label
                        ]
                    )
                    * 100,
                    2,
                )
            )

        table = pd.concat([table, pd.DataFrame(new_columns)], axis=1)

        table = table[table["label value"] != self.background_label]

        self._report_content["summary"] = table
        self._report_content["number_of_regions"] = self.n_elements_

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
                warnings.warn(msg, stacklevel=find_stack_level())
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
            warnings.warn(msg, stacklevel=find_stack_level())
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

        return [display]

    @fill_doc
    def fit(self, imgs=None, y=None):
        """Prepare signal extraction from regions.

        Parameters
        ----------
        imgs : :obj:`list` of Niimg-like objects or None, default=None
            See :ref:`extracting_data`.
            Image data passed to the reporter.

        %(y_dummy)s
        """
        del y
        check_params(self.__dict__)
        check_reduction_strategy(self.strategy)

        if self.resampling_target not in ("labels", "data", None):
            raise ValueError(
                "invalid value for 'resampling_target' "
                f"parameter: {self.resampling_target}"
            )

        self._sanitize_cleaning_parameters()
        self.clean_args_ = {} if self.clean_args is None else self.clean_args

        self._report_content = {
            "description": (
                "This reports shows the regions "
                "defined by the labels of the mask."
            ),
            "warning_message": None,
        }

        self._fit_cache()

        mask_logger("load_regions", self.labels_img, verbose=self.verbose)

        self.labels_img_ = deepcopy(self.labels_img)
        self.labels_img_ = check_niimg_3d(self.labels_img_)

        if self.labels:
            if self.lut is not None:
                raise ValueError(
                    "Pass either labels "
                    "or a lookup table (lut) to the masker, "
                    "but not both."
                )
            self._check_labels()
            if "background" in self.labels:
                idx = self.labels.index("background")
                self.labels[idx] = "Background"

        self.lut_ = generate_lut(
            self.labels_img_, self.background_label, self.lut, self.labels
        )

        self._original_region_ids = self.lut_["index"].to_list()

        if imgs is not None:
            imgs_ = check_niimg(imgs, atleast_4d=True)

        self.mask_img_ = self._load_mask(imgs)

        # Check shapes and affines for resample.
        if self.resampling_target is None:
            images = {"labels": self.labels_img_}
            if self.mask_img_ is not None:
                images["mask"] = self.mask_img_
            if imgs is not None:
                images["data"] = imgs_
            check_same_fov(raise_error=True, **images)

        # resample labels
        if (
            self.resampling_target == "data"
            and imgs is not None
            and not check_same_fov(
                imgs_,
                self.labels_img_,
            )
        ):
            self.labels_img_ = self._resample_labels(imgs_)

        # resample mask
        ref_img = None
        if self.resampling_target == "data" and imgs is not None:
            ref_img = imgs_
        elif self.resampling_target == "labels":
            ref_img = self.labels_img_
        if (
            self.mask_img_ is not None
            and ref_img is not None
            and not check_same_fov(
                ref_img,
                self.mask_img_,
            )
        ):
            mask_logger("resample_mask", verbose=self.verbose)

            # TODO (nilearn >= 0.13.0) force_resample=True
            self.mask_img_ = self._cache(resample_img, func_memory_level=2)(
                self.mask_img_,
                interpolation="nearest",
                target_shape=ref_img.shape[:3],
                target_affine=ref_img.affine,
                copy_header=True,
                force_resample=False,
            )

            # Just check that the mask is valid
            load_mask_img(self.mask_img_)

        if self.reports:
            self._reporting_data = {
                "labels_image": self.labels_img_,
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

        mask_logger("fit_done", verbose=self.verbose)

        return self

    def _check_labels(self):
        """Check labels.

        - checks that labels is a list of strings.
        """
        labels = self.labels
        if not isinstance(labels, list):
            raise TypeError(
                f"'labels' must be a list. Got: {type(labels)}",
            )
        if not all(isinstance(x, str) for x in labels):
            types_labels = {type(x) for x in labels}
            raise TypeError(
                "All elements of 'labels' must be a string.\n"
                f"Got a list of {types_labels}",
            )

    @fill_doc
    def fit_transform(self, imgs, y=None, confounds=None, sample_mask=None):
        """Prepare and perform signal extraction from regions.

        Parameters
        ----------
        imgs : 3D/4D Niimg-like object
            See :ref:`extracting_data`.
            Images to process.
            If a 3D niimg is provided, a 1D array is returned.

        y : None
            This parameter is unused. It is solely included for scikit-learn
            compatibility.

        region_ids_: dict[str | int, int | float] = {}
        if self.background_label in index:
            index.pop(index.index(self.background_label))
            region_ids_["background"] = self.background_label
        for i, id in enumerate(index):
            region_ids_[i] = id  # noqa : PERF403

        return region_ids_

        %(confounds)s

        %(sample_mask)s

            .. versionadded:: 0.8.0

        Returns
        -------
        %(signals_transform_nifti)s

        """
        del y
        return self.fit(imgs).transform(
            imgs, confounds=confounds, sample_mask=sample_mask
        )

    def __sklearn_is_fitted__(self):
        return hasattr(self, "labels_img_") and hasattr(self, "lut_")

    @fill_doc
    def transform_single_imgs(self, imgs, confounds=None, sample_mask=None):
        """Extract signals from a single 4D niimg.

        Parameters
        ----------
        imgs : 3D/4D Niimg-like object
            See :ref:`extracting_data`.
            Images to process.

        %(confounds)s

        %(sample_mask)s

            .. versionadded:: 0.8.0

        Attributes
        ----------
        region_atlas_ : Niimg-like object
            Regions definition as labels.
            The labels correspond to the indices in ``region_ids_``.
            The region in ``region_atlas_``
            that takes the value ``region_ids_[i]``
            is used to compute the signal in ``region_signal[:, i]``.

            .. versionadded:: 0.10.3

        Returns
        -------
        %(signals_transform_nifti)s

        """
        check_is_fitted(self)

        # imgs passed at transform time may be different
        # from those passed at fit time.
        # So it may be needed to resample mask and labels,
        # if 'data' is the resampling target.
        # We handle the resampling of labels and mask separately because the
        # affine of the labels and mask images should not impact the extraction
        # of the signal.
        #
        # Any resampling of the mask or labels is not 'kept' after transform,
        # to avoid modifying the masker after fit.
        #
        # If the resampling target is different,
        # then resampling was already done at fit time
        # (e.g resampling of the mask image to the labels image
        # if the target was 'labels'),
        # or resampling of the data will be done at extract time.
        labels_img_ = self.labels_img_
        mask_img_ = self.mask_img_
        if self.resampling_target == "data":
            imgs_ = check_niimg(imgs, atleast_4d=True)
            if not check_same_fov(
                imgs_,
                labels_img_,
            ):
                warnings.warn(
                    (
                        "Resampling labels at transform time...\n"
                        "To avoid this warning, make sure to pass the images "
                        "you want to transform to fit() first, "
                        "or directly use fit_transform()."
                    ),
                    stacklevel=find_stack_level(),
                )
                labels_img_ = self._resample_labels(imgs_)

            if (mask_img_ is not None) and (
                not check_same_fov(
                    imgs_,
                    mask_img_,
                )
            ):
                warnings.warn(
                    (
                        "Resampling mask at transform time...\n"
                        "To avoid this warning, make sure to pass the images "
                        "you want to transform to fit() first, "
                        "or directly use fit_transform()."
                    ),
                    stacklevel=find_stack_level(),
                )
                mask_img_ = self._cache(resample_img, func_memory_level=2)(
                    mask_img_,
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
            target_shape = labels_img_.shape[:3]
            target_affine = labels_img_.affine

        params = get_params(
            NiftiLabelsMasker,
            self,
            ignore=["resampling_target"],
        )
        params["target_shape"] = target_shape
        params["target_affine"] = target_affine
        params["clean_kwargs"] = self.clean_args_
        # TODO (nilearn  >= 0.13.0) remove
        if self.clean_kwargs:
            params["clean_kwargs"] = self.clean_kwargs_

        region_signals, (ids, masked_atlas) = self._cache(
            filter_and_extract,
            ignore=["verbose", "memory", "memory_level"],
        )(
            # Images
            imgs,
            _ExtractionFunctor(
                labels_img_,
                self.background_label,
                self.strategy,
                self.keep_masked_labels,
                mask_img_,
            ),
            # Pre-processing
            params,
            confounds=confounds,
            sample_mask=sample_mask,
            dtype=self.dtype,
            # Caching
            memory=self.memory_,
            memory_level=self.memory_level,
            verbose=self.verbose,
        )

        # Create a lut that may be different from the fitted lut_
        # and whose rows are sorted according
        # to the columns in the region_signals array.
        self._lut_ = self.lut_.copy()

        labels = set(np.unique(safe_get_data(self.labels_img_)))
        desired_order = [*ids]
        if self.background_label in labels:
            desired_order = [self.background_label, *ids]

        mask = self.lut_["index"].isin(desired_order)
        self._lut_ = self._lut_[mask]
        self._lut_ = sanitize_look_up_table(
            self._lut_, atlas=np.array(desired_order)
        )
        self._lut_ = (
            self._lut_.set_index("index").loc[desired_order].reset_index()
        )

        self.region_atlas_ = masked_atlas

        return region_signals

    def _resample_labels(self, imgs_):
        mask_logger("resample_regions", verbose=self.verbose)

        labels_before_resampling = set(
            np.unique(safe_get_data(self.labels_img_))
        )
        labels_img_ = self._cache(resample_img, func_memory_level=2)(
            self.labels_img_,
            interpolation="nearest",
            target_shape=imgs_.shape[:3],
            target_affine=imgs_.affine,
            copy_header=True,
            force_resample=False,
        )
        labels_after_resampling = set(np.unique(safe_get_data(labels_img_)))
        if labels_diff := labels_before_resampling.difference(
            labels_after_resampling
        ):
            warnings.warn(
                "After resampling the label image to the data image, "
                f"the following labels were removed: {labels_diff}. "
                "Label image only contains "
                f"{len(labels_after_resampling)} labels "
                "(including background).",
                stacklevel=find_stack_level(),
            )

        return labels_img_

    @fill_doc
    def inverse_transform(self, signals):
        """Compute :term:`voxel` signals from region signals.

        Any mask given at initialization is taken into account.

        .. versionchanged:: 0.9.2

            This method now supports 1D arrays, which will produce 3D images.

        Parameters
        ----------
        %(signals_inv_transform)s

        Returns
        -------
        %(img_inv_transform_nifti)s

        """
        from ..regions import signal_extraction

        check_is_fitted(self)

        signals = self._check_array(signals)

        mask_logger("inverse_transform", verbose=self.verbose)

        return signal_extraction.signals_to_img_labels(
            signals,
            self.labels_img_,
            self.mask_img_,
            background_label=self.background_label,
        )

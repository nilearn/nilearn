"""Mixin classes for maskers."""

import abc
import itertools
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.utils.estimator_checks import check_is_fitted

from nilearn._utils.bids import (
    generate_atlas_look_up_table,
    sanitize_look_up_table,
)
from nilearn._utils.docs import fill_doc
from nilearn._utils.numpy_conversions import csv_to_array
from nilearn.image import high_variance_confounds
from nilearn.image.image import get_indices_from_image, iter_check_niimg
from nilearn.surface.surface import SurfaceImage
from nilearn.typing import NiimgLike


class _MultiMixin:
    """Mixin class to add common MultiMasker functionalities."""

    @fill_doc
    def fit_transform(self, imgs, y=None, confounds=None, sample_mask=None):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        imgs : Image object, or a :obj:`list` of Image objects
            See :ref:`extracting_data`.
            Data to be preprocessed

        y : None
            This parameter is unused.
            It is solely included for scikit-learn compatibility.

        %(confounds_multi)s

        %(sample_mask_multi)s

            .. nilearn_versionadded:: 0.8.0

        Returns
        -------
        %(signals_transform_multi_nifti)s
        """
        return self.fit(imgs, y=y).transform(
            imgs, confounds=confounds, sample_mask=sample_mask
        )

    @fill_doc
    def transform_imgs(
        self, imgs_list, confounds=None, n_jobs=1, sample_mask=None
    ):
        """Extract signals from a list of 4D niimgs.

        Parameters
        ----------
        %(imgs)s
            Images to process.

        %(confounds_multi)s

        %(n_jobs)s

        %(sample_mask_multi)s

        Returns
        -------
        %(signals_transform_imgs_multi_nifti)s

        """
        # We handle the resampling of maps and mask separately because the
        # affine of the maps and mask images should not impact the extraction
        # of the signal.

        check_is_fitted(self)

        niimg_iter = iter_check_niimg(
            imgs_list,
            ensure_ndim=None,
            atleast_4d=False,
            memory=self.memory_,
            memory_level=self.memory_level,
        )

        confounds = self._prepare_confounds(imgs_list, confounds)

        sample_mask = self._prepare_sample_mask(imgs_list, sample_mask)

        # rely on the transform_single_imgs method
        # defined in each child class
        func = self._cache(self.transform_single_imgs)

        region_signals = Parallel(n_jobs=n_jobs)(
            delayed(func)(imgs=imgs, confounds=cfs, sample_mask=sms)
            for imgs, cfs, sms in zip(
                niimg_iter, confounds, sample_mask, strict=False
            )
        )
        return region_signals

    @fill_doc
    def transform(self, imgs, confounds=None, sample_mask=None):
        """Apply mask, spatial and temporal preprocessing.

        Parameters
        ----------
        imgs :Image object, or a :obj:`list` of Image objects
            See :ref:`extracting_data`.
            Data to be preprocessed

        %(confounds_multi)s

        %(sample_mask_multi)s

        Returns
        -------
        %(signals_transform_multi_nifti)s

        """
        check_is_fitted(self)

        if not (confounds is None or isinstance(confounds, list)):
            raise TypeError(
                "'confounds' must be a None or a list. "
                f"Got {confounds.__class__.__name__}."
            )
        if not (sample_mask is None or isinstance(sample_mask, list)):
            raise TypeError(
                "'sample_mask' must be a None or a list. "
                f"Got {sample_mask.__class__.__name__}."
            )
        if isinstance(imgs, (*NiimgLike, SurfaceImage)):
            if isinstance(confounds, list):
                confounds = confounds[0]
            if isinstance(sample_mask, list):
                sample_mask = sample_mask[0]
            return super().transform(
                imgs, confounds=confounds, sample_mask=sample_mask
            )

        # TODO throw a proper error
        # check we have consistent type
        if isinstance(imgs[0], SurfaceImage):
            assert all(isinstance(x, SurfaceImage) for x in imgs)

        return self.transform_imgs(
            imgs,
            confounds=confounds,
            sample_mask=sample_mask,
            n_jobs=self.n_jobs,
        )

    def _prepare_confounds(self, imgs_list, confounds):
        """Check and prepare confounds."""
        if confounds is None:
            confounds = list(itertools.repeat(None, len(imgs_list)))
        elif len(confounds) != len(imgs_list):
            raise ValueError(
                f"number of confounds ({len(confounds)}) unequal to "
                f"number of images ({len(imgs_list)})."
            )

        if self.high_variance_confounds:
            for i, img in enumerate(imgs_list):
                hv_confounds = self._cache(high_variance_confounds)(img)

                if confounds[i] is None:
                    confounds[i] = hv_confounds
                elif isinstance(confounds[i], list):
                    confounds[i] += hv_confounds
                elif isinstance(confounds[i], np.ndarray):
                    confounds[i] = np.hstack([confounds[i], hv_confounds])
                elif isinstance(confounds[i], pd.DataFrame):
                    confounds[i] = np.hstack(
                        [confounds[i].to_numpy(), hv_confounds]
                    )
                elif isinstance(confounds[i], (str, Path)):
                    c = csv_to_array(confounds[i])
                    if np.isnan(c.flat[0]):
                        # There may be a header
                        c = csv_to_array(confounds[i], skip_header=1)
                    confounds[i] = np.hstack([c, hv_confounds])
                else:
                    confounds[i].append(hv_confounds)

        return confounds

    def _prepare_sample_mask(self, imgs_list, sample_mask):
        """Check and prepare sample_mask."""
        if sample_mask is None:
            sample_mask = itertools.repeat(None, len(imgs_list))
        elif len(sample_mask) != len(imgs_list):
            raise ValueError(
                f"number of sample_mask ({len(sample_mask)}) unequal to "
                f"number of images ({len(imgs_list)})."
            )

        return sample_mask

    def set_output(self, *, transform=None):
        """Set the output container when ``"transform"`` is called.

        .. warning::

            This has not been implemented yet.
        """
        raise NotImplementedError()


class _LabelMaskerMixin:
    lut_: pd.DataFrame
    _lut_: pd.DataFrame
    background_label: int | float

    @property
    def n_elements_(self) -> int:
        """Return number of regions.

        This is equal to the number of unique values
        in the fitted label image,
        minus the background value.
        """
        check_is_fitted(self)
        lut = self.lut_
        if hasattr(self, "_lut_"):
            lut = self._lut_
        return len(lut[lut["index"] != self.background_label])

    @property
    def labels_(self) -> list[int | float]:
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
        """
        check_is_fitted(self)

        index = self.labels_
        valid_ids = [id for id in index if id != self.background_label]

        sub_df = self.lut_[self.lut_["index"].isin(valid_ids)]

        return sub_df["name"].reset_index(drop=True).to_dict()

    @property
    def region_ids_(self) -> dict[str | int, int | float]:
        """Return dictionary containing the region ids corresponding \
           to each column in the array \
           returned by `transform`.

        The region id corresponding to ``region_signal[:,i]``
        is ``region_ids_[i]``.
        ``region_ids_['background']`` is the background label.
        """
        check_is_fitted(self)

        index = self.labels_

        region_ids_: dict[str | int, int | float] = {}
        if self.background_label in index:
            index.pop(index.index(self.background_label))
            region_ids_["background"] = self.background_label
        for i, id in enumerate(index):
            region_ids_[i] = id  # noqa : PERF403

        return region_ids_

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features :default=None
            Only for sklearn API compatibility.
        """
        del input_features
        return np.asarray(self.region_names_.values(), dtype=object)

    def _generate_lut(self):
        """Generate a look up table if one was not provided.

        Also sanitize its content if necessary.
        """
        labels_img = self.labels_img_
        background_label = self.background_label
        lut = self.lut
        labels = self.labels

        labels_present = get_indices_from_image(labels_img)
        add_background_to_lut = (
            None
            if background_label not in labels_present
            else background_label
        )

        if lut is not None:
            if isinstance(lut, (str, Path)):
                lut = pd.read_table(lut, sep=None, engine="python")

        elif labels:
            lut = generate_atlas_look_up_table(
                function=None,
                name=deepcopy(labels),
                index=labels_img,
                background_label=add_background_to_lut,
            )

        else:
            lut = generate_atlas_look_up_table(
                function=None,
                index=labels_img,
                background_label=add_background_to_lut,
            )

        assert isinstance(lut, pd.DataFrame)

        # passed labels or lut may not include background label
        # because of poor data standardization
        # so we need to update the lut accordingly
        mask_background_index = lut["index"] == background_label
        if (mask_background_index).any():
            # Ensure background is the first row with name "Background"
            # Shift the 'name' column down by one
            # if background row was not named properly
            first_rows = lut[mask_background_index]
            other_rows = lut[~mask_background_index]
            lut = pd.concat([first_rows, other_rows], ignore_index=True)

            lut.loc[0, "name"] = "Background"

        else:
            first_row = {
                "name": "Background",
                "index": background_label,
                "color": "FFFFFF",
            }
            first_row = {
                col: first_row.get(col, np.nan) for col in lut.columns
            }
            lut = pd.concat(
                [pd.DataFrame([first_row]), lut], ignore_index=True
            )

        return (
            sanitize_look_up_table(lut, atlas=labels_img)
            .sort_values("index")
            .reset_index(drop=True)
        )


class _ReportingMixin:
    """A mixin class to be used with classes that require reporting
    functionality.

    ReportingMixin uses one public attribute of type bool:

    reports: The value to indicate if reporting is enabled.

    ReportingMixin uses two private attributes of type dict:

    _report_content : The content to enrich the report. Some fields can be
    initialized in estimator constructor. Others can be added after model fit.
    Possible generic keys are:
        - title : title to be used for the report
        - description : description of the report generated by the estimator
        - summary : summary
        - warning_message : possible warning message
    _report_data : Contains data from model fit. If reporting is disabled, or
    the model is not fit, this attribute does not exist.

    Classes inheriting from ReportingMixin should implement ``_get_displays``
    to return the displays to be embedded to the report.
    """

    _report_content: ClassVar[dict[str, Any]] = {}

    def _has_report_data(self):
        """
        Check if the model is fitted and _reporting_data is populated.

        Returns
        -------
        bool
            True if reporting is enabled, the model is fitted and
        _reporting_data is populated; False otherwise.
        """
        return hasattr(self, "_reporting_data")

    def generate_report(self, title: str | None = None):
        """Generate an HTML report for the current object.

        Parameters
        ----------
        title : :obj:`str` or None, default=None
            title for the report. If None, title will be the class name.

        Returns
        -------
        report : `nilearn.reporting.html_report.HTMLReport`
            HTML report for the masker.
        """
        from nilearn.reporting.html_report import generate_report

        self._report_content["title"] = title

        return generate_report(self)

    @abc.abstractmethod
    def _reporting(self):
        raise NotImplementedError()

"""Mixin classes for maskers."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.estimator_checks import check_is_fitted

from nilearn._utils.bids import (
    generate_atlas_look_up_table,
    sanitize_look_up_table,
)
from nilearn._utils.docs import fill_doc
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn.image.image import get_indices_from_image


class _MultiMixin:
    """Mixin class to add common MultiMasker functionalities."""

    def __sklearn_tags__(self):
        """Return estimator tags.

        See the sklearn documentation for more details on tags
        https://scikit-learn.org/1.6/developers/develop.html#estimator-tags
        """
        # TODO (sklearn  >= 1.6.0) remove if block
        if SKLEARN_LT_1_6:
            from nilearn._utils.tags import tags

            return tags(masker=True, multi_masker=True)

        from nilearn._utils.tags import InputTags

        tags = super().__sklearn_tags__()
        tags.input_tags = InputTags(masker=True, multi_masker=True)
        return tags

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

            .. versionadded:: 0.8.0

        Returns
        -------
        %(signals_transform_multi_nifti)s
        """
        return self.fit(imgs, y=y).transform(
            imgs, confounds=confounds, sample_mask=sample_mask
        )

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

        .. versionadded:: 0.10.3
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

        Parameters
        ----------
        labels_img : Nifti1Image | SurfaceImage

        background_label : int | float

        lut : Optional[str, Path, pd.DataFrame]

        labels : Optional[list[str]]
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

            mask_background_name = lut["name"] == "Background"
            if not (mask_background_name).any():
                lut["name"] = lut["name"].shift(1)

            lut.loc[0, "name"] = "Background"

        else:
            first_row = {
                "name": "Background",
                "index": background_label,
                "color": "FFFFFF",
            }
            first_row = {
                col: first_row[col] if col in lut else np.nan
                for col in lut.columns
            }
            lut = pd.concat(
                [pd.DataFrame([first_row]), lut], ignore_index=True
            )

        return (
            sanitize_look_up_table(lut, atlas=labels_img)
            .sort_values("index")
            .reset_index(drop=True)
        )

"""Masker for surface objects."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.base import BaseEstimator, TransformerMixin

from nilearn import signal
from nilearn._utils.cache_mixin import CacheMixin, cache
from nilearn._utils.class_inspect import get_params
from nilearn.experimental.surface._surface_image import PolyMesh, SurfaceImage


def check_same_n_vertices(mesh_1: PolyMesh, mesh_2: PolyMesh) -> None:
    """Check that 2 meshes have the same keys and that n vertices match."""
    keys_1, keys_2 = set(mesh_1.keys()), set(mesh_2.keys())
    if keys_1 != keys_2:
        diff = keys_1.symmetric_difference(keys_2)
        raise ValueError(
            "Meshes do not have the same keys. " f"Offending keys: {diff}"
        )
    for key in keys_1:
        if mesh_1[key].n_vertices != mesh_2[key].n_vertices:
            raise ValueError(
                f"Number of vertices do not match for '{key}'."
                f"number of vertices in mesh_1: {mesh_1[key].n_vertices}; "
                f"in mesh_2: {mesh_2[key].n_vertices}"
            )


class SurfaceMasker(BaseEstimator, TransformerMixin, CacheMixin):
    """Extract data from a SurfaceImage."""

    mask_img: SurfaceImage | None

    mask_img_: SurfaceImage | None
    output_dimension_: int | None

    def __init__(
        self,
        mask_img=None,
        standardize=False,
        standardize_confounds=True,
        detrend=False,
        high_variance_confounds=False,
        low_pass=None,
        high_pass=None,
        t_r=None,
        memory_level=1,
        memory=Memory(location=None),
        **kwargs,
    ):
        self.mask_img = mask_img
        self.standardize = standardize
        self.standardize_confounds = standardize_confounds
        self.high_variance_confounds = high_variance_confounds
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r

        self.memory = memory
        self.memory_level = memory_level
        self._shelving = False
        self.clean_kwargs = {
            k[7:]: v for k, v in kwargs.items() if k.startswith("clean__")
        }

    def _fit_mask_img(self, img: SurfaceImage | None) -> None:
        if self.mask_img is not None:
            if img is not None:
                check_same_n_vertices(self.mask_img.mesh, img.mesh)
            self.mask_img_ = self.mask_img
            return
        if img is None:
            raise ValueError(
                "Please provide either a mask_img when initializing "
                "the masker or an img when calling fit()."
            )
        # TODO: don't store a full array of 1 to mean "no masking"; use some
        # sentinel value
        mask_data = {
            k: np.ones(v.n_vertices, dtype=bool) for (k, v) in img.mesh.items()
        }
        self.mask_img_ = SurfaceImage(mesh=img.mesh, data=mask_data)

    def fit(
        self, img: SurfaceImage | None = None, y: Any = None
    ) -> SurfaceMasker:
        """Prepare signal extraction from regions.

        Parameters
        ----------
        img : SurfaceImage object
            Mesh and data for both hemispheres.

        y : None
            This parameter is unused. It is solely included for scikit-learn
            compatibility.

        Returns
        -------
        SurfaceMasker object
        """
        del y
        self._fit_mask_img(img)
        assert self.mask_img_ is not None
        start, stop = 0, 0
        self.slices = {}
        for part_name, mask in self.mask_img_.data.items():
            assert isinstance(mask, np.ndarray)
            stop = start + mask.sum()
            self.slices[part_name] = start, stop
            start = stop
        self.output_dimension_ = stop
        return self

    def _check_fitted(self):
        if not hasattr(self, "mask_img_"):
            raise ValueError(
                "This masker has not been fitted. Call fit "
                "before calling transform."
            )

    def transform(
        self,
        img: SurfaceImage,
        confounds: pd.DataFrame | None = None,
        sample_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Extract signals from fitted surface object.

        Parameters
        ----------
        img : SurfaceImage object
            Mesh and data for both hemispheres.

        Returns
        -------
        output : numpy.ndarray
            Signal for each element.
            shape: (img data shape, total number of vertices)
        """
        parameters = get_params(
            self.__class__,
            self,
            ignore=[
                "mask_img",
            ],
        )
        parameters["clean_kwargs"] = self.clean_kwargs

        self._check_fitted()
        assert self.mask_img_ is not None
        assert self.output_dimension_ is not None
        check_same_n_vertices(self.mask_img_.mesh, img.mesh)
        output = np.empty((*img.shape[:-1], self.output_dimension_))
        for part_name, (start, stop) in self.slices.items():
            mask = self.mask_img_.data[part_name]
            assert isinstance(mask, np.ndarray)
            output[..., start:stop] = img.data[part_name][..., mask]

        # signal cleaning here
        output = cache(
            signal.clean,
            memory=self.memory,
            func_memory_level=2,
            memory_level=self.memory_level,
            shelve=self._shelving,
        )(
            output,
            detrend=parameters["detrend"],
            standardize=parameters["standardize"],
            standardize_confounds=parameters["standardize_confounds"],
            t_r=parameters["t_r"],
            low_pass=parameters["low_pass"],
            high_pass=parameters["high_pass"],
            confounds=confounds,
            sample_mask=sample_mask,
            **parameters["clean_kwargs"],
        )
        return output

    def fit_transform(
        self,
        img: SurfaceImage,
        y: Any = None,
        confounds: pd.DataFrame | None = None,
        sample_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Prepare and perform signal extraction from regions.

        Parameters
        ----------
        img : SurfaceImage object
            Mesh and data for both hemispheres.

        y : None
            This parameter is unused. It is solely included for scikit-learn
            compatibility.

        Returns
        -------
        numpy.ndarray
            Signal for each element.
            shape: (img data shape, total number of vertices)
        """
        del y
        return self.fit(img).transform(img, confounds, sample_mask)

    def inverse_transform(self, masked_img: np.ndarray) -> SurfaceImage:
        """Transform extracted signal back to surface object.

        Parameters
        ----------
        masked_img : numpy.ndarray
            Extracted signal.

        Returns
        -------
        SurfaceImage object
            Mesh and data for both hemispheres.
        """
        self._check_fitted()
        assert self.mask_img_ is not None
        if masked_img.shape[-1] != self.output_dimension_:
            raise ValueError(
                "Input to inverse_transform has wrong shape; "
                f"last dimension should be {self.output_dimension_}"
            )
        data = {}
        for part_name, mask in self.mask_img_.data.items():
            assert isinstance(mask, np.ndarray)
            data[part_name] = np.zeros(
                (*masked_img.shape[:-1], mask.shape[0]),
                dtype=masked_img.dtype,
            )
            start, stop = self.slices[part_name]
            data[part_name][..., mask] = masked_img[..., start:stop]
        return SurfaceImage(mesh=self.mask_img_.mesh, data=data)


class SurfaceLabelsMasker:
    """Extract data from a SurfaceImage, averaging over atlas regions.

    Parameters
    ----------
    labels_img : SurfaceImage object
        Region definitions, as one image of labels.

    label_names : :obj:`list` of :obj:`str`, default=None
        Full labels corresponding to the labels image.

    Attributes
    ----------
    labels_data_ : :obj:`numpy.ndarray`

    labels_ : :obj:`numpy.ndarray`

    label_names_ : :obj:`numpy.ndarray`
    """

    # TODO check attribute names after PR 3761 and harmonize with volume labels
    # masker if necessary.
    labels_img: SurfaceImage
    label_names: dict[Any, str] | None

    labels_data_: type[np.ndarray]
    labels_: type[np.ndarray]
    label_names_: type[np.ndarray]

    def __init__(
        self,
        labels_img: SurfaceImage,
        label_names: dict[Any, str] | None = None,
    ) -> None:
        self.labels_img = labels_img
        self.label_names = label_names
        self.labels_data_ = np.concatenate(list(labels_img.data.values()))
        all_labels = set(self.labels_data_.ravel())
        all_labels.discard(0)
        self.labels_ = np.asarray(list(all_labels))
        if label_names is None:
            self.label_names_ = np.asarray(
                [str(label) for label in self.labels_]
            )
        else:
            self.label_names_ = np.asarray(
                [label_names[label] for label in self.labels_]
            )

    def fit(
        self, img: SurfaceImage | None = None, y: Any = None
    ) -> SurfaceLabelsMasker:
        """Prepare signal extraction from regions.

        Parameters
        ----------
        img : SurfaceImage object
            Mesh and data for both hemispheres.

        y : None
            This parameter is unused. It is solely included for scikit-learn
            compatibility.

        Returns
        -------
        SurfaceLabelsMasker object
        """
        del img, y
        return self

    def transform(self, img: SurfaceImage) -> np.ndarray:
        """Extract signals from fitted surface object.

        Parameters
        ----------
        img : SurfaceImage object
            Mesh and data for both hemispheres.

        Returns
        -------
        output : numpy.ndarray
            Signal for each element.
            shape: (img data shape, total number of vertices)
        """
        check_same_n_vertices(self.labels_img.mesh, img.mesh)
        img_data = np.concatenate(list(img.data.values()), axis=-1)
        output = np.empty((*img_data.shape[:-1], len(self.labels_)))
        for i, label in enumerate(self.labels_):
            output[..., i] = img_data[..., self.labels_data_ == label].mean(
                axis=-1
            )
        return output

    def fit_transform(self, img: SurfaceImage, y: Any = None) -> np.ndarray:
        """Prepare and perform signal extraction from regions.

        Parameters
        ----------
        img : SurfaceImage object
            Mesh and data for both hemispheres.

        y : None
            This parameter is unused. It is solely included for scikit-learn
            compatibility.

        Returns
        -------
        numpy.ndarray
            Signal for each element.
            shape: (img data shape, total number of vertices)
        """
        del y
        return self.fit(img).transform(img)

    def inverse_transform(self, masked_img: np.ndarray) -> SurfaceImage:
        """Transform extracted signal back to surface object.

        Parameters
        ----------
        masked_img : numpy.ndarray
            Extracted signal.

        Returns
        -------
        SurfaceImage object
            Mesh and data for both hemispheres.
        """
        data = {}
        for part_name, labels_part in self.labels_img.data.items():
            data[part_name] = np.zeros(
                (*masked_img.shape[:-1], labels_part.shape[0]),
                dtype=masked_img.dtype,
            )
            for label_idx, label in enumerate(self.labels_):
                data[part_name][..., labels_part == label] = masked_img[
                    ..., label_idx
                ]
        return SurfaceImage(mesh=self.labels_img.mesh, data=data)

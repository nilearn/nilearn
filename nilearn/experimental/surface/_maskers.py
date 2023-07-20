from __future__ import annotations

from typing import Any

import numpy as np

from nilearn.experimental.surface._surface_image import SurfaceImage


class SurfaceMasker:
    # TODO:
    # - allow passing an actual roi mask
    # - don't store a full array of 1 to mean "no masking"
    # - 1d output for 1d images

    mask_img_: SurfaceImage | None
    output_dimension_: int | None

    def _fit_mask_img(self, img: SurfaceImage) -> None:
        mask_data = {
            k: np.ones(v.shape[-1], dtype=bool) for (k, v) in img.data.items()
        }
        self.mask_img_ = SurfaceImage(
            data=mask_data, mesh=img.mesh
        )  # type: ignore

    def fit(self, img: SurfaceImage, y: Any = None) -> SurfaceMasker:
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

    def transform(self, img: SurfaceImage) -> np.ndarray:
        assert self.mask_img_ is not None
        assert self.output_dimension_ is not None
        output = np.empty((img.shape[0], self.output_dimension_))
        for part_name, (start, stop) in self.slices.items():
            mask = self.mask_img_.data[part_name]
            assert isinstance(mask, np.ndarray)
            output[:, start:stop] = np.atleast_2d(img.data[part_name])[:, mask]
        return output

    def fit_transform(self, img: SurfaceImage, y: Any = None) -> np.ndarray:
        del y
        return self.fit(img).transform(img)

    def inverse_transform(self, masked_images: np.ndarray) -> SurfaceImage:
        assert self.mask_img_ is not None
        is_2d = len(masked_images.shape) > 1
        masked_images = np.atleast_2d(masked_images)
        data = {}
        for part_name, mask in self.mask_img_.data.items():
            assert isinstance(mask, np.ndarray)
            data[part_name] = np.zeros(
                (masked_images.shape[0], mask.shape[0]),
                dtype=masked_images.dtype,
            )
            start, stop = self.slices[part_name]
            data[part_name][:, mask] = masked_images[:, start:stop]
            if not is_2d:
                data[part_name] = data[part_name].squeeze()
        return SurfaceImage(
            data=data, mesh=self.mask_img_.mesh
        )  # type: ignore


class SurfaceLabelsMasker:
    labels_img: SurfaceImage
    labels_data_: np.ndarray
    labels_: np.ndarray
    label_names_: np.ndarray

    def __init__(
        self,
        labels_img: SurfaceImage,
        label_names: dict[Any, str] | None = None,
    ) -> None:
        self.labels_img = labels_img
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
        del img, y
        return self

    def transform(self, img: SurfaceImage) -> np.ndarray:
        img_data = np.hstack(
            [np.atleast_2d(data_part) for data_part in img.data.values()]
        )
        output = np.empty((img_data.shape[0], len(self.labels_)))
        for i, label in enumerate(self.labels_):
            output[:, i] = img_data[:, self.labels_data_ == label].mean(axis=1)
        return output

    def fit_transform(self, img: SurfaceImage, y: Any = None) -> np.ndarray:
        del y
        return self.fit(img).transform(img)

    def inverse_transform(self, masked_images: np.ndarray) -> SurfaceImage:
        is_2d = len(masked_images.shape) > 1
        masked_images = np.atleast_2d(masked_images)
        data = {}
        for part_name, labels_part in self.labels_img.data.items():
            data[part_name] = np.zeros(
                (masked_images.shape[0], labels_part.shape[0]),
                dtype=masked_images.dtype,
            )
            for label_idx, label in enumerate(self.labels_):
                data[part_name][:, labels_part == label] = masked_images[
                    :, label_idx
                ]
            if not is_2d:
                data[part_name] = data[part_name].squeeze()
        return SurfaceImage(
            data=data, mesh=self.labels_img.mesh
        )  # type: ignore

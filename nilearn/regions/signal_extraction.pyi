from collections.abc import Callable
from typing import List, Optional, Tuple, Union

from nibabel.nifti1 import Nifti1Image
from numpy import int32, int64, memmap, ndarray, uint8

def _check_affine_equality(img1: Nifti1Image, img2: Nifti1Image) -> None: ...
def _check_reduction_strategy(strategy: str) -> None: ...
def _check_shape_and_affine_compatibility(
    img1: Nifti1Image,
    img2: str | Nifti1Image | None = ...,
    dim: int | None = ...,
) -> bool: ...
def _check_shape_compatibility(
    img1: Nifti1Image, img2: str | Nifti1Image, dim: int | None = ...
) -> None: ...
def _get_labels_data(
    target_img: Nifti1Image,
    labels_img: Nifti1Image,
    mask_img: str | Nifti1Image | None = ...,
    background_label: int = ...,
    dim: None = ...,
) -> (
    tuple[list[int32], ndarray]
    | tuple[list[int32], memmap]
    | tuple[list[uint8], ndarray]
): ...
def _trim_maps(
    maps: ndarray, mask: ndarray, keep_empty: bool = ..., order: str = ...
) -> tuple[ndarray, ndarray, ndarray]: ...
def img_to_signals_labels(
    imgs: str | Nifti1Image,
    labels_img: Nifti1Image,
    mask_img: Nifti1Image | None = ...,
    background_label: int = ...,
    order: str = ...,
    strategy: str = ...,
) -> tuple[ndarray, list[int32]] | tuple[ndarray, list[uint8]]: ...
def img_to_signals_maps(
    imgs: Callable | Nifti1Image,
    maps_img: Nifti1Image,
    mask_img: Nifti1Image | None = ...,
) -> tuple[ndarray, list[int64]]: ...
def signals_to_img_labels(
    signals: ndarray | Nifti1Image,
    labels_img: str | Nifti1Image,
    mask_img: str | Nifti1Image | None = ...,
    background_label: int = ...,
    order: str = ...,
) -> Nifti1Image: ...
def signals_to_img_maps(
    region_signals: ndarray,
    maps_img: Nifti1Image,
    mask_img: Nifti1Image | None = ...,
) -> Nifti1Image: ...

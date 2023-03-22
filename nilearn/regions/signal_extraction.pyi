from typing import Callable, List, Optional, Tuple, Union

from nibabel.nifti1 import Nifti1Image
from numpy import int32, int64, memmap, ndarray, uint8

def _check_affine_equality(img1: Nifti1Image, img2: Nifti1Image) -> None: ...
def _check_reduction_strategy(strategy: str) -> None: ...
def _check_shape_and_affine_compatibility(
    img1: Nifti1Image,
    img2: Optional[Union[str, Nifti1Image]] = ...,
    dim: Optional[int] = ...,
) -> bool: ...
def _check_shape_compatibility(
    img1: Nifti1Image, img2: Union[str, Nifti1Image], dim: Optional[int] = ...
) -> None: ...
def _get_labels_data(
    target_img: Nifti1Image,
    labels_img: Nifti1Image,
    mask_img: Optional[Union[str, Nifti1Image]] = ...,
    background_label: int = ...,
    dim: None = ...,
) -> Union[
    Tuple[List[int32], ndarray],
    Tuple[List[int32], memmap],
    Tuple[List[uint8], ndarray],
]: ...
def _trim_maps(
    maps: ndarray, mask: ndarray, keep_empty: bool = ..., order: str = ...
) -> Tuple[ndarray, ndarray, ndarray]: ...
def img_to_signals_labels(
    imgs: Union[str, Nifti1Image],
    labels_img: Nifti1Image,
    mask_img: Optional[Nifti1Image] = ...,
    background_label: int = ...,
    order: str = ...,
    strategy: str = ...,
) -> Union[Tuple[ndarray, List[int32]], Tuple[ndarray, List[uint8]]]: ...
def img_to_signals_maps(
    imgs: Union[Callable, Nifti1Image],
    maps_img: Nifti1Image,
    mask_img: Optional[Nifti1Image] = ...,
) -> Tuple[ndarray, List[int64]]: ...
def signals_to_img_labels(
    signals: Union[ndarray, Nifti1Image],
    labels_img: Union[str, Nifti1Image],
    mask_img: Optional[Union[str, Nifti1Image]] = ...,
    background_label: int = ...,
    order: str = ...,
) -> Nifti1Image: ...
def signals_to_img_maps(
    region_signals: ndarray,
    maps_img: Nifti1Image,
    mask_img: Optional[Nifti1Image] = ...,
) -> Nifti1Image: ...

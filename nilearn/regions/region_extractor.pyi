from typing import List, Optional, Tuple, Union

from joblib.memory import Memory
from nibabel.nifti1 import Nifti1Image
from numpy import ndarray

def _remove_small_regions(
    input_data: ndarray, index: ndarray, affine: ndarray, min_size: int
) -> ndarray: ...
def _threshold_maps_ratio(
    maps_img: Nifti1Image, threshold: str | float
) -> Nifti1Image: ...
def connected_label_regions(
    labels_img: Nifti1Image,
    min_size: int | str | None = ...,
    connect_diag: bool | None = ...,
    labels: str | ndarray | list[str] | None = ...,
) -> Nifti1Image: ...
def connected_regions(
    maps_img: Nifti1Image,
    min_region_size: int | float = ...,
    extract_type: str | int = ...,
    smoothing_fwhm: int | float = ...,
    mask_img: Nifti1Image | None = ...,
) -> tuple[Nifti1Image, list[int]]: ...

class RegionExtractor:
    def __init__(
        self,
        maps_img: Nifti1Image,
        mask_img: Nifti1Image | None = ...,
        min_region_size: int | float = ...,
        threshold: int | float | str | None = ...,
        thresholding_strategy: str = ...,
        extractor: str = ...,
        smoothing_fwhm: int | float = ...,
        standardize: bool = ...,
        detrend: bool = ...,
        low_pass: None = ...,
        high_pass: None = ...,
        t_r: None = ...,
        memory: Memory = ...,
        memory_level: int = ...,
        verbose: int = ...,
    ) -> None: ...
    def fit(self, X: None = ..., y: None = ...) -> RegionExtractor: ...

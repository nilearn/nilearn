from typing import List, Optional, Tuple, Union

from joblib.memory import Memory
from nibabel.nifti1 import Nifti1Image
from numpy import ndarray

def _remove_small_regions(
    input_data: ndarray, index: ndarray, affine: ndarray, min_size: int
) -> ndarray: ...
def _threshold_maps_ratio(
    maps_img: Nifti1Image, threshold: Union[str, float]
) -> Nifti1Image: ...
def connected_label_regions(
    labels_img: Nifti1Image,
    min_size: Optional[Union[int, str]] = ...,
    connect_diag: Optional[bool] = ...,
    labels: Optional[Union[str, ndarray, List[str]]] = ...,
) -> Nifti1Image: ...
def connected_regions(
    maps_img: Nifti1Image,
    min_region_size: Union[int, float] = ...,
    extract_type: Union[str, int] = ...,
    smoothing_fwhm: Union[int, float] = ...,
    mask_img: Optional[Nifti1Image] = ...,
) -> Tuple[Nifti1Image, List[int]]: ...

class RegionExtractor:
    def __init__(
        self,
        maps_img: Nifti1Image,
        mask_img: Optional[Nifti1Image] = ...,
        min_region_size: Union[int, float] = ...,
        threshold: Optional[Union[int, float, str]] = ...,
        thresholding_strategy: str = ...,
        extractor: str = ...,
        smoothing_fwhm: Union[int, float] = ...,
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

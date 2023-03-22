from typing import Optional, Tuple, Union

from joblib.memory import Memory
from nibabel.nifti1 import Nifti1Image
from nilearn.maskers.nifti_masker import NiftiMasker
from numpy import ndarray
from numpy.random.mtrand import RandomState

def _compute_loadings(components: ndarray, data: ndarray) -> ndarray: ...

class DictLearning:
    def __init__(
        self,
        n_components: int = ...,
        n_epochs: int = ...,
        alpha: int = ...,
        reduction_ratio: str = ...,
        dict_init: Optional[Nifti1Image] = ...,
        random_state: Optional[Union[RandomState, int]] = ...,
        batch_size: int = ...,
        method: str = ...,
        mask: Optional[Union[Nifti1Image, NiftiMasker]] = ...,
        smoothing_fwhm: Union[int, float] = ...,
        standardize: bool = ...,
        detrend: bool = ...,
        low_pass: None = ...,
        high_pass: None = ...,
        t_r: None = ...,
        target_affine: Optional[ndarray] = ...,
        target_shape: Optional[Tuple[int, int, int]] = ...,
        mask_strategy: str = ...,
        mask_args: None = ...,
        n_jobs: int = ...,
        verbose: int = ...,
        memory: Memory = ...,
        memory_level: int = ...,
    ) -> None: ...
    def _init_dict(self, data: ndarray) -> None: ...
    def _init_loadings(self, data: ndarray) -> None: ...
    def _raw_fit(self, data: ndarray) -> DictLearning: ...

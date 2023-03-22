from typing import Optional, Tuple, Union

from joblib.memory import Memory
from nibabel.nifti1 import Nifti1Image
from nilearn.maskers.multi_nifti_masker import MultiNiftiMasker
from numpy import ndarray
from numpy.random.mtrand import RandomState

class _MultiPCA:
    def __init__(
        self,
        n_components: int = ...,
        mask: Optional[Union[Nifti1Image, MultiNiftiMasker]] = ...,
        smoothing_fwhm: Optional[Union[int, float]] = ...,
        do_cca: bool = ...,
        random_state: Optional[Union[RandomState, int]] = ...,
        standardize: bool = ...,
        standardize_confounds: bool = ...,
        detrend: bool = ...,
        low_pass: None = ...,
        high_pass: None = ...,
        t_r: None = ...,
        target_affine: Optional[ndarray] = ...,
        target_shape: Optional[Tuple[int, int, int]] = ...,
        mask_strategy: str = ...,
        mask_args: None = ...,
        memory: Memory = ...,
        memory_level: int = ...,
        n_jobs: int = ...,
        verbose: int = ...,
    ) -> None: ...
    def _raw_fit(self, data: ndarray) -> ndarray: ...

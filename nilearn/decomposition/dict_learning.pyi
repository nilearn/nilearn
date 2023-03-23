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
        dict_init: Nifti1Image | None = ...,
        random_state: RandomState | int | None = ...,
        batch_size: int = ...,
        method: str = ...,
        mask: Nifti1Image | NiftiMasker | None = ...,
        smoothing_fwhm: int | float = ...,
        standardize: bool = ...,
        detrend: bool = ...,
        low_pass: None = ...,
        high_pass: None = ...,
        t_r: None = ...,
        target_affine: ndarray | None = ...,
        target_shape: tuple[int, int, int] | None = ...,
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

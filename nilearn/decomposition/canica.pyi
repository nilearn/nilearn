from joblib.memory import Memory
from nibabel.nifti1 import Nifti1Image
from nilearn.maskers.multi_nifti_masker import MultiNiftiMasker
from numpy import ndarray
from numpy.random.mtrand import RandomState

class CanICA:
    def __init__(
        self,
        mask: MultiNiftiMasker | Nifti1Image | None = ...,
        n_components: int = ...,
        smoothing_fwhm: int | float = ...,
        do_cca: bool = ...,
        threshold: str | float = ...,
        n_init: int = ...,
        random_state: int | RandomState | None = ...,
        standardize: bool = ...,
        standardize_confounds: bool = ...,
        detrend: bool = ...,
        low_pass: None = ...,
        high_pass: None = ...,
        t_r: None = ...,
        target_affine: ndarray | None = ...,
        target_shape: tuple[int, int, int] | None = ...,
        mask_strategy: str = ...,
        mask_args: None = ...,
        memory: Memory = ...,
        memory_level: int = ...,
        n_jobs: int = ...,
        verbose: int = ...,
    ) -> None: ...
    def _raw_fit(self, data: ndarray) -> CanICA: ...
    def _unmix_components(self, components: ndarray) -> None: ...

from joblib.memory import Memory
from nibabel.nifti1 import Nifti1Image
from nilearn.decomposition._multi_pca import _MultiPCA
from nilearn.decomposition.canica import CanICA
from nilearn.decomposition.dict_learning import DictLearning
from nilearn.maskers.multi_nifti_masker import MultiNiftiMasker
from nilearn.maskers.nifti_masker import NiftiMasker
from numpy import float64, ndarray, random

def _explained_variance(
    X: ndarray, components: ndarray, per_component: bool = ...
) -> float64 | ndarray: ...
def _fast_svd(
    X: ndarray,
    n_components: int,
    random_state: random.mtrand.RandomState | int | None = ...,
) -> tuple[ndarray, ndarray, ndarray]: ...
def _mask_and_reduce(
    masker: MultiNiftiMasker,
    imgs: Nifti1Image | list[str] | list[Nifti1Image],
    confounds: list[ndarray] | None = ...,
    reduction_ratio: str | float = ...,
    n_components: int | None = ...,
    random_state: int | random.mtrand.RandomState | None = ...,
    memory_level: int = ...,
    memory: Memory = ...,
    n_jobs: int = ...,
) -> ndarray: ...
def _mask_and_reduce_single(
    masker: MultiNiftiMasker,
    img: str | Nifti1Image,
    confound: ndarray | None,
    reduction_ratio: int | float | None = ...,
    n_samples: int | None = ...,
    memory: Memory | None = ...,
    memory_level: int = ...,
    random_state: int | random.mtrand.RandomState | None = ...,
) -> ndarray: ...

class _BaseDecomposition:
    def __init__(
        self,
        n_components: int = ...,
        random_state: random.mtrand.RandomState | int | None = ...,
        mask: None | (Nifti1Image | NiftiMasker | MultiNiftiMasker) = ...,
        smoothing_fwhm: int | float | None = ...,
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
    def _check_components_(self) -> None: ...
    def _raw_score(
        self, data: ndarray, per_component: bool = ...
    ) -> ndarray | float64: ...
    def fit(
        self,
        imgs: str | Nifti1Image | list[Nifti1Image],
        y: None = ...,
        confounds: list[ndarray] | None = ...,
    ) -> DictLearning | CanICA | _MultiPCA: ...
    def inverse_transform(
        self, loadings: list[ndarray]
    ) -> list[Nifti1Image]: ...
    def score(
        self,
        imgs: Nifti1Image | list[Nifti1Image],
        confounds: None = ...,
        per_component: bool = ...,
    ) -> ndarray | float64: ...
    def transform(
        self, imgs: list[Nifti1Image], confounds: None = ...
    ) -> list[ndarray]: ...

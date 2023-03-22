from typing import List, Optional, Tuple, Union

from joblib.memory import Memory
from nibabel.nifti1 import Nifti1Image
from nilearn.decomposition._multi_pca import _MultiPCA
from nilearn.decomposition.canica import CanICA
from nilearn.decomposition.dict_learning import DictLearning
from nilearn.maskers.multi_nifti_masker import MultiNiftiMasker
from nilearn.maskers.nifti_masker import NiftiMasker
from numpy import float64, ndarray
from numpy.random.mtrand import RandomState

def _explained_variance(
    X: ndarray, components: ndarray, per_component: bool = ...
) -> Union[float64, ndarray]: ...
def _fast_svd(
    X: ndarray,
    n_components: int,
    random_state: Optional[Union[random.mtrand.RandomState, int]] = ...,
) -> Tuple[ndarray, ndarray, ndarray]: ...
def _mask_and_reduce(
    masker: MultiNiftiMasker,
    imgs: Union[Nifti1Image, List[str], List[Nifti1Image]],
    confounds: Optional[List[ndarray]] = ...,
    reduction_ratio: Union[str, float] = ...,
    n_components: Optional[int] = ...,
    random_state: Optional[Union[int, random.mtrand.RandomState]] = ...,
    memory_level: int = ...,
    memory: Memory = ...,
    n_jobs: int = ...,
) -> ndarray: ...
def _mask_and_reduce_single(
    masker: MultiNiftiMasker,
    img: Union[str, Nifti1Image],
    confound: Optional[ndarray],
    reduction_ratio: Optional[Union[int, float]] = ...,
    n_samples: Optional[int] = ...,
    memory: Optional[Memory] = ...,
    memory_level: int = ...,
    random_state: Optional[Union[int, random.mtrand.RandomState]] = ...,
) -> ndarray: ...

class _BaseDecomposition:
    def __init__(
        self,
        n_components: int = ...,
        random_state: Optional[Union[RandomState, int]] = ...,
        mask: Optional[
            Union[Nifti1Image, NiftiMasker, MultiNiftiMasker]
        ] = ...,
        smoothing_fwhm: Optional[Union[int, float]] = ...,
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
    def _check_components_(self) -> None: ...
    def _raw_score(
        self, data: ndarray, per_component: bool = ...
    ) -> Union[ndarray, float64]: ...
    def fit(
        self,
        imgs: Union[str, Nifti1Image, List[Nifti1Image]],
        y: None = ...,
        confounds: Optional[List[ndarray]] = ...,
    ) -> Union[DictLearning, CanICA, _MultiPCA]: ...
    def inverse_transform(
        self, loadings: List[ndarray]
    ) -> List[Nifti1Image]: ...
    def score(
        self,
        imgs: Union[Nifti1Image, List[Nifti1Image]],
        confounds: None = ...,
        per_component: bool = ...,
    ) -> Union[ndarray, float64]: ...
    def transform(
        self, imgs: List[Nifti1Image], confounds: None = ...
    ) -> List[ndarray]: ...

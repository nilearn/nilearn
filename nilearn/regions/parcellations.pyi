from joblib.memory import Memory
from nibabel.nifti1 import Nifti1Image
from nilearn.maskers.nifti_labels_masker import NiftiLabelsMasker
from nilearn.regions.hierarchical_kmeans_clustering import HierarchicalKMeans
from nilearn.regions.rena_clustering import ReNA
from numpy import ndarray
from pandas.core.frame import DataFrame
from sklearn.cluster._agglomerative import AgglomerativeClustering
from sklearn.cluster._kmeans import MiniBatchKMeans
from typing import (
    Any,
    List,
    Optional,
    Tuple,
    Union,
)


def _check_parameters_transform(
    imgs: Union[List[Nifti1Image], Nifti1Image],
    confounds: Any
) -> Union[Tuple[List[Nifti1Image], List[List[DataFrame]], bool], Tuple[List[Nifti1Image], List[DataFrame], bool], Tuple[List[Nifti1Image], List[None], bool], Tuple[List[Nifti1Image], List[ndarray], bool]]: ...


def _estimator_fit(
    data: ndarray,
    estimator: Union[HierarchicalKMeans, ReNA, MiniBatchKMeans, AgglomerativeClustering],
    method: Optional[str] = ...
) -> ndarray: ...


def _labels_masker_extraction(
    img: Nifti1Image,
    masker: NiftiLabelsMasker,
    confound: Optional[ndarray]
) -> ndarray: ...


class Parcellations:
    def __init__(
        self,
        method: Optional[str],
        n_parcels: int = ...,
        random_state: int = ...,
        mask: Optional[Nifti1Image] = ...,
        smoothing_fwhm: float = ...,
        standardize: bool = ...,
        detrend: bool = ...,
        low_pass: None = ...,
        high_pass: None = ...,
        t_r: None = ...,
        target_affine: None = ...,
        target_shape: None = ...,
        mask_strategy: str = ...,
        mask_args: None = ...,
        scaling: bool = ...,
        n_iter: int = ...,
        memory: Memory = ...,
        memory_level: int = ...,
        n_jobs: int = ...,
        verbose: Union[bool, int] = ...
    ) -> None: ...
    def _check_fitted(self) -> None: ...
    def _raw_fit(self, data: ndarray) -> Parcellations: ...
    def fit_transform(
        self,
        imgs: Union[List[Nifti1Image], Nifti1Image],
        confounds: Optional[List[ndarray]] = ...
    ) -> Union[ndarray, List[ndarray]]: ...
    def inverse_transform(
        self,
        signals: Union[ndarray, List[ndarray]]
    ) -> Union[List[Nifti1Image], Nifti1Image]: ...
    def transform(
        self,
        imgs: Union[List[Nifti1Image], Nifti1Image],
        confounds: Optional[List[ndarray]] = ...
    ) -> Union[ndarray, List[ndarray]]: ...

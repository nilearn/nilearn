from typing import Any, List, Optional, Tuple, Union

from joblib.memory import Memory
from nibabel.nifti1 import Nifti1Image
from nilearn.maskers.nifti_labels_masker import NiftiLabelsMasker
from nilearn.regions.hierarchical_kmeans_clustering import HierarchicalKMeans
from nilearn.regions.rena_clustering import ReNA
from numpy import ndarray
from pandas.core.frame import DataFrame
from sklearn.cluster._agglomerative import AgglomerativeClustering
from sklearn.cluster._kmeans import MiniBatchKMeans

def _check_parameters_transform(
    imgs: list[Nifti1Image] | Nifti1Image, confounds: Any
) -> (
    tuple[list[Nifti1Image], list[list[DataFrame]], bool]
    | tuple[list[Nifti1Image], list[DataFrame], bool]
    | tuple[list[Nifti1Image], list[None], bool]
    | tuple[list[Nifti1Image], list[ndarray], bool]
): ...
def _estimator_fit(
    data: ndarray,
    estimator: (
        HierarchicalKMeans | ReNA | MiniBatchKMeans | AgglomerativeClustering
    ),
    method: str | None = ...,
) -> ndarray: ...
def _labels_masker_extraction(
    img: Nifti1Image, masker: NiftiLabelsMasker, confound: ndarray | None
) -> ndarray: ...

class Parcellations:
    def __init__(
        self,
        method: str | None,
        n_parcels: int = ...,
        random_state: int = ...,
        mask: Nifti1Image | None = ...,
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
        verbose: bool | int = ...,
    ) -> None: ...
    def _check_fitted(self) -> None: ...
    def _raw_fit(self, data: ndarray) -> Parcellations: ...
    def fit_transform(
        self,
        imgs: list[Nifti1Image] | Nifti1Image,
        confounds: list[ndarray] | None = ...,
    ) -> ndarray | list[ndarray]: ...
    def inverse_transform(
        self, signals: ndarray | list[ndarray]
    ) -> list[Nifti1Image] | Nifti1Image: ...
    def transform(
        self,
        imgs: list[Nifti1Image] | Nifti1Image,
        confounds: list[ndarray] | None = ...,
    ) -> ndarray | list[ndarray]: ...

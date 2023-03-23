from typing import Optional, Tuple

from joblib.memory import Memory
from nibabel.nifti1 import Nifti1Image
from numpy import ndarray
from scipy.sparse._coo import coo_matrix
from scipy.sparse._csr import csr_matrix

def _compute_weights(X: ndarray, mask_img: Nifti1Image) -> ndarray: ...
def _make_3d_edges(vertices: ndarray, is_mask: bool) -> ndarray: ...
def _make_edges_and_weights(
    X: ndarray, mask_img: Nifti1Image
) -> tuple[ndarray, ndarray]: ...
def _nearest_neighbor_grouping(
    X: ndarray,
    connectivity: csr_matrix,
    n_clusters: int,
    threshold: float = ...,
) -> tuple[csr_matrix, ndarray, ndarray]: ...
def _nn_connectivity(
    connectivity: csr_matrix, threshold: float = ...
) -> coo_matrix: ...
def _reduce_data_and_connectivity(
    X: ndarray,
    labels: ndarray,
    n_components: int,
    connectivity: csr_matrix,
    threshold: float = ...,
) -> tuple[csr_matrix, ndarray]: ...
def _weighted_connectivity_graph(
    X: ndarray, mask_img: Nifti1Image
) -> csr_matrix: ...
def recursive_neighbor_agglomeration(
    X: ndarray,
    mask_img: Nifti1Image,
    n_clusters: int,
    n_iter: int = ...,
    threshold: float = ...,
    verbose: int = ...,
) -> tuple[int, ndarray]: ...

class ReNA:
    def __init__(
        self,
        mask_img: Nifti1Image,
        n_clusters: int = ...,
        scaling: bool = ...,
        n_iter: int = ...,
        threshold: float = ...,
        memory: Memory | None = ...,
        memory_level: int = ...,
        verbose: int = ...,
    ) -> None: ...
    def fit(self, X: ndarray, y: None = ...) -> ReNA: ...
    def inverse_transform(self, X_red: ndarray) -> ndarray: ...
    def transform(self, X: ndarray, y: None = ...) -> ndarray: ...

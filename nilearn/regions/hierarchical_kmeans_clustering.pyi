from numpy import ndarray
from typing import Optional


def _adjust_small_clusters(array: ndarray, n_clusters: int) -> ndarray: ...


def _remove_empty_labels(labels: ndarray) -> ndarray: ...


def hierarchical_k_means(
    X: ndarray,
    n_clusters: int,
    init: str = ...,
    batch_size: int = ...,
    n_init: int = ...,
    max_no_improvement: int = ...,
    verbose: int = ...,
    random_state: int = ...
) -> ndarray: ...


class HierarchicalKMeans:
    def __init__(
        self,
        n_clusters: int,
        init: str = ...,
        batch_size: int = ...,
        n_init: int = ...,
        max_no_improvement: int = ...,
        verbose: int = ...,
        random_state: int = ...,
        scaling: bool = ...
    ) -> None: ...
    def fit(self, X: ndarray, y: None = ...) -> HierarchicalKMeans: ...
    def inverse_transform(self, X_red: ndarray) -> ndarray: ...
    def transform(self, X: ndarray, y: None = ...) -> ndarray: ...

from typing import List, Optional, Union

from nibabel.nifti1 import Nifti1Image
from numpy import ndarray
from scipy.sparse._lil import lil_matrix
from sklearn.model_selection._split import KFold, LeaveOneGroupOut
from sklearn.svm._classes import LinearSVC

def _group_iter_search_light(
    list_rows: ndarray,
    estimator: LinearSVC,
    X: ndarray,
    y: ndarray | list[int],
    groups: ndarray | None,
    scoring: str | None,
    cv: LeaveOneGroupOut | KFold | None,
    thread_id: int,
    total: int,
    verbose: int = ...,
) -> ndarray: ...
def search_light(
    X: ndarray,
    y: ndarray | list[int],
    estimator: LinearSVC,
    A: lil_matrix,
    groups: ndarray | None = ...,
    scoring: str | None = ...,
    cv: LeaveOneGroupOut | KFold | None = ...,
    n_jobs: int = ...,
    verbose: int = ...,
) -> ndarray: ...

class GroupIterator:
    def __init__(self, n_features: int, n_jobs: int = ...) -> None: ...
    def __iter__(self) -> None: ...

class SearchLight:
    def __init__(
        self,
        mask_img: Nifti1Image,
        process_mask_img: Nifti1Image | None = ...,
        radius: int | float = ...,
        estimator: str = ...,
        n_jobs: int = ...,
        scoring: str | None = ...,
        cv: LeaveOneGroupOut | KFold | None = ...,
        verbose: int = ...,
    ) -> None: ...
    def fit(
        self,
        imgs: list[Nifti1Image] | Nifti1Image,
        y: ndarray | list[int],
        groups: ndarray | None = ...,
    ) -> SearchLight: ...

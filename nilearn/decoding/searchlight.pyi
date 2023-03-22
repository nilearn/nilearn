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
    y: Union[ndarray, List[int]],
    groups: Optional[ndarray],
    scoring: Optional[str],
    cv: Optional[Union[LeaveOneGroupOut, KFold]],
    thread_id: int,
    total: int,
    verbose: int = ...,
) -> ndarray: ...
def search_light(
    X: ndarray,
    y: Union[ndarray, List[int]],
    estimator: LinearSVC,
    A: lil_matrix,
    groups: Optional[ndarray] = ...,
    scoring: Optional[str] = ...,
    cv: Optional[Union[LeaveOneGroupOut, KFold]] = ...,
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
        process_mask_img: Optional[Nifti1Image] = ...,
        radius: Union[int, float] = ...,
        estimator: str = ...,
        n_jobs: int = ...,
        scoring: Optional[str] = ...,
        cv: Optional[Union[LeaveOneGroupOut, KFold]] = ...,
        verbose: int = ...,
    ) -> None: ...
    def fit(
        self,
        imgs: Union[List[Nifti1Image], Nifti1Image],
        y: Union[ndarray, List[int]],
        groups: Optional[ndarray] = ...,
    ) -> SearchLight: ...

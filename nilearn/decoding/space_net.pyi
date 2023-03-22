from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from joblib.memory import Memory
from nibabel.nifti1 import Nifti1Image
from numpy import float64, ndarray

def _crop_mask(mask: ndarray) -> ndarray: ...
def _space_net_alpha_grid(
    X: ndarray,
    y: ndarray,
    eps: float = ...,
    n_alphas: int = ...,
    l1_ratio: float = ...,
    logistic: bool = ...,
) -> ndarray: ...
def _univariate_feature_screening(
    X: ndarray,
    y: ndarray,
    mask: ndarray,
    is_classif: bool,
    screening_percentile: float,
    smoothing_fwhm: float = ...,
) -> Tuple[ndarray, ndarray, ndarray]: ...
def path_scores(
    solver: Union[Callable, partial],
    X: ndarray,
    y: ndarray,
    mask: ndarray,
    alphas: Optional[Union[List[float], List[float64]]],
    l1_ratios: Union[List[int], List[float], float, List[float64]],
    train: ndarray,
    test: ndarray,
    solver_params: Dict[str, Union[float, int]],
    is_classif: bool = ...,
    n_alphas: int = ...,
    eps: float = ...,
    key: Optional[Tuple[int, int]] = ...,
    debias: bool = ...,
    Xmean: None = ...,
    screening_percentile: float = ...,
    verbose: int = ...,
) -> Any: ...

class BaseSpaceNet:
    def __init__(
        self,
        penalty: str = ...,
        is_classif: bool = ...,
        loss: Optional[str] = ...,
        l1_ratios: Union[int, float] = ...,
        alphas: Optional[Union[List[float], float]] = ...,
        n_alphas: Union[int, float] = ...,
        mask: Optional[Union[str, Nifti1Image]] = ...,
        target_affine: None = ...,
        target_shape: None = ...,
        low_pass: None = ...,
        high_pass: None = ...,
        t_r: None = ...,
        max_iter: int = ...,
        tol: float = ...,
        memory: Optional[Memory] = ...,
        memory_level: int = ...,
        standardize: bool = ...,
        verbose: Union[int, bool] = ...,
        mask_args: None = ...,
        n_jobs: int = ...,
        eps: float = ...,
        cv: int = ...,
        fit_intercept: bool = ...,
        screening_percentile: Union[int, float] = ...,
        debias: bool = ...,
    ) -> None: ...
    def _set_coef_and_intercept(self, w: ndarray) -> None: ...
    def check_params(self) -> None: ...
    def decision_function(self, X: ndarray) -> ndarray: ...
    def fit(
        self, X: Nifti1Image, y: ndarray
    ) -> Union[BaseSpaceNet, SpaceNetRegressor, SpaceNetClassifier]: ...
    def predict(self, X: Nifti1Image) -> ndarray: ...

class SpaceNetClassifier:
    def __init__(
        self,
        penalty: str = ...,
        loss: str = ...,
        l1_ratios: float = ...,
        alphas: Optional[float] = ...,
        n_alphas: int = ...,
        mask: Optional[Union[str, Nifti1Image]] = ...,
        target_affine: None = ...,
        target_shape: None = ...,
        low_pass: None = ...,
        high_pass: None = ...,
        t_r: None = ...,
        max_iter: int = ...,
        tol: float = ...,
        memory: Memory = ...,
        memory_level: int = ...,
        standardize: bool = ...,
        verbose: Union[int, bool] = ...,
        n_jobs: int = ...,
        eps: float = ...,
        cv: int = ...,
        fit_intercept: bool = ...,
        screening_percentile: float = ...,
        debias: bool = ...,
    ) -> None: ...
    def _binarize_y(self, y: ndarray) -> ndarray: ...
    def score(self, X: Nifti1Image, y: ndarray) -> float64: ...

class SpaceNetRegressor:
    def __init__(
        self,
        penalty: str = ...,
        l1_ratios: float = ...,
        alphas: Optional[float] = ...,
        n_alphas: int = ...,
        mask: Optional[Union[str, Nifti1Image]] = ...,
        target_affine: None = ...,
        target_shape: None = ...,
        low_pass: None = ...,
        high_pass: None = ...,
        t_r: None = ...,
        max_iter: int = ...,
        tol: float = ...,
        memory: Memory = ...,
        memory_level: int = ...,
        standardize: bool = ...,
        verbose: Union[int, bool] = ...,
        n_jobs: int = ...,
        eps: float = ...,
        cv: int = ...,
        fit_intercept: bool = ...,
        screening_percentile: float = ...,
        debias: bool = ...,
    ) -> None: ...

class _EarlyStoppingCallback:
    def __call__(self, variables: Dict[str, Any]) -> Optional[bool]: ...
    def __init__(
        self,
        X_test: ndarray,
        y_test: ndarray,
        is_classif: bool,
        debias: bool = ...,
        verbose: int = ...,
    ) -> None: ...
    def _debias(self, w: ndarray) -> ndarray: ...
    def test_score(
        self, w: ndarray
    ) -> Union[
        Tuple[float64, float64], Tuple[float, float], Tuple[float64, float]
    ]: ...

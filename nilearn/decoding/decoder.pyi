from typing import Any

from nibabel.nifti1 import Nifti1Image
from nilearn.maskers.nifti_masker import NiftiMasker
from numpy import float64, int64, ndarray, str_
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble._forest import RandomForestClassifier
from sklearn.feature_selection._univariate_selection import SelectPercentile
from sklearn.metrics._scorer import _PredictScorer, _ThresholdScorer
from sklearn.model_selection._split import KFold, LeaveOneGroupOut
from sklearn.svm._classes import LinearSVC

def _check_estimator(
    estimator: (
        str | DummyRegressor | DummyClassifier | RandomForestClassifier
    ),
) -> BaseEstimator: ...
def _check_param_grid(
    estimator: Any,
    X: ndarray,
    y: ndarray,
    param_grid: dict[str, ndarray] | None = ...,
) -> dict[str, ndarray]: ...
def _parallel_fit(
    estimator: BaseEstimator,
    X: ndarray,
    y: ndarray,
    train: ndarray | range,
    test: ndarray | range,
    param_grid: dict[str, ndarray] | None,
    is_classification: bool,
    selector: SelectPercentile | None,
    scorer: _PredictScorer | _ThresholdScorer,
    mask_img: Nifti1Image | None,
    class_index: int,
    clustering_percentile: int,
) -> (
    tuple[int, ndarray, float64, dict[Any, Any], float64, None]
    | tuple[int, None, None, dict[Any, Any], float64, ndarray]
    | tuple[int, ndarray, ndarray, dict[str, float64], float64, None]
): ...

class Decoder:
    def __init__(
        self,
        estimator: str | DummyClassifier = ...,
        mask: Nifti1Image | NiftiMasker | None = ...,
        cv: LinearSVC | LeaveOneGroupOut | int | str | KFold = ...,
        param_grid: None = ...,
        screening_percentile: int | float | None = ...,
        scoring: str | _PredictScorer | None = ...,
        smoothing_fwhm: float | None = ...,
        standardize: bool = ...,
        target_affine: ndarray | None = ...,
        target_shape: tuple[int, int, int] | None = ...,
        mask_strategy: str = ...,
        low_pass: int | None = ...,
        high_pass: int | None = ...,
        t_r: int | None = ...,
        memory: None = ...,
        memory_level: int = ...,
        n_jobs: int = ...,
        verbose: int = ...,
    ) -> None: ...

class DecoderRegressor:
    def __init__(
        self,
        estimator: str | DummyRegressor = ...,
        mask: Nifti1Image | None = ...,
        cv: int = ...,
        param_grid: None = ...,
        screening_percentile: int | float | None = ...,
        scoring: str | None = ...,
        smoothing_fwhm: None = ...,
        standardize: bool = ...,
        target_affine: None = ...,
        target_shape: None = ...,
        mask_strategy: str = ...,
        low_pass: None = ...,
        high_pass: None = ...,
        t_r: None = ...,
        memory: None = ...,
        memory_level: int = ...,
        n_jobs: int = ...,
        verbose: int = ...,
    ) -> None: ...

class FREMClassifier:
    def __init__(
        self,
        estimator: str = ...,
        mask: Nifti1Image | NiftiMasker | None = ...,
        cv: int = ...,
        param_grid: None = ...,
        clustering_percentile: int = ...,
        screening_percentile: int | float = ...,
        scoring: str = ...,
        smoothing_fwhm: None = ...,
        standardize: bool = ...,
        target_affine: None = ...,
        target_shape: None = ...,
        mask_strategy: str = ...,
        low_pass: None = ...,
        high_pass: None = ...,
        t_r: None = ...,
        memory: None = ...,
        memory_level: int = ...,
        n_jobs: int = ...,
        verbose: int = ...,
    ) -> None: ...

class FREMRegressor:
    def __init__(
        self,
        estimator: str = ...,
        mask: Nifti1Image | None = ...,
        cv: int = ...,
        param_grid: None = ...,
        clustering_percentile: int = ...,
        screening_percentile: int | float = ...,
        scoring: str = ...,
        smoothing_fwhm: None = ...,
        standardize: bool = ...,
        target_affine: None = ...,
        target_shape: None = ...,
        mask_strategy: str = ...,
        low_pass: None = ...,
        high_pass: None = ...,
        t_r: None = ...,
        memory: None = ...,
        memory_level: int = ...,
        n_jobs: int = ...,
        verbose: int = ...,
    ) -> None: ...

class _BaseDecoder:
    def __init__(
        self,
        estimator: (
            str | DummyRegressor | DummyClassifier | RandomForestClassifier
        ) = ...,
        mask: Nifti1Image | NiftiMasker | None = ...,
        cv: Any = ...,
        param_grid: None = ...,
        clustering_percentile: int = ...,
        screening_percentile: int | float | None = ...,
        scoring: str | _PredictScorer | None = ...,
        smoothing_fwhm: float | None = ...,
        standardize: bool = ...,
        target_affine: ndarray | None = ...,
        target_shape: tuple[int, int, int] | None = ...,
        low_pass: int | None = ...,
        high_pass: int | None = ...,
        t_r: int | None = ...,
        mask_strategy: str = ...,
        is_classification: bool = ...,
        memory: None = ...,
        memory_level: int = ...,
        n_jobs: int = ...,
        verbose: int = ...,
    ) -> None: ...
    def _apply_mask(self, X: Nifti1Image) -> ndarray: ...
    def _binarize_y(self, y: ndarray) -> ndarray: ...
    def _fetch_parallel_fit_outputs(
        self,
        parallel_fit_outputs: list[
            (
                tuple[int, ndarray, ndarray, dict[str, float64], float64, None]
                | tuple[int, None, None, dict[Any, Any], float64, ndarray]
                | tuple[int, ndarray, float64, dict[Any, Any], float64, None]
            )
        ],
        y: ndarray,
        n_problems: int,
    ) -> Any: ...
    def _output_image(
        self,
        classes: ndarray | list[str],
        coefs: ndarray,
        std_coef: ndarray,
    ) -> (
        tuple[dict[int64, Nifti1Image], dict[int64, Nifti1Image]]
        | tuple[dict[str, Nifti1Image], dict[str, Nifti1Image]]
        | tuple[dict[str_, Nifti1Image], dict[str_, Nifti1Image]]
    ): ...
    def _predict_dummy(self, n_samples: int) -> ndarray: ...
    def _set_scorer(self) -> None: ...
    def decision_function(self, X: ndarray) -> ndarray: ...
    def fit(
        self,
        X: Nifti1Image,
        y: ndarray | list[str],
        groups: ndarray | None = ...,
    ) -> None: ...
    def predict(self, X: Nifti1Image) -> ndarray: ...
    def score(self, X: Nifti1Image, y: ndarray, *args) -> float64: ...

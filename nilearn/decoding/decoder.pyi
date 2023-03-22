from typing import Any, Dict, List, Optional, Tuple, Union

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
    estimator: Union[
        str, DummyRegressor, DummyClassifier, RandomForestClassifier
    ]
) -> BaseEstimator: ...
def _check_param_grid(
    estimator: Any,
    X: ndarray,
    y: ndarray,
    param_grid: Optional[Dict[str, ndarray]] = ...,
) -> Dict[str, ndarray]: ...
def _parallel_fit(
    estimator: BaseEstimator,
    X: ndarray,
    y: ndarray,
    train: Union[ndarray, range],
    test: Union[ndarray, range],
    param_grid: Optional[Dict[str, ndarray]],
    is_classification: bool,
    selector: Optional[SelectPercentile],
    scorer: Union[_PredictScorer, _ThresholdScorer],
    mask_img: Optional[Nifti1Image],
    class_index: int,
    clustering_percentile: int,
) -> Union[
    Tuple[int, ndarray, float64, Dict[Any, Any], float64, None],
    Tuple[int, None, None, Dict[Any, Any], float64, ndarray],
    Tuple[int, ndarray, ndarray, Dict[str, float64], float64, None],
]: ...

class Decoder:
    def __init__(
        self,
        estimator: Union[str, DummyClassifier] = ...,
        mask: Optional[Union[Nifti1Image, NiftiMasker]] = ...,
        cv: Union[LinearSVC, LeaveOneGroupOut, int, str, KFold] = ...,
        param_grid: None = ...,
        screening_percentile: Optional[Union[int, float]] = ...,
        scoring: Optional[Union[str, _PredictScorer]] = ...,
        smoothing_fwhm: Optional[float] = ...,
        standardize: bool = ...,
        target_affine: Optional[ndarray] = ...,
        target_shape: Optional[Tuple[int, int, int]] = ...,
        mask_strategy: str = ...,
        low_pass: Optional[int] = ...,
        high_pass: Optional[int] = ...,
        t_r: Optional[int] = ...,
        memory: None = ...,
        memory_level: int = ...,
        n_jobs: int = ...,
        verbose: int = ...,
    ) -> None: ...

class DecoderRegressor:
    def __init__(
        self,
        estimator: Union[str, DummyRegressor] = ...,
        mask: Optional[Nifti1Image] = ...,
        cv: int = ...,
        param_grid: None = ...,
        screening_percentile: Optional[Union[int, float]] = ...,
        scoring: Optional[str] = ...,
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
        mask: Optional[Union[Nifti1Image, NiftiMasker]] = ...,
        cv: int = ...,
        param_grid: None = ...,
        clustering_percentile: int = ...,
        screening_percentile: Union[int, float] = ...,
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
        mask: Optional[Nifti1Image] = ...,
        cv: int = ...,
        param_grid: None = ...,
        clustering_percentile: int = ...,
        screening_percentile: Union[int, float] = ...,
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
        estimator: Union[
            str, DummyRegressor, DummyClassifier, RandomForestClassifier
        ] = ...,
        mask: Optional[Union[Nifti1Image, NiftiMasker]] = ...,
        cv: Any = ...,
        param_grid: None = ...,
        clustering_percentile: int = ...,
        screening_percentile: Optional[Union[int, float]] = ...,
        scoring: Optional[Union[str, _PredictScorer]] = ...,
        smoothing_fwhm: Optional[float] = ...,
        standardize: bool = ...,
        target_affine: Optional[ndarray] = ...,
        target_shape: Optional[Tuple[int, int, int]] = ...,
        low_pass: Optional[int] = ...,
        high_pass: Optional[int] = ...,
        t_r: Optional[int] = ...,
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
        parallel_fit_outputs: List[
            Union[
                Tuple[
                    int, ndarray, ndarray, Dict[str, float64], float64, None
                ],
                Tuple[int, None, None, Dict[Any, Any], float64, ndarray],
                Tuple[int, ndarray, float64, Dict[Any, Any], float64, None],
            ]
        ],
        y: ndarray,
        n_problems: int,
    ) -> Any: ...
    def _output_image(
        self,
        classes: Union[ndarray, List[str]],
        coefs: ndarray,
        std_coef: ndarray,
    ) -> Union[
        Tuple[Dict[int64, Nifti1Image], Dict[int64, Nifti1Image]],
        Tuple[Dict[str, Nifti1Image], Dict[str, Nifti1Image]],
        Tuple[Dict[str_, Nifti1Image], Dict[str_, Nifti1Image]],
    ]: ...
    def _predict_dummy(self, n_samples: int) -> ndarray: ...
    def _set_scorer(self) -> None: ...
    def decision_function(self, X: ndarray) -> ndarray: ...
    def fit(
        self,
        X: Nifti1Image,
        y: Union[ndarray, List[str]],
        groups: Optional[ndarray] = ...,
    ) -> None: ...
    def predict(self, X: Nifti1Image) -> ndarray: ...
    def score(self, X: Nifti1Image, y: ndarray, *args) -> float64: ...

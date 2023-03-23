from typing import Dict, Optional, Tuple, Union

from numpy import float64, ndarray

def _dual_gap_prox_tvl1(
    input_img_norm: float64,
    new: ndarray,
    gap: ndarray,
    weight: float,
    l1_ratio: float = ...,
) -> float64: ...
def _projector_on_tvl1_dual(grad: ndarray, l1_ratio: float) -> ndarray: ...
def _prox_l1(
    y: ndarray, alpha: float64 | float, copy: bool = ...
) -> ndarray: ...
def _prox_l1_with_intercept(x: ndarray, tau: float64) -> ndarray: ...
def _prox_tvl1(
    input_img: ndarray,
    l1_ratio: float = ...,
    weight: float64 = ...,
    dgap_tol: float64 | float = ...,
    x_tol: None = ...,
    max_iter: int = ...,
    check_gap_frequency: int = ...,
    val_min: None = ...,
    val_max: None = ...,
    verbose: bool | int = ...,
    fista: bool = ...,
    init: ndarray | None = ...,
) -> tuple[ndarray, dict[str, bool]]: ...
def _prox_tvl1_with_intercept(
    w: ndarray,
    shape: tuple[int, int, int] | tuple[int],
    l1_ratio: float,
    weight: float64,
    dgap_tol: float64 | float,
    max_iter: int = ...,
    init: ndarray | None = ...,
    verbose: int = ...,
) -> tuple[ndarray, dict[str, bool]]: ...

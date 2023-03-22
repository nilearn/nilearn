from typing import Dict, List, Optional, Tuple, Union

from nilearn.decoding.space_net import _EarlyStoppingCallback
from numpy import float64, ndarray

def _graph_net_adjoint_data_function(
    X: ndarray, w: ndarray, adjoint_mask: ndarray, grad_weight: float
) -> ndarray: ...
def _graph_net_data_function(
    X: ndarray, w: ndarray, mask: ndarray, grad_weight: float
) -> ndarray: ...
def _graph_net_logistic(
    X: ndarray,
    y: ndarray,
    alpha: Union[float64, float],
    l1_ratio: float,
    mask: ndarray,
    init: Optional[Dict[str, Union[ndarray, float, float64]]] = ...,
    max_iter: int = ...,
    tol: float = ...,
    callback: Optional[_EarlyStoppingCallback] = ...,
    verbose: int = ...,
    lipschitz_constant: None = ...,
) -> Tuple[
    ndarray, List[float64], Dict[str, Union[ndarray, float, float64]]
]: ...
def _graph_net_squared_loss(
    X: ndarray,
    y: ndarray,
    alpha: Union[float64, float],
    l1_ratio: Union[int, float64, float],
    mask: ndarray,
    init: Optional[Dict[str, Union[ndarray, float, float64]]] = ...,
    max_iter: int = ...,
    tol: float = ...,
    callback: Optional[_EarlyStoppingCallback] = ...,
    lipschitz_constant: None = ...,
    verbose: int = ...,
) -> Tuple[
    ndarray, List[float64], Dict[str, Union[ndarray, float, float64]]
]: ...
def _logistic_data_loss_and_spatial_grad(
    X: ndarray,
    y: ndarray,
    w: ndarray,
    mask: ndarray,
    grad_weight: Union[int, float64, float],
) -> float64: ...
def _logistic_data_loss_and_spatial_grad_derivative(
    X: ndarray,
    y: ndarray,
    w: ndarray,
    mask: ndarray,
    grad_weight: Union[int, float64, float],
) -> ndarray: ...
def _logistic_derivative_lipschitz_constant(
    X: ndarray,
    mask: ndarray,
    grad_weight: Union[float64, float],
    n_iterations: int = ...,
) -> float64: ...
def _squared_loss_and_spatial_grad(
    X: ndarray,
    y: ndarray,
    w: ndarray,
    mask: ndarray,
    grad_weight: Union[float64, float, int],
) -> float64: ...
def _squared_loss_and_spatial_grad_derivative(
    X: ndarray,
    y: ndarray,
    w: ndarray,
    mask: ndarray,
    grad_weight: Union[float64, float, int],
) -> ndarray: ...
def _squared_loss_derivative_lipschitz_constant(
    X: ndarray,
    mask: ndarray,
    grad_weight: Union[float64, float, int],
    n_iterations: int = ...,
) -> float64: ...
def _tvl1_objective(
    X: Optional[ndarray],
    y: Optional[ndarray],
    w: Optional[ndarray],
    alpha: Optional[float],
    l1_ratio: Optional[float],
    mask: Optional[ndarray],
    loss: str = ...,
) -> float64: ...
def _tvl1_objective_from_gradient(gradient: ndarray) -> float64: ...
def tvl1_solver(
    X: ndarray,
    y: Optional[ndarray],
    alpha: Optional[float],
    l1_ratio: Optional[float],
    mask: Optional[ndarray],
    loss: Optional[str] = ...,
    max_iter: int = ...,
    lipschitz_constant: None = ...,
    init: None = ...,
    prox_max_iter: int = ...,
    tol: float = ...,
    callback: None = ...,
    verbose: int = ...,
) -> Tuple[
    ndarray, List[float64], Dict[str, Union[ndarray, float, float64]]
]: ...

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
    alpha: float64 | float,
    l1_ratio: float,
    mask: ndarray,
    init: dict[str, ndarray | float | float64] | None = ...,
    max_iter: int = ...,
    tol: float = ...,
    callback: _EarlyStoppingCallback | None = ...,
    verbose: int = ...,
    lipschitz_constant: None = ...,
) -> tuple[ndarray, list[float64], dict[str, ndarray | float | float64]]: ...
def _graph_net_squared_loss(
    X: ndarray,
    y: ndarray,
    alpha: float64 | float,
    l1_ratio: int | float64 | float,
    mask: ndarray,
    init: dict[str, ndarray | float | float64] | None = ...,
    max_iter: int = ...,
    tol: float = ...,
    callback: _EarlyStoppingCallback | None = ...,
    lipschitz_constant: None = ...,
    verbose: int = ...,
) -> tuple[ndarray, list[float64], dict[str, ndarray | float | float64]]: ...
def _logistic_data_loss_and_spatial_grad(
    X: ndarray,
    y: ndarray,
    w: ndarray,
    mask: ndarray,
    grad_weight: int | float64 | float,
) -> float64: ...
def _logistic_data_loss_and_spatial_grad_derivative(
    X: ndarray,
    y: ndarray,
    w: ndarray,
    mask: ndarray,
    grad_weight: int | float64 | float,
) -> ndarray: ...
def _logistic_derivative_lipschitz_constant(
    X: ndarray,
    mask: ndarray,
    grad_weight: float64 | float,
    n_iterations: int = ...,
) -> float64: ...
def _squared_loss_and_spatial_grad(
    X: ndarray,
    y: ndarray,
    w: ndarray,
    mask: ndarray,
    grad_weight: float64 | float | int,
) -> float64: ...
def _squared_loss_and_spatial_grad_derivative(
    X: ndarray,
    y: ndarray,
    w: ndarray,
    mask: ndarray,
    grad_weight: float64 | float | int,
) -> ndarray: ...
def _squared_loss_derivative_lipschitz_constant(
    X: ndarray,
    mask: ndarray,
    grad_weight: float64 | float | int,
    n_iterations: int = ...,
) -> float64: ...
def _tvl1_objective(
    X: ndarray | None,
    y: ndarray | None,
    w: ndarray | None,
    alpha: float | None,
    l1_ratio: float | None,
    mask: ndarray | None,
    loss: str = ...,
) -> float64: ...
def _tvl1_objective_from_gradient(gradient: ndarray) -> float64: ...
def tvl1_solver(
    X: ndarray,
    y: ndarray | None,
    alpha: float | None,
    l1_ratio: float | None,
    mask: ndarray | None,
    loss: str | None = ...,
    max_iter: int = ...,
    lipschitz_constant: None = ...,
    init: None = ...,
    prox_max_iter: int = ...,
    tol: float = ...,
    callback: None = ...,
    verbose: int = ...,
) -> tuple[ndarray, list[float64], dict[str, ndarray | float | float64]]: ...

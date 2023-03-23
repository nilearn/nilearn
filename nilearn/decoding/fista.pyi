from collections.abc import Callable

from nilearn.decoding.space_net import _EarlyStoppingCallback
from numpy import float64, ndarray

def _check_lipschitz_continuous(
    f: Callable,
    ndim: int,
    lipschitz_constant: float64,
    n_trials: int = ...,
    random_state: int = ...,
) -> None: ...
def mfista(
    f1_grad: Callable,
    f2_prox: Callable,
    total_energy: Callable,
    lipschitz_constant: float64 | float,
    w_size: int,
    dgap_tol: None = ...,
    init: dict[str, ndarray | float | float64] | None = ...,
    max_iter: int = ...,
    tol: float = ...,
    check_lipschitz: bool = ...,
    dgap_factor: float64 | float | None = ...,
    callback: _EarlyStoppingCallback | Callable | None = ...,
    verbose: int = ...,
) -> (
    tuple[ndarray, list[float64], dict[str, ndarray | float]]
    | tuple[ndarray, list[float64], dict[str, ndarray | float | float64]]
): ...

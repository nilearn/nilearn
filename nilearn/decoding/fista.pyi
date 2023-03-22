from typing import Callable, Dict, List, Optional, Tuple, Union

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
    lipschitz_constant: Union[float64, float],
    w_size: int,
    dgap_tol: None = ...,
    init: Optional[Dict[str, Union[ndarray, float, float64]]] = ...,
    max_iter: int = ...,
    tol: float = ...,
    check_lipschitz: bool = ...,
    dgap_factor: Optional[Union[float64, float]] = ...,
    callback: Optional[Union[_EarlyStoppingCallback, Callable]] = ...,
    verbose: int = ...,
) -> Union[
    Tuple[ndarray, List[float64], Dict[str, Union[ndarray, float]]],
    Tuple[ndarray, List[float64], Dict[str, Union[ndarray, float, float64]]],
]: ...

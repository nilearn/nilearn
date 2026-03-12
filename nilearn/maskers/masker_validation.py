"""Utilities for masker validation."""

import warnings
from string import Template
from typing import Any, Literal

import numpy as np

from nilearn._base import NilearnBaseEstimator
from nilearn._utils.cache_mixin import check_memory
from nilearn._utils.logger import find_stack_level
from nilearn._utils.tags import is_glm
from nilearn.maskers import (
    MultiNiftiMasker,
    MultiSurfaceMasker,
    NiftiMasker,
    SurfaceMasker,
)
from nilearn.maskers._mixin import _MultiMixin


def get_params(
    cls: type[
        NiftiMasker | SurfaceMasker | MultiNiftiMasker | MultiSurfaceMasker
    ],
    instance: NilearnBaseEstimator,
    ignore: None | list[str] = None,
) -> dict[str, Any]:
    """Retrieve the initialization parameters corresponding to a class.

    This helper function retrieves the parameters of function __init__ for
    class 'cls' and returns the value for these parameters in object
    'instance'.
    When using a composition pattern (e.g. with a NiftiMasker class),
    it is useful to forward parameters from one instance to another.

    Parameters
    ----------
    cls : class
        The class that gives us the list of parameters we are interested in.

    instance : object, instance of NilearnBaseEstimator
        The object that gives us the values of the parameters.

    ignore : None or list of strings
        Names of the parameters that are not returned.

    Returns
    -------
    params : dict
        The dict of parameters.

    """
    _ignore = {"memory", "memory_level", "verbose", "copy", "n_jobs"}
    if ignore is not None:
        _ignore.update(ignore)

    param_names = cls._get_param_names()

    params = {}
    for param_name in param_names:
        if param_name in _ignore:
            continue
        if hasattr(instance, param_name):
            params[param_name] = getattr(instance, param_name)

    return params


def check_embedded_masker(
    estimator,
    masker_type: Literal["nii", "surface", "multi_nii", "multi_surface"],
    ignore: list[str] | None = None,
) -> NiftiMasker | SurfaceMasker | MultiNiftiMasker | MultiSurfaceMasker:
    """Create a masker from instance parameters.

    Base function for using a masker within a NilearnBaseEstimator class

    This creates a masker from instance parameters :

    - If instance contains a mask image in mask parameter,
    we use this image as new masker mask_img, forwarding instance parameters to
    new masker : smoothing_fwhm, standardize, detrend, low_pass, high_pass,
    t_r, target_affine, target_shape, mask_strategy, mask_args...

    - If instance contains a masker in mask parameter, we use a copy of
    this masker, overriding all instance masker related parameters.
    In all case, we forward system parameters of instance to new masker :
    memory, memory_level, verbose, n_jobs

    Parameters
    ----------
    instance : object, instance of NilearnBaseEstimator
        The object that gives us the values of the parameters

    masker_type : {"multi_nii", "nii", "surface", "multi_surface"}
        Indicates whether to return a MultiNiftiMasker, NiftiMasker,
        SurfaceMasker, or a MultiSurfaceMasker.

    ignore : None or :obj:`list` of :obj:`str`
        Names of the parameters of the estimator that should not be
        transferred to the new masker.

    Returns
    -------
    masker : MultiNiftiMasker, NiftiMasker, SurfaceMasker, MultiSurfaceMasker
        New masker

    """
    masker: type[
        NiftiMasker | SurfaceMasker | MultiNiftiMasker | MultiSurfaceMasker
    ] = NiftiMasker
    if masker_type == "surface":
        masker = SurfaceMasker
    elif masker_type == "multi_surface":
        masker = MultiSurfaceMasker
    elif masker_type == "multi_nii":
        masker = MultiNiftiMasker

    estimator_params = get_params(masker, estimator, ignore=ignore)

    mask = getattr(estimator, "mask", None)
    if is_glm(estimator):
        mask = getattr(estimator, "mask_img", None)

    if isinstance(mask, (NiftiMasker, SurfaceMasker)):
        # Creating masker from provided masker
        masker_params = get_params(masker, mask)
        new_masker_params = masker_params
    else:
        # Creating a masker with parameters extracted from estimator
        new_masker_params = estimator_params
        new_masker_params["mask_img"] = mask

    # Forwarding system parameters of instance to new masker in all case
    if issubclass(masker, (_MultiMixin)) and hasattr(estimator, "n_jobs"):
        # For MultiMaskers only
        new_masker_params["n_jobs"] = estimator.n_jobs

    warning_msg = Template(
        "Provided estimator has no '$attribute' attribute set. "
        "Setting '$attribute' to '$default_value' by default."
    )

    if hasattr(estimator, "memory"):
        new_masker_params["memory"] = check_memory(estimator.memory)
    else:
        warnings.warn(
            warning_msg.substitute(
                attribute="memory",
                default_value="Memory(location=None)",
            ),
            stacklevel=find_stack_level(),
        )
        new_masker_params["memory"] = check_memory(None)

    if hasattr(estimator, "memory_level"):
        new_masker_params["memory_level"] = max(0, estimator.memory_level - 1)
    else:
        warnings.warn(
            warning_msg.substitute(
                attribute="memory_level", default_value="0"
            ),
            stacklevel=find_stack_level(),
        )
        new_masker_params["memory_level"] = 0

    if hasattr(estimator, "verbose"):
        new_masker_params["verbose"] = estimator.verbose
    else:
        warnings.warn(
            warning_msg.substitute(attribute="verbose", default_value="0"),
            stacklevel=find_stack_level(),
        )
        new_masker_params["verbose"] = 0

    conflicting_param = [
        k
        for k in sorted(estimator_params)
        if np.any(new_masker_params[k] != estimator_params[k])
    ]
    if conflicting_param:
        conflict_string = "".join(
            (
                f"Parameter {k} :\n"
                f"    Masker parameter {new_masker_params[k]}"
                f" - overriding estimator parameter {estimator_params[k]}\n"
            )
            for k in conflicting_param
        )
        warn_str = (
            "Overriding provided-default estimator parameters with"
            f" provided masker parameters :\n{conflict_string}"
        )
        warnings.warn(warn_str, stacklevel=find_stack_level())

    masker_instance = masker(**new_masker_params)

    # Forwarding potential attribute of provided masker
    if mask is not None and hasattr(mask, "mask_img_"):
        # Allow free fit of returned mask
        masker_instance.mask_img = mask.mask_img_

    # TODO (nilearn >= 0.15.0) remove if and elif
    # avoid some FutureWarning the user cannot affect
    if masker_instance.standardize is False:
        masker_instance.standardize = None
    elif masker_instance.standardize is True:
        masker_instance.standardize = "zscore_sample"

    return masker_instance

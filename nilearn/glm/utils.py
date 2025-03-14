"""Utilities for GLM."""

from collections import OrderedDict

import numpy as np

from nilearn.glm.first_level import FirstLevelModel
from nilearn.maskers import SurfaceMasker
from nilearn.surface import SurfaceImage


def is_volume_glm(model):
    """Return if mpdel is run on volume data or not."""
    return not isinstance(model.mask_img, (SurfaceMasker, SurfaceImage)) or (
        hasattr(model, "masker_") and isinstance(model.masker_, SurfaceMasker)
    )


def return_model_type(model):
    """Return model type as string."""
    return (
        "First Level Model"
        if isinstance(model, FirstLevelModel)
        else "Second Level Model"
    )


def glm_model_attributes_to_dict(model):
    """Return fict with pertinent model attributes & information.

    Parameters
    ----------
    model : FirstLevelModel or SecondLevelModel object.

    Returns
    -------
    dict
    """
    selected_attributes = [
        "subject_label",
        "drift_model",
        "hrf_model",
        "standardize",
        "noise_model",
        "t_r",
        "signal_scaling",
        "scaling_axis",
        "smoothing_fwhm",
        "slice_time_ref",
    ]
    if is_volume_glm(model):
        selected_attributes.extend(["target_shape", "target_affine"])
    if hasattr(model, "hrf_model") and model.hrf_model == "fir":
        selected_attributes.append("fir_delays")
    if hasattr(model, "drift_model"):
        if model.drift_model == "cosine":
            selected_attributes.append("high_pass")
        elif model.drift_model == "polynomial":
            selected_attributes.append("drift_order")

    selected_attributes.sort()

    model_param = OrderedDict(
        (attr_name, getattr(model, attr_name))
        for attr_name in selected_attributes
        if getattr(model, attr_name, None) is not None
    )

    for k, v in model_param.items():
        if isinstance(v, np.ndarray):
            model_param[k] = v.tolist()

    return model_param

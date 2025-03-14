"""Utilities for GLM."""

from nilearn.glm.first_level import FirstLevelModel


def return_model_type(model):
    """Return model type as string."""
    return (
        "First Level Model"
        if isinstance(model, FirstLevelModel)
        else "Second Level Model"
    )

"""Functions for surface manipulation."""

# TODO the following are not mentioned in the API part of the doc
# "load_surface",
# "check_surface",
# "check_mesh_and_data",
# "Mesh",
# "Surface",

from .surface import (
    Mesh,
    Surface,
    check_mesh_and_data,
    check_surface,
    load_surf_data,
    load_surf_mesh,
    load_surface,
    vol_to_surf,
)

__all__ = [
    "Mesh",
    "Surface",
    "check_surface",
    "check_mesh_and_data",
    "load_surf_data",
    "load_surf_mesh",
    "load_surface",
    "vol_to_surf",
]

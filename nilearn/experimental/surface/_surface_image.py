"""Surface API."""

from __future__ import annotations

import abc
import dataclasses
import pathlib
from typing import Dict

import numpy as np

from nilearn.experimental.surface import _io

PolyData = Dict[str, np.ndarray]


class Mesh(abc.ABC):
    """A surface :term:`mesh` having vertex, \
    coordinates and faces (triangles)."""

    n_vertices: int

    # TODO those are properties for now for compatibility with plot_surf_img
    # for the demo.
    # But they should probably become functions as they can take some time to
    # return or even fail
    coordinates: np.ndarray
    faces: np.ndarray

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"with {getattr(self, 'n_vertices', '??')} vertices>"
        )

    def to_gifti(self, gifti_file: pathlib.Path | str):
        """Write surface mesh to a Gifti file on disk.

        Parameters
        ----------
        gifti_file : path-like or str
            filename to save the mesh.
        """
        _io.mesh_to_gifti(self.coordinates, self.faces, gifti_file)


class InMemoryMesh(Mesh):
    """A surface mesh stored as in-memory numpy arrays."""

    n_vertices: int

    coordinates: np.ndarray
    faces: np.ndarray

    def __init__(self, coordinates: np.ndarray, faces: np.ndarray) -> None:
        self.coordinates = coordinates
        self.faces = faces
        self.n_vertices = coordinates.shape[0]


class FileMesh(Mesh):
    """A surface mesh stored in a Gifti or Freesurfer file."""

    n_vertices: int

    file_path: pathlib.Path

    def __init__(self, file_path: pathlib.Path | str) -> None:
        self.file_path = pathlib.Path(file_path)
        self.n_vertices = _io.read_mesh(self.file_path)["coordinates"].shape[0]

    @property
    def coordinates(self) -> np.ndarray:
        """Get x, y, z, values for each mesh vertex."""
        return _io.read_mesh(self.file_path)["coordinates"]

    @property
    def faces(self) -> np.ndarray:
        """Get array of adjacent vertices."""
        return _io.read_mesh(self.file_path)["faces"]

    def loaded(self) -> InMemoryMesh:
        """Load surface mesh into memory."""
        loaded_arrays = _io.read_mesh(self.file_path)
        return InMemoryMesh(
            loaded_arrays["coordinates"], loaded_arrays["faces"]
        )


PolyMesh = Dict[str, Mesh]


def _check_data_consistent_shape(data: PolyData):
    """Check that shapes of PolyData parts match.

    They must match in all but the last dimension (which is the number of
    vertices, and can be different for each part).

    """
    if len(data) == 0:
        raise ValueError("Surface image data must have at least one item.")
    first_name = next(iter(data.keys()))
    first_shape = data[first_name].shape
    for part_name, part_data in data.items():
        if part_data.shape[:-1] != first_shape[:-1]:
            raise ValueError(
                f"Data arrays for keys '{first_name}' and '{part_name}' "
                "have incompatible shapes: "
                f"{first_shape} and {part_data.shape}"
            )


def _check_data_and_mesh_compat(mesh: PolyMesh, data: PolyData):
    """Check that mesh and data have the same keys and that shapes match."""
    data_keys, mesh_keys = set(data.keys()), set(mesh.keys())
    if data_keys != mesh_keys:
        diff = data_keys.symmetric_difference(mesh_keys)
        raise ValueError(
            "Data and mesh do not have the same keys. "
            f"Offending keys: {diff}"
        )
    for key in mesh_keys:
        if data[key].shape[-1] != mesh[key].n_vertices:
            raise ValueError(
                "Data shape does not match number of vertices"
                f" for '{key}':"
                f"\ndata shape: {data[key].shape}",
                f"\nn vertices: {mesh[key].n_vertices}",
            )


@dataclasses.dataclass
class SurfaceImage:
    """Surface image, usually containing meshes & data for both hemispheres."""

    mesh: PolyMesh
    data: PolyData
    shape: tuple[int, ...] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        _check_data_consistent_shape(self.data)
        _check_data_and_mesh_compat(self.mesh, self.data)
        total_n_vertices = sum(
            mesh_part.n_vertices for mesh_part in self.mesh.values()
        )
        first_data_shape = list(self.data.values())[0].shape
        self.shape = (*first_data_shape[:-1], total_n_vertices)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {getattr(self, 'shape', '')}>"

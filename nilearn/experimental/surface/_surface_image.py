from __future__ import annotations

import dataclasses
import pathlib
from typing import Dict

import numpy as np

from nilearn.experimental.surface import _io

PolyData = Dict[str, np.ndarray]


class Mesh:
    """A surface mesh stored in a Gifti or Freesurfer file."""

    n_vertices: int

    _file: pathlib.Path

    def __init__(self, file_path: pathlib.Path | str) -> None:
        self._file = pathlib.Path(file_path)
        self.n_vertices = _io.read_mesh(self._file)["coordinates"].shape[0]

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} with {self.n_vertices} vertices>"

    # TODO those are properties for now for compatibility with plot_surf_img
    # for the demo but should become functions as they can take some time to
    # return
    @property
    def coordinates(self) -> np.ndarray:
        return _io.read_mesh(self._file)["coordinates"]

    @property
    def faces(self) -> np.ndarray:
        return _io.read_mesh(self._file)["faces"]


class InMemoryMesh(Mesh):
    """A surface mesh stored as in-memory numpy arrays."""

    n_vertices: int

    _coordinates: np.ndarray
    _faces: np.ndarray

    def __init__(self, coordinates: np.ndarray, faces: np.ndarray) -> None:
        self._coordinates = coordinates
        self._faces = faces
        self.n_vertices = coordinates.shape[0]

    @property
    def coordinates(self) -> np.ndarray:
        return self._coordinates

    @property
    def faces(self) -> np.ndarray:
        return self._faces


PolyMesh = Dict[str, Mesh]


def _get_vertex_counts(mesh: PolyMesh) -> dict[str, int]:
    """Get the total number of vertices across all mesh parts (hemispheres)."""
    return {part_name: part.n_vertices for (part_name, part) in mesh.items()}


@dataclasses.dataclass
class SurfaceImage:
    """Surface image, usually containing meshes & data for both hemispheres."""

    data: PolyData
    mesh: PolyMesh
    shape: tuple[int, ...] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        data_shapes = {k: v.shape for (k, v) in self.data.items()}
        vertex_counts = _get_vertex_counts(self.mesh)
        assert {k: v[-1] for (k, v) in data_shapes.items()} == vertex_counts
        total_n_vertices = sum(vertex_counts.values())
        first_data_shape = list(data_shapes.values())[0]
        if len(first_data_shape) == 1:
            self.shape = (total_n_vertices,)
        else:
            assert len(first_data_shape) == 2
            self.shape = (first_data_shape[0], total_n_vertices)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.shape}>"

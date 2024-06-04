"""Surface API."""

from __future__ import annotations

import abc
import pathlib

import numpy as np

from nilearn.experimental.surface import _io


class PolyData:
    """A collection of data arrays.

    It is a shallow wrapper around the ``parts`` dictionary, which cannot be
    empty and whose keys must be a subset of {"left", "right"}.
    """

    def __init__(
        self, left: np.ndarray | None = None, right: np.ndarray | None = None
    ) -> None:
        if left is None and right is None:
            raise ValueError(
                "Cannot create an empty PolyData. "
                "Either left or right (or both) must be provided."
            )
        parts = {}
        if left is not None:
            parts["left"] = left
        if right is not None:
            parts["right"] = right
        if len(parts) == 1:
            self.parts = parts
            self.shape = next(iter(self.parts.values())).shape
            return
        if parts["left"].shape[:-1] != parts["right"].shape[:-1]:
            raise ValueError(
                f"Data arrays for keys 'left' and 'right' "
                "have incompatible shapes: "
                f"{parts['left'].shape} and {parts['right'].shape}"
            )
        self.parts = parts
        first_shape = next(iter(parts.values())).shape
        concat_dim = sum(p.shape[-1] for p in parts.values())
        self.shape = (*first_shape[:-1], concat_dim)


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


class PolyMesh:
    """A collection of meshes.

    It is a shallow wrapper around the ``parts`` dictionary, which cannot be
    empty and whose keys must be a subset of {"left", "right"}.
    """

    n_vertices: int

    def __init__(
        self, left: Mesh | None = None, right: Mesh | None = None
    ) -> None:
        if left is None and right is None:
            raise ValueError(
                "Cannot create an empty PolyMesh. "
                "Either left or right (or both) must be provided."
            )
        self.parts = {}
        if left is not None:
            self.parts["left"] = left
        if right is not None:
            self.parts["right"] = right

        self.n_vertices = sum(p.n_vertices for p in self.parts.values())


def _check_data_and_mesh_compat(mesh: PolyMesh, data: PolyData):
    """Check that mesh and data have the same keys and that shapes match."""
    data_keys, mesh_keys = set(data.parts.keys()), set(mesh.parts.keys())
    if data_keys != mesh_keys:
        diff = data_keys.symmetric_difference(mesh_keys)
        raise ValueError(
            "Data and mesh do not have the same keys. "
            f"Offending keys: {diff}"
        )
    for key in mesh_keys:
        if data.parts[key].shape[-1] != mesh.parts[key].n_vertices:
            raise ValueError(
                "Data shape does not match number of vertices"
                f" for '{key}':"
                f"\ndata shape: {data.parts[key].shape}",
                f"\nn vertices: {mesh.parts[key].n_vertices}",
            )


class SurfaceImage:
    """Surface image, usually containing meshes & data for both hemispheres."""

    def __init__(
        self,
        mesh: PolyMesh | dict[str, Mesh],
        data: PolyData | dict[str, Mesh],
    ) -> None:
        if isinstance(mesh, PolyMesh):
            self.mesh = mesh
        else:
            self.mesh = PolyMesh(**mesh)
        if isinstance(data, PolyData):
            self.data = data
        else:
            self.data = PolyData(**data)
        _check_data_and_mesh_compat(self.mesh, self.data)
        self.shape = self.data.shape

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {getattr(self, 'shape', '')}>"

"""Surface API."""
from __future__ import annotations

import abc
import dataclasses
import pathlib

from nilearn.surface import _io


class _Mesh(abc.ABC):
    """A surface :term:`mesh` having vertex, \
    coordinates and :term:`faces` (triangles)."""

    n_vertices: ...

    # TODO those are properties for now for compatibility with plot_surf_img
    # for the demo.
    # But they should probably become functions as they can take some time to
    # return or even fail
    def __repr__(self):
        return (
            f"<{self.__class__.__name__} "
            f"with {getattr(self, 'n_vertices', '??')} vertices>"
        )

    def to_gifti(self, gifti_file):
        """Write surface :term:`mesh` to a Gifti file on disk.

        Parameters
        ----------
        gifti_file : path-like or :obj:`str`
            filename to save the mesh.
        """
        _io.mesh_to_gifti(self.coordinates, self.faces, gifti_file)


class InMemoryMesh(_Mesh):
    """A surface :term:`mesh` stored as in-memory numpy arrays.

    Parameters
    ----------
    coordinates : :obj:`numpy.ndarray`
        3d coordinates of the vertices with shape (n_vertices, 3).

    faces : :obj:`numpy.ndarray`
        Each row represents 3 vertices that form a triangle in the mesh.

    Attributes
    ----------
    n_vertices : :obj:`int`
        Number of vertices in a mesh.
    """

    def __init__(self, coordinates, faces):
        self.coordinates = coordinates
        self.faces = faces
        self.n_vertices = coordinates.shape[0]


class FileMesh(_Mesh):
    """A surface :term:`mesh` stored in a Gifti or Freesurfer file.

    Parameters
    ----------
    file_path : path-like or :obj:`str`

    Attributes
    ----------
    n_vertices : :obj:`int`
        Number of vertices in a mesh.
    """

    def __init__(self, file_path):
        self.file_path = pathlib.Path(file_path)
        self.n_vertices = _io.read_mesh(self.file_path)["coordinates"].shape[0]

    @property
    def coordinates(self):
        """Get x, y, z, values for each mesh vertex."""
        return _io.read_mesh(self.file_path)["coordinates"]

    @property
    def faces(self):
        """Get array of adjacent vertices."""
        return _io.read_mesh(self.file_path)["faces"]

    def loaded(self):
        """Load surface mesh into memory."""
        loaded_arrays = _io.read_mesh(self.file_path)
        return InMemoryMesh(
            loaded_arrays["coordinates"], loaded_arrays["faces"]
        )


def _check_data_consistent_shape(data):
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


def _check_data_and_mesh_compat(mesh, data):
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
    """Surface image, usually containing meshes & data for both hemispheres.

    Parameters
    ----------
    mesh : :obj:`dict`[:obj:`str`, mesh object]
        A dictionary relating hemispheres represented by string keys to
        their geometry represented by a mesh.

    data : :obj:`dict`[:obj:`str`, :class:`numpy.ndarray`]
        A dictionary relating hemispheres represented by string keys to
        their surface data.

    Attributes
    ----------
    shape : :obj:`tuple`
    """

    mesh: ...
    data: ...
    shape: ... = dataclasses.field(init=False)

    def __post_init__(self):
        _check_data_consistent_shape(self.data)
        _check_data_and_mesh_compat(self.mesh, self.data)
        total_n_vertices = sum(
            mesh_part.n_vertices for mesh_part in self.mesh.values()
        )
        first_data_shape = list(self.data.values())[0].shape
        self.shape = (*first_data_shape[:-1], total_n_vertices)

    def __repr__(self):
        return f"<{self.__class__.__name__} {getattr(self, 'shape', '')}>"

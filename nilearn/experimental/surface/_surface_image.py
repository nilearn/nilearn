"""Surface API."""

from __future__ import annotations

import abc
import pathlib
import sys
from pathlib import Path

import numpy as np
from nibabel import Nifti1Image

from nilearn._utils.niimg_conversions import check_niimg
from nilearn.experimental.surface import _io
from nilearn.surface import vol_to_surf


class PolyData:
    """A collection of data arrays.

    It is a shallow wrapper around the ``parts`` dictionary, which cannot be
    empty and whose keys must be a subset of {"left", "right"}.
    """

    def __init__(
        self,
        left: np.ndarray | str | Path | None = None,
        right: np.ndarray | str | Path | None = None,
    ) -> None:
        if left is None and right is None:
            raise ValueError(
                "Cannot create an empty PolyData. "
                "Either left or right (or both) must be provided."
            )

        parts = {}
        if left is not None:
            if not isinstance(left, np.ndarray):
                left = _io.read_array(left)
            parts["left"] = left
        if right is not None:
            if not isinstance(right, np.ndarray):
                right = _io.read_array(right)
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
        self,
        left: Mesh | str | Path | None = None,
        right: Mesh | str | Path | None = None,
    ) -> None:
        if left is None and right is None:
            raise ValueError(
                "Cannot create an empty PolyMesh. "
                "Either left or right (or both) must be provided."
            )

        self.parts = {}
        if left is not None:
            if not isinstance(left, Mesh):
                left = FileMesh(left).loaded()
            self.parts["left"] = left
        if right is not None:
            if not isinstance(right, Mesh):
                right = FileMesh(right).loaded()
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
                f"Data shape does not match number of vertices for '{key}':\n"
                f"- data shape: {data.parts[key].shape}\n"
                f"- n vertices: {mesh.parts[key].n_vertices}"
            )


class SurfaceImage:
    """Surface image, usually containing meshes & data for both hemispheres."""

    def __init__(
        self,
        mesh: PolyMesh | dict[str, Mesh | str | Path] | None,
        data: (
            PolyData | dict[str, Mesh | str | Path] | Nifti1Image | str | Path
        ),
    ) -> None:
        """Create a SurfaceImage instance.

        Parameters
        ----------
        mesh : PolyMesh | dict[str, Mesh  |  str  |  Path] | None
            Defaults to fsaverage if None is passed.
        data : PolyData | dict[str, Mesh  |  str  |  Path] | Niimg-like object
        """
        if mesh is None:
            from nilearn.experimental.surface._datasets import load_fsaverage

            fsaverage5 = load_fsaverage("fsaverage5")
            mesh = fsaverage5["pial"]
        self.mesh = mesh if isinstance(mesh, PolyMesh) else PolyMesh(**mesh)

        if not isinstance(data, (PolyData, dict, str, Path, Nifti1Image)):
            raise TypeError(
                "'data' must be one of"
                "[PolyData, dict, str, Path, Nifti1Image].\n"
                f"Got {type(data)}"
            )

        if isinstance(data, PolyData):
            self.data = data
        elif isinstance(data, dict):
            self.data = PolyData(**data)
        elif isinstance(data, (Nifti1Image, str, Path)):
            self._vol_to_surf(data)

        _check_data_and_mesh_compat(self.mesh, self.data)

        self.shape = self.data.shape

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {getattr(self, 'shape', '')}>"

    def _vol_to_surf(self, img: Nifti1Image | str | Path, **kwargs) -> None:
        """Project a Nifti image on a Surface.

        Parameters
        ----------
        img :  Niimg-like object, 3d or 4d.
               See :ref:`extracting_data`.

        kwargs:
               Extra arguments to pass
               to :func:`nilearn.surface.vol_to_surf`
        """
        if isinstance(img, (str, Path)):
            img = check_niimg(img)

        texture_left = vol_to_surf(img, self.mesh.parts["left"], **kwargs)
        texture_right = vol_to_surf(img, self.mesh.parts["right"], **kwargs)

        self.data = PolyData(left=texture_left.T, right=texture_right.T)

    def to_filename(self, filename: str | Path) -> None:
        """Save mesh to gifti.

        Parameters
        ----------
        filename : str | Path
                   If the filename contains `hemi-L`
                   then only the left part of the mesh will be saved.
                   If the filename contains `hemi-R`
                   then only the right part of the mesh will be saved.
                   If the filename contains neither of those,
                   then `_hemi-L` and `_hemi-R`
                   will be appended to the filename and both will be saved.
        """
        filename = Path(filename)

        if "hemi-L" in filename.stem and "hemi-R" in filename.stem:
            raise ValueError(
                "'filename' cannot contain both "
                "'hemi-L' and 'hemi-R'. \n"
                f"Got: {filename}"
            )

        if "hemi-L" not in filename.stem and "hemi-R" not in filename.stem:
            for hemi in ["L", "R"]:
                # TODO simplify when dropping python 3.8
                if sys.version_info.minor >= 9:
                    self.to_filename(
                        filename.with_stem(f"{filename.stem}_hemi-{hemi}")
                    )
                else:
                    self.to_filename(
                        _with_stem_compat(
                            filename, new_stem=f"{filename.stem}_hemi-{hemi}"
                        )
                    )

            return None

        if "hemi-L" in filename.stem:
            mesh = self.mesh.parts["left"]
        if "hemi-R" in filename.stem:
            mesh = self.mesh.parts["right"]
        mesh.to_gifti(filename)


def _with_stem_compat(path: Path, new_stem: str) -> Path:
    """Provide equivalent of `with_stem` for Python < 3.9.

    TODO remove when dropping python 3.8
    """
    return path.with_name(new_stem + path.suffix)

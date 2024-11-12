"""Surface API."""

from __future__ import annotations

import abc
import pathlib
from pathlib import Path

import numpy as np

from nilearn._utils.niimg_conversions import check_niimg
from nilearn.surface.surface import (
    data_to_gifti,
    load_surf_data,
    load_surf_mesh,
    mesh_to_gifti,
    vol_to_surf,
)


class PolyData:
    """A collection of data arrays.

    It is a shallow wrapper around the ``parts`` dictionary, which cannot be
    empty and whose keys must be a subset of {"left", "right"}.

    Parameters
    ----------
    left : numpy.ndarray or :obj:`str` of :obj:`pathlib.Path` or None,\
           default = None

    right : numpy.ndarray or :obj:`str` of :obj:`pathlib.Path` or None,\
           default = None

    Attributes
    ----------
    parts : dict[str, numpy.ndarray]

    shape : tuple[int, int]
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
                left = load_surf_data(left)
            parts["left"] = left
        if right is not None:
            if not isinstance(right, np.ndarray):
                right = load_surf_data(right)
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

    def to_filename(self, filename: str | Path) -> None:
        """Save data to gifti.

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
        filename = _sanitize_filename(filename)

        if "hemi-L" not in filename.stem and "hemi-R" not in filename.stem:
            for hemi in ["L", "R"]:
                self.to_filename(
                    filename.with_stem(f"{filename.stem}_hemi-{hemi}")
                )
            return None

        if "hemi-L" in filename.stem:
            data = self.parts["left"]
        if "hemi-R" in filename.stem:
            data = self.parts["right"]

        data_to_gifti(data, filename)


class Mesh(abc.ABC):
    """A surface :term:`mesh` having vertex, \
    coordinates and faces (triangles).

    Attributes
    ----------
    n_vertices : int
        number of vertices
    """

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
            Filename to save the mesh to.
        """
        mesh_to_gifti(self.coordinates, self.faces, gifti_file)


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
        self.n_vertices = load_surf_mesh(self.file_path).coordinates.shape[0]

    @property
    def coordinates(self) -> np.ndarray:
        """Get x, y, z, values for each mesh vertex."""
        return load_surf_mesh(self.file_path).coordinates

    @property
    def faces(self) -> np.ndarray:
        """Get array of adjacent vertices."""
        return load_surf_mesh(self.file_path).faces

    def loaded(self) -> InMemoryMesh:
        """Load surface mesh into memory."""
        loaded = load_surf_mesh(self.file_path)
        return InMemoryMesh(loaded.coordinates, loaded.faces)


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
        filename = _sanitize_filename(filename)

        if "hemi-L" not in filename.stem and "hemi-R" not in filename.stem:
            for hemi in ["L", "R"]:
                self.to_filename(
                    filename.with_stem(f"{filename.stem}_hemi-{hemi}")
                )
            return None

        if "hemi-L" in filename.stem:
            mesh = self.parts["left"]
        if "hemi-R" in filename.stem:
            mesh = self.parts["right"]

        mesh.to_gifti(filename)


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
    """Surface image, usually containing meshes & data for both hemispheres.


    Parameters
    ----------
    mesh : PolyMesh | dict[str, Mesh  |  str  |  Path]

    data : PolyData | dict[str, Mesh  |  str  |  Path]

    Attributes
    ----------
    shape : (int, int)
        shape of the surface data array
    """

    def __init__(
        self,
        mesh: PolyMesh | dict[str, Mesh | str | Path],
        data: PolyData | dict[str, Mesh | str | Path],
    ) -> None:
        """Create a SurfaceImage instance."""
        self.mesh = mesh if isinstance(mesh, PolyMesh) else PolyMesh(**mesh)

        if not isinstance(data, (PolyData, dict)):
            raise TypeError(
                "'data' must be one of"
                "[PolyData, dict].\n"
                f"Got {type(data)}"
            )

        if isinstance(data, PolyData):
            self.data = data
        elif isinstance(data, dict):
            self.data = PolyData(**data)

        _check_data_and_mesh_compat(self.mesh, self.data)

        self.shape = self.data.shape

    @classmethod
    def from_volume(
        cls, mesh, volume_img, inner_mesh=None, **vol_to_surf_kwargs
    ):
        """Create surface image from volume image.

        Parameters
        ----------
        mesh : PolyMesh or dict[str, Mesh | str | Path]
            Surface mesh.

        volume_img : Niimg-like object
            3D or 4D volume image to project to the surface mesh.

        inner_mesh: PolyMesh or dict[str, Mesh | str | Path], optional
            Inner mesh to pass to :func:`nilearn.surface.vol_to_surf`.

        vol_to_surf_kwargs: dict[str, Any]
            Dictionary of extra key-words arguments to pass
            to :func:`nilearn.surface.vol_to_surf`.

        Examples
        --------
        >>> from nilearn.experimental.surface import (
        ...     SurfaceImage,
        ...     load_fsaverage,
        ... )
        >>> from nilearn.datasets import load_sample_motor_activation_image

        >>> fsavg = load_fsaverage()
        >>> vol_img = load_sample_motor_activation_image()
        >>> img = SurfaceImage.from_volume(fsavg["white_matter"], vol_img)
        >>> img
        <SurfaceImage (20484,)>
        >>> img = SurfaceImage.from_volume(
        ...     fsavg["white_matter"], vol_img, inner_mesh=fsavg["pial"]
        ... )
        >>> img
        <SurfaceImage (20484,)>
        """
        mesh = mesh if isinstance(mesh, PolyMesh) else PolyMesh(**mesh)
        if inner_mesh is not None:
            inner_mesh = (
                inner_mesh
                if isinstance(inner_mesh, PolyMesh)
                else PolyMesh(**inner_mesh)
            )
            left_kwargs = {"inner_mesh": inner_mesh.parts["left"]}
            right_kwargs = {"inner_mesh": inner_mesh.parts["right"]}
        else:
            left_kwargs, right_kwargs = {}, {}

        if isinstance(volume_img, (str, Path)):
            volume_img = check_niimg(volume_img)

        texture_left = vol_to_surf(
            volume_img, mesh.parts["left"], **vol_to_surf_kwargs, **left_kwargs
        )
        texture_right = vol_to_surf(
            volume_img,
            mesh.parts["right"],
            **vol_to_surf_kwargs,
            **right_kwargs,
        )

        data = PolyData(left=texture_left.T, right=texture_right.T)

        return cls(mesh=mesh, data=data)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {getattr(self, 'shape', '')}>"


def _sanitize_filename(filename: str | Path) -> Path:
    filename = Path(filename)

    if not filename.suffix:
        filename = filename.with_suffix(".gii")
    if filename.suffix != ".gii":
        raise ValueError(
            "Mesh / Data should be saved as gifti files "
            "with the extension '.gii'.\n"
            f"Got '{filename.suffix}'."
        )

    if "hemi-L" in filename.stem and "hemi-R" in filename.stem:
        raise ValueError(
            "'filename' cannot contain both "
            "'hemi-L' and 'hemi-R'. \n"
            f"Got: {filename}"
        )
    return filename


def concatenate_surface_images(imgs):
    """Concatenate the data of a list or tuple of SurfaceImages.

    Assumes all images have same meshes.
    """
    if not isinstance(imgs, (tuple, list)) or any(
        not isinstance(x, SurfaceImage) for x in imgs
    ):
        raise TypeError(
            "'imgs' must be a list or a tuple of SurfaceImage instances."
        )
    output = imgs[0]

    if len(imgs) == 1:
        return output

    for part in output.data.parts:
        tmp = [x.data.parts[part] for x in imgs]
        output.data.parts[part] = np.concatenate(tmp)

    return output

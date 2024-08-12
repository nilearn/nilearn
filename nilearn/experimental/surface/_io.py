"""Input/output for surface data and meshes."""

import pathlib
from typing import Dict, Union

import nibabel as nib
import numpy as np

from nilearn import surface as old_surface


def read_array(array_file: Union[pathlib.Path, str]) -> np.ndarray:
    """Load surface data into a Numpy array."""
    return old_surface.load_surf_data(array_file)


def read_mesh(mesh_file: Union[pathlib.Path, str]) -> Dict[str, np.ndarray]:
    """Load surface mesh geometry into Numpy arrays."""
    loaded = old_surface.load_surf_mesh(mesh_file)
    return {"coordinates": loaded.coordinates, "faces": loaded.faces}


def mesh_to_gifti(
    coordinates: np.ndarray,
    faces: np.ndarray,
    gifti_file: Union[pathlib.Path, str],
) -> None:
    """Write surface mesh to gifti file on disk."""
    gifti_file = pathlib.Path(gifti_file)
    gifti_img = nib.gifti.GiftiImage()
    coords_array = nib.gifti.GiftiDataArray(
        coordinates, intent="NIFTI_INTENT_POINTSET", datatype="float32"
    )
    faces_array = nib.gifti.GiftiDataArray(
        faces, intent="NIFTI_INTENT_TRIANGLE", datatype="int32"
    )
    gifti_img.add_gifti_data_array(coords_array)
    gifti_img.add_gifti_data_array(faces_array)
    nib.save(gifti_img, gifti_file)

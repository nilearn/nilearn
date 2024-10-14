"""Input/output for surface data and meshes."""

import pathlib
from typing import Union

import numpy as np
from nibabel import gifti

from nilearn import surface as old_surface


def read_array(array_file: Union[pathlib.Path, str]) -> np.ndarray:
    """Load surface data into a Numpy array."""
    return old_surface.load_surf_data(array_file)


def read_mesh(mesh_file: Union[pathlib.Path, str]) -> dict[str, np.ndarray]:
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
    gifti_img = gifti.GiftiImage()
    coords_array = gifti.GiftiDataArray(
        coordinates, intent="NIFTI_INTENT_POINTSET", datatype="float32"
    )
    faces_array = gifti.GiftiDataArray(
        faces, intent="NIFTI_INTENT_TRIANGLE", datatype="int32"
    )
    gifti_img.add_gifti_data_array(coords_array)
    gifti_img.add_gifti_data_array(faces_array)
    gifti_img.to_filename(gifti_file)


def data_to_gifti(data: np.ndarray, gifti_file: pathlib.Path | str):
    # see
    # https://github.com/nipy/nibabel/blob/master/nibabel/gifti/gifti.py
    # Only the following are 'supported' for now
    # - NIFTI_TYPE_UINT8
    # - NIFTI_TYPE_INT32
    # - NIFTI_TYPE_FLOAT32
    if data.dtype in [np.uint16, np.uint32, np.uint64]:
        data = data.astype(np.uint8)
    elif data.dtype in [np.int8, np.int16, np.int64]:
        data = data.astype(np.int32)
    elif data.dtype in [np.float64]:
        data = data.astype(np.float32)

    if data.dtype == np.uint8:
        datatype = "NIFTI_TYPE_UINT8"
    elif data.dtype == np.int32:
        datatype = "NIFTI_TYPE_INT32"
    elif data.dtype == np.float32:
        datatype = "NIFTI_TYPE_FLOAT32"

    darray = gifti.GiftiDataArray(data=data, datatype=datatype)

    gii = gifti.GiftiImage(darrays=[darray])
    gii.to_filename(pathlib.Path(gifti_file))

import pathlib
from typing import Dict, Union

import numpy as np
import nibabel as nib

from nilearn import surface as old_surface


def read_array(array_file: Union[pathlib.Path, str]) -> np.ndarray:
    return old_surface.load_surf_data(array_file)


def read_mesh(mesh_file: Union[pathlib.Path, str]) -> Dict[str, np.ndarray]:
    loaded = old_surface.load_surf_mesh(mesh_file)
    return {"coordinates": loaded.coordinates, "faces": loaded.faces}


def mesh_to_gifti(
    coordinates: np.ndarray,
    faces: np.ndarray,
    gifti_file: Union[pathlib.Path, str],
) -> None:
    gifti_file = pathlib.Path(gifti_file)
    gifti_img = nib.gifti.GiftiImage()
    coords_array = nib.gifti.GiftiDataArray(
        coordinates, intent="NIFTI_INTENT_POINTSET"
    )
    faces_array = nib.gifti.GiftiDataArray(
        faces, intent="NIFTI_INTENT_TRIANGLE"
    )
    gifti_img.add_gifti_data_array(coords_array)
    gifti_img.add_gifti_data_array(faces_array)
    nib.save(gifti_img, gifti_file)

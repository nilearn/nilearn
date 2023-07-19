import pathlib
from typing import Dict, Union

import numpy as np

from nilearn import surface as old_surface


def read_array(array_file: Union[pathlib.Path, str]) -> np.ndarray:
    return old_surface.load_surf_data(array_file)


def read_mesh(mesh_file: Union[pathlib.Path, str]) -> Dict[str, np.ndarray]:
    loaded = old_surface.load_surf_mesh(mesh_file)
    return {"coordinates": loaded.coordinates, "faces": loaded.faces}

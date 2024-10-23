import numpy as np

from nilearn import image
from nilearn.experimental.surface._surface_image import PolyMesh, SurfaceImage


def _check_dims(imgs):
    # check dims of one image if given a list
    if isinstance(imgs, list):
        im = imgs[0]
        dim = image.load_img(im).shape
        # in case of 4D (timeseries) + 1D (subjects) return first subject
        if len(dim) == 4:
            return im, (*dim, 1)
        else:
            return imgs, (*dim, 1)
    else:
        dim = image.load_img(imgs).shape
        return imgs, dim


def compute_middle_image(img):
    """Compute middle image of timeseries (4D data)."""
    img, dim = _check_dims(img)
    if len(dim) == 4 or len(dim) == 5:
        img = image.index_img(img, dim[-1] // 2)
    return img, len(dim)


def check_same_n_vertices(mesh_1: PolyMesh, mesh_2: PolyMesh) -> None:
    """Check that 2 meshes have the same keys and that n vertices match."""
    keys_1, keys_2 = set(mesh_1.parts.keys()), set(mesh_2.parts.keys())
    if keys_1 != keys_2:
        diff = keys_1.symmetric_difference(keys_2)
        raise ValueError(
            f"Meshes do not have the same keys. Offending keys: {diff}"
        )
    for key in keys_1:
        if mesh_1.parts[key].n_vertices != mesh_2.parts[key].n_vertices:
            raise ValueError(
                f"Number of vertices do not match for '{key}'."
                "number of vertices in mesh_1: "
                f"{mesh_1.parts[key].n_vertices}; "
                f"in mesh_2: {mesh_2.parts[key].n_vertices}"
            )


def _compute_mean_image(img: SurfaceImage):
    """Compute mean of the surface (for 'time series')."""
    if len(img.shape) <= 1:
        return img
    for part, value in img.data.parts.items():
        img.data.parts[part] = np.squeeze(value.mean(axis=0)).astype(float)
    return img


def _get_min_max(img: SurfaceImage):
    """Get min and max across hemisphere for a SurfaceImage."""
    vmin = min(min(x) for x in img.data.parts.values())
    vmax = max(max(x) for x in img.data.parts.values())
    return vmin, vmax

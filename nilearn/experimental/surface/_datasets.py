# """Fetching a few example datasets to use during development.

# eventually nilearn.datasets would be updated
# """

from __future__ import annotations

from collections.abc import Sequence

from nilearn import datasets

# from nilearn.datasets.struct import load_fsaverage_data
from nilearn.experimental.surface import _io
from nilearn.experimental.surface._surface_image import (
    SurfaceImage,
)


def load_fsaverage(
    mesh_name: str = "fsaverage5",
):
    return datasets.fetch_surf_fsaverage(mesh=mesh_name, as_polymesh=True)


ALLOWED_MESH_TYPES = (
    "pial",
    "white_matter",
    "inflated",
    "sphere",
    "flat",
)


def fetch_nki(mesh_type: str = "pial", **kwargs) -> Sequence[SurfaceImage]:
    """Load NKI enhanced surface data into a surface object."""
    if mesh_type not in ALLOWED_MESH_TYPES:
        raise ValueError(
            f"'mesh_type' must be one of {ALLOWED_MESH_TYPES}.\n"
            f"Got: {mesh_type}."
        )

    fsaverage = load_fsaverage("fsaverage5")

    nki_dataset = datasets.fetch_surf_nki_enhanced(**kwargs)

    images = []
    for left, right in zip(
        nki_dataset["func_left"], nki_dataset["func_right"]
    ):
        left_data = _io.read_array(left).T
        right_data = _io.read_array(right).T
        img = SurfaceImage(
            mesh=fsaverage[mesh_type],
            data={
                "left": left_data,
                "right": right_data,
            },
        )
        images.append(img)

    return images


def fetch_destrieux(): ...

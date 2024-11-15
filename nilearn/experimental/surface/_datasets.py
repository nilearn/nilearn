"""Fetching a few example datasets to use during development.

eventually nilearn.datasets would be updated
"""

from __future__ import annotations

from collections.abc import Sequence

from nilearn import datasets
from nilearn.surface import (
    FileMesh,
    PolyMesh,
    SurfaceImage,
    load_surf_data,
)


def load_fsaverage(
    mesh_name: str = "fsaverage5",
) -> dict[str, PolyMesh]:
    """Load fsaverage for both hemispheres."""
    fsaverage = datasets.fetch_surf_fsaverage(mesh_name)
    renaming = {
        "pial": "pial",
        "white": "white_matter",
        "infl": "inflated",
        "sphere": "sphere",
        "flat": "flat",
    }
    meshes = {}
    for key, value in renaming.items():
        left = FileMesh(fsaverage[f"{key}_left"])
        right = FileMesh(fsaverage[f"{key}_right"])
        meshes[value] = PolyMesh(left=left, right=right)
    return meshes


ALLOWED_DATA_TYPES = (
    "curvature",
    "sulcal",
    "thickness",
)


def load_fsaverage_data(
    mesh_name: str = "fsaverage5",
    mesh_type: str = "pial",
    data_type: str = "sulcal",
) -> SurfaceImage:
    """Return freesurfer data on an fsaverage mesh."""
    if mesh_type not in ALLOWED_MESH_TYPES:
        raise ValueError(
            f"'mesh_type' must be one of {ALLOWED_MESH_TYPES}.\n"
            f"Got: {mesh_type}."
        )
    if data_type not in ALLOWED_DATA_TYPES:
        raise ValueError(
            f"'data_type' must be one of {ALLOWED_DATA_TYPES}.\n"
            f"Got: {data_type}."
        )

    fsaverage = load_fsaverage(mesh_name)
    old_fsaverage = datasets.fetch_surf_fsaverage(mesh_name)

    renaming = {"curvature": "curv", "sulcal": "sulc", "thickness": "thick"}
    img = SurfaceImage(
        mesh=fsaverage[mesh_type],
        data={
            "left": load_surf_data(
                old_fsaverage[f"{renaming[data_type]}_left"]
            ),
            "right": load_surf_data(
                old_fsaverage[f"{renaming[data_type]}_right"]
            ),
        },
    )

    return img


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
        left_data = load_surf_data(left).T
        right_data = load_surf_data(right).T
        img = SurfaceImage(
            mesh=fsaverage[mesh_type],
            data={
                "left": left_data,
                "right": right_data,
            },
        )
        images.append(img)

    return images

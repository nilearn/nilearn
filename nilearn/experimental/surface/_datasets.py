"""Fetching a few example datasets to use during development.

eventually nilearn.datasets would be updated
"""

from typing import Dict, Sequence, Tuple

from nilearn import datasets
from nilearn.experimental.surface import _io
from nilearn.experimental.surface._surface_image import (
    FileMesh,
    PolyMesh,
    SurfaceImage,
)


def load_fsaverage(mesh_name: str = "fsaverage5") -> Dict[str, PolyMesh]:
    """Load several fsaverage mesh types for both hemispheres."""
    fsaverage = datasets.fetch_surf_fsaverage(mesh_name)
    meshes: Dict[str, PolyMesh] = {}
    renaming = {"pial": "pial", "white": "white_matter", "infl": "inflated"}
    for mesh_type, mesh_name in renaming.items():
        parts = {}
        for hemisphere in "left", "right":
            parts[hemisphere] = FileMesh(
                fsaverage[f"{mesh_type}_{hemisphere}"]
            )
        meshes[mesh_name] = PolyMesh(**parts)
    return meshes


def fetch_nki(n_subjects=1) -> Sequence[SurfaceImage]:
    """Load NKI enhanced surface data into a surface object."""
    fsaverage = load_fsaverage("fsaverage5")
    nki_dataset = datasets.fetch_surf_nki_enhanced(n_subjects=n_subjects)
    images = []
    for left, right in zip(
        nki_dataset["func_left"], nki_dataset["func_right"]
    ):
        left_data = _io.read_array(left).T
        right_data = _io.read_array(right).T
        img = SurfaceImage(
            mesh=fsaverage["pial"],
            data={
                "left": left_data,
                "right": right_data,
            },
        )
        images.append(img)
    return images


def fetch_destrieux() -> Tuple[SurfaceImage, Dict[int, str]]:
    """Load Destrieux surface atlas into a surface object."""
    fsaverage = load_fsaverage("fsaverage5")
    destrieux = datasets.fetch_atlas_surf_destrieux()
    label_names = {
        i: label.decode("utf-8") for (i, label) in enumerate(destrieux.labels)
    }
    return (
        SurfaceImage(
            mesh=fsaverage["pial"],
            data={
                "left": destrieux["map_left"],
                "right": destrieux["map_right"],
            },
        ),
        label_names,
    )

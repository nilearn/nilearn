"""Generate HTML figure to run JS tests."""

from pathlib import Path

import numpy as np

from nilearn.datasets import (
    fetch_atlas_surf_destrieux,
    load_fsaverage,
    load_fsaverage_data,
    load_sample_motor_activation_image,
)
from nilearn.plotting import (
    view_connectome,
    view_img,
    view_img_on_surf,
    view_markers,
    view_surf,
)
from nilearn.surface import SurfaceImage

output_path = Path(__file__).parent

WIDTH = 1200
HEIGHT = 800

fig = view_img(load_sample_motor_activation_image())
fig.resize(WIDTH, HEIGHT)
fig.save_as_html(output_path / "view_img.html")

fig = view_surf(surf_map=load_fsaverage_data())
fig.resize(WIDTH, HEIGHT)
fig.save_as_html(output_path / "view_surf.html")

fig = view_surf(surf_map=load_fsaverage_data(), engine="niivue")
fig.resize(WIDTH, HEIGHT)
fig.save_as_html(output_path / "view_surf_niivue.html")

fig = view_img_on_surf(load_sample_motor_activation_image())
fig.resize(WIDTH, HEIGHT)
fig.save_as_html(output_path / "view_img_on_surf.html")

fsaverage = load_fsaverage("fsaverage5")
destrieux = fetch_atlas_surf_destrieux()
destrieux_atlas = SurfaceImage(
    mesh=fsaverage["pial"],
    data={
        "left": destrieux["map_left"],
        "right": destrieux["map_right"],
    },
)

coordinates = []
for hemi in ["left", "right"]:
    data = destrieux_atlas.data.parts[hemi]
    mesh_coordinates = destrieux_atlas.mesh.parts[hemi].coordinates
    coordinates.extend(
        np.mean(mesh_coordinates[data == k], axis=0)
        for k, label in enumerate(destrieux.labels)
        if "Unknown" not in str(label)
    )
coordinates = np.array(coordinates)


n_parcels = len(coordinates)
corr = np.zeros((n_parcels, n_parcels))
n_parcels_hemi = n_parcels // 2
corr[np.arange(n_parcels_hemi), np.arange(n_parcels_hemi) + n_parcels_hemi] = 1
corr = corr + corr.T

fig = view_connectome(
    corr,
    coordinates,
    edge_threshold="90%",
)
fig.resize(WIDTH, HEIGHT)
fig.save_as_html(output_path / "view_connectome.html")

coords = np.arange(12).reshape((4, 3))

fig = view_markers(
    coords,
    marker_size=5.0,
    marker_color=["r", "g", "black", "white"],
    marker_labels=[
        "red marker",
        "green marker",
        "black marker",
        "white marker",
    ],
    title="test view_markers",
    title_fontsize=30,
)
fig.resize(WIDTH, HEIGHT)
fig.save_as_html(output_path / "view_markers.html")

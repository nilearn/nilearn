"""TODO."""

import matplotlib.pyplot as plt
from neuromaps.transforms import mni152_to_fsaverage

from nilearn.datasets import load_fsaverage, load_sample_motor_activation_image
from nilearn.plotting import plot_surf_stat_map, show
from nilearn.surface import SurfaceImage

stat_img = load_sample_motor_activation_image()

hemi = "both"

for space, fsavg_density in zip(
    ["fsaverage4", "fsaverage5", "fsaverage6", "fsaverage7"],
    ["3k", "10k", "41k", "164k"],
    strict=False,
):
    fsaverage_meshes = load_fsaverage(mesh=space)

    fslr = mni152_to_fsaverage(stat_img, fsavg_density=fsavg_density)
    fslr_lh, fslr_rh = fslr
    surface_image_neuromaps = SurfaceImage(
        mesh=fsaverage_meshes["pial"], data={"left": fslr_lh, "right": fslr_rh}
    )

    surface_image = SurfaceImage.from_volume(
        mesh=fsaverage_meshes["pial"],
        volume_img=stat_img,
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(15, 5), subplot_kw={"projection": "3d"}
    )

    plot_surf_stat_map(
        stat_map=surface_image,
        surf_mesh=fsaverage_meshes["inflated"],
        hemi=hemi,
        title="nilearn",
        threshold=1.0,
        axes=axes[0],
    )

    plot_surf_stat_map(
        stat_map=surface_image_neuromaps,
        surf_mesh=fsaverage_meshes["inflated"],
        hemi=hemi,
        title="neuromaps",
        threshold=1.0,
        axes=axes[1],
    )

    fig.suptitle(f"{space}: {fsavg_density}")

show()

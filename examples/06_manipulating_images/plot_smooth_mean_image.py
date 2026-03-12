"""
Smoothing an image
==================

Here we smooth a mean :term:`EPI` image and plot the result

As we vary the smoothing :term:`FWHM`,
note how we decrease the amount of noise, but also lose spatial details.
In general, the best amount of smoothing for a given analysis
depends on the spatial extent of the effects that are expected.

We then show how to smooth a SurfaceImage.

"""

# %%
from nilearn.datasets import fetch_development_fmri
from nilearn.image import mean_img, smooth_img
from nilearn.plotting import plot_epi, show

data = fetch_development_fmri(n_subjects=1)

# Print basic information on the dataset
print(
    f"First subject functional nifti image (4D) are located at: {data.func[0]}"
)

first_epi_file = data.func[0]

# First compute the mean image, from the 4D series of image
mean_func = mean_img(first_epi_file)

# %%
# Then we smooth, with a varying amount of smoothing, from none to 20mm
# by increments of 5mm
for fwhm in [None, 5, 10, 15, 20, 25]:
    smoothed_img = smooth_img(mean_func, fwhm)
    title = "No smoothing" if fwhm is None else f"Smoothing {fwhm} mm"
    plot_epi(
        smoothed_img,
        title=title,
        colorbar=True,
        cmap="gray",
        vmin=0,
    )

show()


# %%
from nilearn.datasets import (
    load_fsaverage,
    load_fsaverage_data,
    load_sample_motor_activation_image,
)
from nilearn.plotting import plot_surf_stat_map
from nilearn.surface import SurfaceImage

fsaverage_meshes = load_fsaverage()

stat_img = load_sample_motor_activation_image()

curvature = load_fsaverage_data(data_type="curvature")

surface_image = SurfaceImage.from_volume(
    mesh=fsaverage_meshes["pial"],
    volume_img=stat_img,
)

for fwhm in [None, 5, 10, 15, 20, 25]:
    smoothed_img = smooth_img(surface_image, fwhm)
    title = "No smoothing" if fwhm is None else f"Smoothing {fwhm} mm"
    plot_surf_stat_map(
        surf_mesh=fsaverage_meshes["inflated"],
        stat_map=smoothed_img,
        title=title,
        threshold=1.0,
        vmax=8,
        bg_map=curvature,
    )

show()

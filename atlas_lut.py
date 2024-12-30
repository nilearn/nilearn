"""Demo how to convert atlas labels to TSV."""

from itertools import product

from rich import print

from nilearn.datasets import (
    fetch_atlas_aal,
    fetch_atlas_basc_multiscale_2015,
    fetch_atlas_destrieux_2009,
    fetch_atlas_harvard_oxford,
    fetch_atlas_juelich,
    fetch_atlas_pauli_2017,
    fetch_atlas_schaefer_2018,
    fetch_atlas_surf_destrieux,
    fetch_atlas_talairach,
    fetch_atlas_yeo_2011,
)

# %%
print(fetch_atlas_destrieux_2009.__name__.upper())
atlas = fetch_atlas_destrieux_2009()

# %%
print(fetch_atlas_yeo_2011.__name__.upper())
atlas = fetch_atlas_yeo_2011()

#  %%
print(fetch_atlas_aal.__name__.upper())
atlas = fetch_atlas_aal(version="SPM12")
atlas = fetch_atlas_aal(version="SPM8")
atlas = fetch_atlas_aal(version="SPM5")
atlas = fetch_atlas_aal(version="3v2")

# %%
# TODO with surface image
print(fetch_atlas_surf_destrieux.__name__.upper())
atlas = fetch_atlas_surf_destrieux()

# %%
print(fetch_atlas_talairach.__name__.upper())
atlas = fetch_atlas_talairach(level_name="ba")
atlas = fetch_atlas_talairach(level_name="hemisphere")
atlas = fetch_atlas_talairach(level_name="lobe")
atlas = fetch_atlas_talairach(level_name="gyrus")
atlas = fetch_atlas_talairach(level_name="tissue")

# %%
n_rois = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
yeo_networks = {7, 17}
resolution_mm = {1, 2}
print(fetch_atlas_schaefer_2018.__name__.upper())
for n_roi, network, res in product(n_rois, yeo_networks, resolution_mm):
    atlas = fetch_atlas_schaefer_2018(
        resolution_mm=res, yeo_networks=network, n_rois=n_roi
    )


# %%
print(fetch_atlas_pauli_2017.__name__.upper())
atlas = fetch_atlas_pauli_2017(atlas_type="deterministic")


# %%
print(fetch_atlas_harvard_oxford.__name__.upper())
atlas = fetch_atlas_harvard_oxford(atlas_name="cort-maxprob-thr0-1mm")

#  %%
print(fetch_atlas_juelich.__name__.upper())
atlas = fetch_atlas_juelich(atlas_name="maxprob-thr50-2mm")

# %%
print(fetch_atlas_basc_multiscale_2015.__name__.upper())
resolution = 325
atlas = fetch_atlas_basc_multiscale_2015(resolution=resolution)

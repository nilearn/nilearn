"""Demo how to convert atlas labels to TSV."""

import pandas as pd
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
from nilearn.datasets.atlas import _check_look_up_table, rgb_to_hex_lookup

# %%
print(fetch_atlas_destrieux_2009.__name__.upper())
atlas = fetch_atlas_destrieux_2009()

# %%
# TODO do also 17 networks
print(fetch_atlas_yeo_2011.__name__.upper())
atlas = fetch_atlas_yeo_2011()
# lut = _generate_atlas_look_up_table(
#     fetch_atlas_yeo_2011, index=atlas.indices, name=atlas.labels
# )
# print(lut)
lut = pd.read_csv(
    atlas.colors_7,
    sep="\\s+",
    names=["index", "name", "r", "g", "b", "fs"],
    header=0,
)
lut = pd.concat(
    [pd.DataFrame([[0, "Background", 0, 0, 0, 0]], columns=lut.columns), lut],
    ignore_index=True,
)
lut["color"] = "#" + rgb_to_hex_lookup(lut.r, lut.g, lut.b).astype(str)
lut = lut.drop(["r", "g", "b", "fs"], axis=1)

_check_look_up_table(lut, atlas.thin_7)

#  %%
# TODO try all versions
print(fetch_atlas_aal.__name__.upper())
atlas = fetch_atlas_aal()

# %%
# TODO with surface image
print(fetch_atlas_surf_destrieux.__name__.upper())
atlas = fetch_atlas_surf_destrieux()

# %%
# TODO try all level_name
print(fetch_atlas_talairach.__name__.upper())
atlas = fetch_atlas_talairach(level_name="ba")

# %%
# TODO: try all n_rois and yeos
print(fetch_atlas_schaefer_2018.__name__.upper())
atlas = fetch_atlas_schaefer_2018()


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

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

dec_to_hex_nums = pd.DataFrame(
    {"hex": [f"{x:02x}" for x in range(256)]}, dtype=str
)


def rgb_to_hex_lookup(
    red: pd.Series, green: pd.Series, blue: pd.Series
) -> pd.Series:
    """Turn RGB in hex."""
    # see https://stackoverflow.com/questions/53875880/convert-a-pandas-dataframe-of-rgb-colors-to-hex
    # Look everything up
    rr = dec_to_hex_nums.loc[red, "hex"]
    gg = dec_to_hex_nums.loc[green, "hex"]
    bb = dec_to_hex_nums.loc[blue, "hex"]
    # Reindex
    rr.index = red.index
    gg.index = green.index
    bb.index = blue.index
    # Concatenate and return
    return rr + gg + bb


def _generate_atlas_look_up_table(function, labels, index=None):
    if function.__name__ in [
        "fetch_atlas_aal",
        "fetch_atlas_talairach",
        "fetch_atlas_pauli_2017",
        "fetch_atlas_harvard_oxford",
        "fetch_atlas_juelich",
    ]:
        name = labels
    elif function.__name__ in [
        "fetch_atlas_surf_destrieux",
        "fetch_atlas_schaefer_2018",
    ]:
        name = [x.decode() for x in labels]
    elif function.__name__ in ["fetch_atlas_basc_multiscale_2015"]:
        name = [str(x) for x in range(labels)]

    if function.__name__ in [
        "fetch_atlas_surf_destrieux",
        "fetch_atlas_talairach",
        "fetch_atlas_harvard_oxford",
        "fetch_atlas_juelich",
    ]:
        index = list(range(len(labels)))
    elif function.__name__ in ["fetch_atlas_basc_multiscale_2015"]:
        index = list(range(labels))

    if function.__name__ in [
        "fetch_atlas_harvard_oxford",
        "fetch_atlas_juelich",
        "fetch_atlas_talairach",
        "fetch_atlas_aal",
        "fetch_atlas_basc_multiscale_2015",
    ]:
        return pd.DataFrame({"index": index, "name": name})

    elif function.__name__ == "fetch_atlas_surf_destrieux":
        lut = pd.DataFrame(
            {
                "index": index,
                "name": name,
            }
        )
        return lut.replace("Unknown", "Background")

    elif function.__name__ in [
        "fetch_atlas_schaefer_2018",
        "fetch_atlas_pauli_2017",
    ]:
        lut = pd.DataFrame(
            {
                "index": list(range(1, len(name) + 1)),
                "name": name,
            }
        )
        return pd.concat(
            [pd.DataFrame([[0, "Background"]], columns=lut.columns), lut],
            ignore_index=True,
        )


# %%
atlas = fetch_atlas_destrieux_2009()
print(atlas.labels)

# %%
# TODO do also 17 networks
atlas = fetch_atlas_yeo_2011()
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
print(lut)

#  %%
# TODO try all versions
atlas = fetch_atlas_aal()
lut = _generate_atlas_look_up_table(
    fetch_atlas_aal, index=atlas.index, labels=atlas.labels
)
print(lut)

# %%

atlas = fetch_atlas_surf_destrieux()
lut = _generate_atlas_look_up_table(
    fetch_atlas_surf_destrieux, labels=atlas.labels
)
print(lut)

# %%
# TODO try all level_name
atlas = fetch_atlas_talairach(level_name="ba")
lut = _generate_atlas_look_up_table(fetch_atlas_talairach, labels=atlas.labels)
print(lut)


# %%
# TODO: try all n_rois and yeos
atlas = fetch_atlas_schaefer_2018()
lut = _generate_atlas_look_up_table(
    fetch_atlas_schaefer_2018, labels=atlas.labels
)
print(lut)


# %%
atlas = fetch_atlas_pauli_2017(version="det")
lut = _generate_atlas_look_up_table(
    fetch_atlas_pauli_2017, labels=atlas.labels
)
print(lut)


# %%
atlas = fetch_atlas_harvard_oxford(atlas_name="cort-maxprob-thr0-1mm")
lut = _generate_atlas_look_up_table(
    fetch_atlas_pauli_2017, labels=atlas.labels
)
print(lut)

#  %%
atlas = fetch_atlas_juelich(atlas_name="maxprob-thr50-2mm")
lut = _generate_atlas_look_up_table(fetch_atlas_juelich, labels=atlas.labels)
print(lut)


# %%
resolution = 444
atlas = fetch_atlas_basc_multiscale_2015(resolution=resolution)
lut = _generate_atlas_look_up_table(
    fetch_atlas_basc_multiscale_2015, labels=resolution
)
print(lut)

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "nilearn[min_plotting]",
#    "tabulate"
# ]
# ///
"""Generate markdown files with table summarizing information about atlases."""

from pathlib import Path
from ssl import SSLCertVerificationError

import matplotlib.pyplot as plt
import pandas as pd
from requests.exceptions import SSLError
from urllib3.exceptions import MaxRetryError

from nilearn.datasets import (
    fetch_atlas_aal,
    fetch_atlas_allen_2011,
    fetch_atlas_basc_multiscale_2015,
    fetch_atlas_craddock_2012,
    fetch_atlas_destrieux_2009,
    fetch_atlas_difumo,
    fetch_atlas_harvard_oxford,
    fetch_atlas_juelich,
    fetch_atlas_msdl,
    fetch_atlas_pauli_2017,
    fetch_atlas_schaefer_2018,
    fetch_atlas_smith_2009,
    fetch_atlas_surf_destrieux,
    fetch_atlas_talairach,
    fetch_atlas_yeo_2011,
    load_fsaverage,
    load_fsaverage_data,
)
from nilearn.maskers import (
    NiftiLabelsMasker,
    NiftiMapsMasker,
    SurfaceLabelsMasker,
)
from nilearn.plotting import plot_prob_atlas, plot_roi, plot_surf_roi
from nilearn.surface import SurfaceImage


def _update_dict(dict_for_df, name, fn, data, doc_dir, output_file, n_rois=1):
    """Update dictionary to use to create dataframe of atlases."""
    fn_name = fn.__name__.replace("fetch_atlas_", "")

    dict_for_df["name"].append(
        f"**{name.replace('_', ' ')}**<br>"
        + f"*template*: {data.template}<br>"
        + f"*number of regions*: {n_rois}<br>"
        + "{ref}`description "
        + f"<{fn_name}_atlas>`<br>"
    )

    dict_for_df["image"].append(
        f"![name](../{output_file.relative_to(doc_dir)!s})"
    )

    return dict_for_df


def _generate_markdown_file(filename, dict_for_df):
    """Generate a markdown file with a table of atlases."""
    atlas_table = pd.DataFrame(dict_for_df)
    atlas_table = atlas_table.sort_values(by=["name"])

    markdown_file = doc_dir / "modules" / filename
    with markdown_file.open("w") as f:
        f.write(f"""
<!--
!!!!! DO NOT EDIT MANUALLY !!!!!
This file is auto-generated.
To modify the content of this file do it via the script:
{Path(__file__).name}
-->
""")

        f.write("""
```{warning}
The atlases shipped with Nilearn
do not necessarily use the same MNI template
as the default MNI template used by Nilearn for plotting.

This may lead to atlas being poorly coregistered
to the underlay image:
the atlas can appear smaller or bigger than the brain.

This can be seen clearly in some of the images below.

You also should not use maskers with an atlas
that is not coregistered properly
with the images you want to extract data from
as this may lead to invalid results.
```

""")
        atlas_table.to_markdown(buf=f, index=False)


DEBUG = False
GENERATE_FIG = True

doc_dir = Path(__file__).parent
output_dir = doc_dir / "images"

plot_config = {
    "draw_cross": False,
    "colorbar": True,
    "display_mode": "ortho",
    "cut_coords": [0, 0, 0],
}


"""
VOLUME DETERMINISTIC ATLASES
"""

# dict to define fetching options for each atlas
deterministic_atlases = {
    "AAL": {"fn": fetch_atlas_aal, "n_rois": [117, 167]},
    "BASC multiscale (2015)": {
        "fn": fetch_atlas_basc_multiscale_2015,
        "params": {"resolution": 20, "version": "sym"},
        "n_rois": [7, 12, 20, 36, 64, 122, 197, 325, 444],
    },
    "Destrieux (2009; volume)": {"fn": fetch_atlas_destrieux_2009},
    "Harvard-Oxford (cortical)": {
        "fn": fetch_atlas_harvard_oxford,
        "params": {
            "atlas_name": "cort-maxprob-thr0-1mm",
            "symmetric_split": False,
        },
    },
    "Harvard-Oxford (subcortical)": {
        "fn": fetch_atlas_harvard_oxford,
        "params": {
            "atlas_name": "sub-maxprob-thr0-1mm",
            "symmetric_split": False,
        },
    },
    "Juelich": {
        "fn": fetch_atlas_juelich,
        "params": {"atlas_name": "maxprob-thr0-1mm"},
    },
    "Pauli (2017)": {
        "fn": fetch_atlas_pauli_2017,
        "params": {"atlas_type": "deterministic"},
    },
    "Schaefer (2018)": {
        "fn": fetch_atlas_schaefer_2018,
        "n_rois": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    },
    "Talairach": {
        "fn": fetch_atlas_talairach,
        "params": {"level_name": "ba"},
        "n_rois": [
            3,
            7,
            12,
            55,
            71,
        ],
    },
    "Yeo (2011)": {
        "fn": fetch_atlas_yeo_2011,
        "params": {"n_networks": 17, "thickness": "thick"},
        "n_rois": [7, 17],
    },
}

dict_for_df = {"name": [], "image": []}

for display_name, details in deterministic_atlases.items():
    fn = details["fn"]
    params = details.get("params", {})

    try:
        data = fn(**params)
    except (SSLError, MaxRetryError, SSLCertVerificationError):
        continue

    name = fn.__name__.replace("fetch_atlas_", "")

    extra_title = [f"{k}={v}" for k, v in params.items()]
    title = f"{fn.__name__}({', '.join(extra_title)})"

    params_str = ""
    for k, v in params.items():
        params_str += f"_{k}-{v}"
    output_file = output_dir / f"deterministic_atlas_{name}{params_str}.png"

    if GENERATE_FIG:
        plot_roi(
            data.maps,
            title=title,
            cmap=data.lut,
            output_file=output_file,
            figure=plt.figure(figsize=[11, 4]),
            **plot_config,
        )
    plt.close("all")

    n_rois = details.get("n_rois")
    if n_rois is None:
        masker = NiftiLabelsMasker(labels_img=data.maps).fit()
        n_rois = masker.n_elements_

    dict_for_df = _update_dict(
        dict_for_df,
        display_name,
        fn,
        data,
        doc_dir,
        output_file=output_file,
        n_rois=n_rois,
    )

#
# SURFACE DETERMINISTIC ATLASES
#
# Surface atlases are bit different so we deal with them separately.
#

fsaverage = load_fsaverage("fsaverage5")
fsaverage_sulcal = load_fsaverage_data(data_type="sulcal")
destrieux = fetch_atlas_surf_destrieux()
destrieux_atlas = SurfaceImage(
    mesh=fsaverage["inflated"],
    data={
        "left": destrieux["map_left"],
        "right": destrieux["map_right"],
    },
)

name = fetch_atlas_surf_destrieux.__name__.replace("fetch_atlas_", "")

output_file = output_dir / f"deterministic_atlas_{name}.png"

if GENERATE_FIG:
    fig, ax = plt.subplots(
        1, 2, figsize=[6, 3], subplot_kw={"projection": "3d"}
    )

    plot_surf_roi(
        roi_map=destrieux_atlas,
        hemi="left",
        view="lateral",
        bg_map=fsaverage_sulcal,
        bg_on_data=True,
        colorbar=True,
        axes=ax[0],
    )
    plot_surf_roi(
        roi_map=destrieux_atlas,
        hemi="right",
        view="lateral",
        bg_map=fsaverage_sulcal,
        bg_on_data=True,
        colorbar=True,
        axes=ax[1],
    )

    fig.suptitle("surf_destrieux", fontsize=16)

    fig.savefig(output_file)

masker = SurfaceLabelsMasker(labels_img=destrieux_atlas).fit()
n_rois = masker.n_elements_

dict_for_df = _update_dict(
    dict_for_df,
    "Destrieux (2009; surface)",
    fetch_atlas_destrieux_2009,
    data,
    doc_dir,
    output_file=output_file,
    n_rois=n_rois,
)

_generate_markdown_file("deterministic_atlases.md", dict_for_df)

#
# PROBABILISTIC ATLASES
#

# dict to define fetching options for each atlas
probablistic_atlases = {
    "Allen (2011)": {"fn": fetch_atlas_allen_2011},
    "Craddock (2012)": {
        "fn": fetch_atlas_craddock_2012,
        "params": {"homogeneity": "spatial", "grp_mean": True},
    },
    "Difumo": {
        "fn": fetch_atlas_difumo,
        "params": {"dimension": 64, "resolution_mm": 2},
        "n_rois": [64, 128, 256, 512, 1024],
    },
    "Harvard-Oxford (cortical)": {
        "fn": fetch_atlas_harvard_oxford,
        "params": {"atlas_name": "cort-prob-1mm"},
    },
    "Harvard-Oxford (subcortical)": {
        "fn": fetch_atlas_harvard_oxford,
        "params": {"atlas_name": "sub-prob-1mm", "symmetric_split": False},
    },
    "Juelich": {
        "fn": fetch_atlas_juelich,
        "params": {"atlas_name": "prob-1mm"},
    },
    "MSDL": {"fn": fetch_atlas_msdl},
    "Smith (2009)": {
        "fn": fetch_atlas_smith_2009,
        "params": {"resting": False, "dimension": 10},
        "n_rois": [10, 20, 70],
    },
}

dict_for_df = {"name": [], "image": []}

for display_name, details in probablistic_atlases.items():
    fn = details["fn"]
    params = details.get("params", {})

    try:
        data = fn(**params)
    except (SSLError, MaxRetryError, SSLCertVerificationError):
        continue

    name = fn.__name__.replace("fetch_atlas_", "")

    extra_title = [f"{k}={v}" for k, v in params.items()]
    title = f"{fn.__name__}({', '.join(extra_title)})"

    params_str = ""
    for k, v in params.items():
        params_str += f"_{k}-{v}"
    output_file = output_dir / f"probablistic_atlas_{name}{params_str}.png"

    if GENERATE_FIG:
        plot_prob_atlas(
            data.maps,
            title=title,
            output_file=output_file,
            view_type="contours",
            figure=plt.figure(figsize=[11, 4]),
            **plot_config,
        )

    n_rois = details.get("n_rois")
    if n_rois is None:
        masker = NiftiMapsMasker(data.maps).fit()
        n_rois = masker.n_elements_

    dict_for_df = _update_dict(
        dict_for_df,
        display_name,
        fn,
        data,
        doc_dir,
        output_file=output_file,
        n_rois=n_rois,
    )

    if DEBUG:
        # probabilistic atlases take a long time to plot
        # so only do one in debug mode
        break

_generate_markdown_file("probabilistic_atlases.md", dict_for_df)

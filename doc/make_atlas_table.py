"""Generate markdown files with table summarizing information about atlases."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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
from nilearn.plotting import plot_prob_atlas, plot_roi, plot_surf_roi
from nilearn.surface import SurfaceImage


def _update_dict(dict_for_df, name, data, doc_dir, output_file, extra=None):
    """Update dictionary to use to create dataframe of atlases."""
    if extra is None:
        extra = []
    if isinstance(extra, str):
        extra = [extra]
    dict_for_df["name"].append(name + "<br>".join(extra))

    dict_for_df["template"].append(data.template)

    dict_for_df["description"].append(
        "{ref}`description " + f"<{name}_atlas>" + "`"
    )

    dict_for_df["image"].append(
        f"![name](../{output_file.relative_to(doc_dir)!s})"
    )

    return dict_for_df


def _generate_markdown_file(filename, dict_for_df):
    """Generate a markdown file with a table of atlases."""
    atlas_table = pd.DataFrame(dict_for_df)
    atlas_table.sort_values(by=["name"])

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
        atlas_table.to_markdown(buf=f, index=False)


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
    "aal": {
        "fn": fetch_atlas_aal,
    },
    "basc_multiscale_2015": {
        "fn": fetch_atlas_basc_multiscale_2015,
        "params": {"resolution": 20, "version": "sym"},
    },
    "destrieux_2009": {"fn": fetch_atlas_destrieux_2009},
    "harvard_oxford": {
        "fn": fetch_atlas_harvard_oxford,
        "params": {
            "atlas_name": "cort-maxprob-thr0-1mm",
            "symmetric_split": False,
        },
    },
    "harvard_oxford_2": {
        "fn": fetch_atlas_harvard_oxford,
        "extra": "(subcortical)",
        "params": {
            "atlas_name": "sub-maxprob-thr0-1mm",
            "symmetric_split": False,
        },
    },
    "juelich": {
        "fn": fetch_atlas_juelich,
        "params": {"atlas_name": "maxprob-thr0-1mm"},
    },
    "pauli_2017": {
        "fn": fetch_atlas_pauli_2017,
        "params": {"atlas_type": "deterministic"},
    },
    "schaefer_2018": {
        "fn": fetch_atlas_schaefer_2018,
    },
    "talairach": {"fn": fetch_atlas_talairach, "params": {"level_name": "ba"}},
    "yeo_2011": {
        "fn": fetch_atlas_yeo_2011,
        "params": {"n_networks": 17, "thickness": "thick"},
    },
}

dict_for_df = {"name": [], "template": [], "description": [], "image": []}

for details in deterministic_atlases.values():
    fn = details["fn"]
    params = details.get("params", {})

    data = fn(**params)

    name = fn.__name__.replace("fetch_atlas_", "")

    extra_title = [f"{k}={v}" for k, v in params.items()]
    title = f"{fn.__name__}({', '.join(extra_title)})"

    params_str = ""
    for k, v in params.items():
        params_str += f"_{k}-{v}"
    output_file = output_dir / f"deterministic_atlas_{name}{params_str}.png"

    plot_roi(
        data.maps,
        title=title,
        cmap=data.lut,
        output_file=output_file,
        figure=plt.figure(figsize=[8, 3]),
        **plot_config,
    )
    plt.close("all")

    dict_for_df = _update_dict(
        dict_for_df,
        name,
        data,
        doc_dir,
        output_file=output_file,
        extra=details.get("extra", None),
    )

"""
SURFACE DETERMINISTIC ATLASES

Surface atlases are bit different so we deal with them separately.
"""

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

fig, ax = plt.subplots(1, 2, figsize=[6, 3], subplot_kw={"projection": "3d"})

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

output_file = output_dir / f"deterministic_atlas_{name}.png"
fig.savefig(output_file)

dict_for_df = _update_dict(
    dict_for_df, "destrieux_2009", data, doc_dir, output_file=output_file
)

_generate_markdown_file("deterministic_atlases.md", dict_for_df)

"""
PROBABILISTIC ATLASES
"""

# dict to define fetching options for each atlas
probablistic_atlases = {
    "allen_2011": {"fn": fetch_atlas_allen_2011},
    "craddock_2012": {
        "fn": fetch_atlas_craddock_2012,
        "params": {"homogeneity": "spatial", "grp_mean": True},
    },
    "difumo": {
        "fn": fetch_atlas_difumo,
        "params": {"dimension": 64, "resolution_mm": 2},
    },
    "harvard_oxford": {
        "fn": fetch_atlas_harvard_oxford,
        "params": {"atlas_name": "cort-prob-1mm"},
    },
    "harvard_oxford_2": {
        "fn": fetch_atlas_harvard_oxford,
        "params": {"atlas_name": "sub-prob-1mm", "symmetric_split": False},
    },
    "juelich": {
        "fn": fetch_atlas_juelich,
        "params": {"atlas_name": "prob-1mm"},
    },
    "msdl": {"fn": fetch_atlas_msdl},
    "smith_2009": {
        "fn": fetch_atlas_smith_2009,
        "params": {"resting": False, "dimension": 20},
    },
}

dict_for_df = {"name": [], "template": [], "description": [], "image": []}

for details in probablistic_atlases.values():
    fn = details["fn"]
    params = details.get("params", {})

    data = fn(**params)

    name = fn.__name__.replace("fetch_atlas_", "")

    extra_title = [f"{k}={v}" for k, v in params.items()]
    title = f"{fn.__name__}({', '.join(extra_title)})"

    params_str = ""
    for k, v in params.items():
        params_str += f"_{k}-{v}"
    output_file = output_dir / f"probablistic_atlas_{name}{params_str}.png"
    plot_prob_atlas(
        data.maps,
        title=title,
        output_file=output_file,
        view_type="contours",
        **plot_config,
    )

    dict_for_df = _update_dict(
        dict_for_df, name, data, doc_dir, output_file=output_file
    )

_generate_markdown_file("probabilistic_atlases.md", dict_for_df)

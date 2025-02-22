"""Generate markdown files with table summarizing information about atlases."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from nilearn.datasets import (
    fetch_atlas_aal,
    fetch_atlas_allen_2011,
    fetch_atlas_basc_multiscale_2015,
    fetch_atlas_destrieux_2009,
    fetch_atlas_harvard_oxford,
    fetch_atlas_juelich,
    fetch_atlas_pauli_2017,
    fetch_atlas_schaefer_2018,
    fetch_atlas_smith_2009,
    fetch_atlas_surf_destrieux,
    fetch_atlas_talairach,
    fetch_atlas_yeo_2011,
    load_fsaverage,
    load_fsaverage_data,
)
from nilearn.plotting import plot_prob_atlas, plot_roi, plot_surf_roi, show
from nilearn.surface import SurfaceImage

doc_dir = Path(__file__).parent
output_dir = Path(__file__).parent / "images"


def update_dict(dict_for_df, name, data, doc_dir):
    dict_for_df["name"].append(name)
    dict_for_df["template"].append(data.template)
    dict_for_df["description"].append(
        "{ref}`description " + f"<{name}_atlas>" + "`"
    )
    dict_for_df["image"].append(
        f"![name](../{output_file.relative_to(doc_dir)!s})"
    )
    return dict_for_df


plot_config = {
    "draw_cross": False,
    "colorbar": False,
    "display_mode": "ortho",
    "cut_coords": [0, 0, 0],
    "figure": plt.figure(figsize=[8, 3]),
}

"""
VOLUME DETERMINISTIC ATLASES
"""

deterministic_atlases = {
    "aal": {"fn": fetch_atlas_aal, "params": {}},
    "basc_multiscale_2015": {
        "fn": fetch_atlas_basc_multiscale_2015,
        "params": {"resolution": 20},
    },
    "destrieux_2009": {"fn": fetch_atlas_destrieux_2009, "params": {}},
    "harvard_oxford": {
        "fn": fetch_atlas_harvard_oxford,
        "params": {"atlas_name": "cort-maxprob-thr0-1mm"},
    },
    "harvard_oxford_2": {
        "fn": fetch_atlas_harvard_oxford,
        "params": {"atlas_name": "sub-maxprob-thr0-1mm"},
    },
    "juelich": {
        "fn": fetch_atlas_juelich,
        "params": {"atlas_name": "maxprob-thr0-1mm"},
    },
    "pauli_2017": {
        "fn": fetch_atlas_pauli_2017,
        "params": {"atlas_type": "deterministic"},
    },
    "schaefer_2018": {"fn": fetch_atlas_schaefer_2018, "params": {}},
    "talairach": {"fn": fetch_atlas_talairach, "params": {"level_name": "ba"}},
    "yeo_2011": {
        "fn": fetch_atlas_yeo_2011,
        "params": {"n_networks": 17, "thickness": "thick"},
    },
}

dict_for_df = {"name": [], "template": [], "description": [], "image": []}

for details in deterministic_atlases.values():
    fn = details["fn"]
    params = details["params"]

    data = fn(**params)

    name = fn.__name__.replace("fetch_atlas_", "")

    extra_title = [f"{k}={v}" for k, v in params.items()]
    title = f"{fn.__name__}({', '.join(extra_title)})"

    fig = plot_roi(
        data.maps, title=title, cmap=data.lut, colorbar=False, **plot_config
    )

    details = ""
    for k, v in params.items():
        details += f"_{k}-{v}"
    output_file = output_dir / f"deterministic_atlas_{name}{details}.png"
    fig.savefig(output_file)

    dict_for_df = update_dict(dict_for_df, name, data, doc_dir)

show()

"""
SURFACE DETERMINISTIC ATLASES
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

fig = plot_surf_roi(
    roi_map=destrieux_atlas,
    hemi="left",
    view="lateral",
    bg_map=fsaverage_sulcal,
    bg_on_data=True,
    title=name,
    colorbar=False,
    figure=plt.figure(figsize=[8, 3]),
)

output_file = output_dir / f"deterministic_atlas_{name}.png"
fig.savefig(output_file)

dict_for_df = update_dict(dict_for_df, "destrieux_2009", data, doc_dir)

show()

deterministic_atlases_df = pd.DataFrame(dict_for_df)
deterministic_atlases_df.to_markdown(
    Path(__file__).parent / "modules" / "deterministic_atlases.md", index=False
)

"""
PROBABILISTIC ATLASES
"""

#    fetch_atlas_craddock_2012
#    fetch_atlas_difumo
#    fetch_atlas_msdl

probablistic_atlases = {
    "allen_2011": {"fn": fetch_atlas_allen_2011, "params": {}, "key": "rsn28"},
    "harvard_oxford": {
        "fn": fetch_atlas_harvard_oxford,
        "params": {"atlas_name": "cort-prob-1mm"},
    },
    "harvard_oxford_2": {
        "fn": fetch_atlas_harvard_oxford,
        "params": {"atlas_name": "sub-prob-1mm"},
    },
    "juelich": {
        "fn": fetch_atlas_juelich,
        "params": {"atlas_name": "prob-1mm"},
    },
    "smith_2009": {
        "fn": fetch_atlas_smith_2009,
        "params": {"resting": False, "dimension": 20},
    },
}

dict_for_df = {"name": [], "template": [], "description": [], "image": []}

for details in probablistic_atlases.values():
    fn = details["fn"]
    params = details["params"]

    data = fn(**params)

    name = fn.__name__.replace("fetch_atlas_", "")

    extra_title = [f"{k}={v}" for k, v in params.items()]
    title = f"{fn.__name__}({', '.join(extra_title)})"

    image = data[details.get("key", "maps")]

    fig = plot_prob_atlas(data.maps, title=title, colorbar=True, **plot_config)

    details = ""
    for k, v in params.items():
        details += f"_{k}-{v}"
    output_file = output_dir / f"probablistic_atlas_{name}{details}.png"
    fig.savefig(output_file)

    dict_for_df = update_dict(dict_for_df, name, data, doc_dir)

show()


probablistic_atlases_df = pd.DataFrame(dict_for_df)
probablistic_atlases_df.to_markdown(
    Path(__file__).parent / "modules" / "probabilistic_atlases.md", index=False
)

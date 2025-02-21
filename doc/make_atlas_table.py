"""Generate markdown files with table summarizing information about atlases."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from nilearn.datasets import (
    fetch_atlas_aal,
    fetch_atlas_basc_multiscale_2015,
    fetch_atlas_destrieux_2009,
    fetch_atlas_harvard_oxford,
    fetch_atlas_juelich,
    fetch_atlas_pauli_2017,
    fetch_atlas_schaefer_2018,
    fetch_atlas_talairach,
    fetch_atlas_yeo_2011,
)
from nilearn.plotting import plot_roi, show

doc_dir = Path(__file__).parent
output_dir = Path(__file__).parent / "images"

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

dict_for_df = {"name": [], "template": [], "image": []}

for details in deterministic_atlases.values():
    fn = details["fn"]
    params = details["params"]

    data = fn(**params)

    name = fn.__name__.replace("fetch_atlas_", "")

    dict_for_df["name"].append(name)
    dict_for_df["template"].append(data.template)

    extra_title = [f"{k}={v}" for k, v in params.items()]
    title = f"{fn.__name__}({', '.join(extra_title)})"
    fig = plot_roi(
        data.maps,
        title=title,
        draw_cross=False,
        display_mode="ortho",
        cut_coords=[0, 0, 0],
        figure=plt.figure(figsize=[8, 3]),
    )

    details = ""
    for k, v in params.items():
        details += f"_{k}-{v}"
    output_file = output_dir / f"{name}{details}.png"
    fig.savefig(output_file)

    dict_for_df["image"].append(
        f"![name](../{output_file.relative_to(doc_dir)!s})"
    )

show()

deterministic_atlases_df = pd.DataFrame(dict_for_df)
deterministic_atlases_df.to_markdown(
    Path(__file__).parent / "modules" / "tables.md", index=False
)

#    fetch_atlas_surf_destrieux : {},

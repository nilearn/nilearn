r"""

Using brain templates from TemplateFlow
=======================================

In this example, we see how to fetch and use brain templates from
`TemplateFlow <https://https://www.templateflow.org/>`. We see how to fetch and use the
Desikan-Killiany (DK) atlas in order to make surface plots.
"""

# %%
# Fetch DK Surface template
# -------------------------
#
# We fetch the DK atlas from TemplateFlow. The template is available at multiple
# resolutions (number of vertices per hemisphere). Here, we fetch the 10k
# resolution.

import templateflow.api as tflow

from nilearn import surface
from nilearn.surface import SurfaceImage

mesh = {}
data = {}

template = "fsaverage"
density = "10k"  # number of vertices per hemisphere

for hemi in ["left", "right"]:
    mesh[hemi] = tflow.get(
        template,
        extension="surf.gii",
        suffix="pial",
        density=density,
        hemi=hemi[0].upper(),
    )

    desc = "curated" if density == "164k" else None
    roi_data = surface.load_surf_data(
        tflow.get(
            template,
            atlas="Desikan2006",
            density=density,
            hemi=hemi[0].upper(),
            extension="label.gii",
            desc=desc,
        )
    )
    if density == "164k":
        roi_data[roi_data < 0] = 0

    data[hemi] = roi_data

desikan = SurfaceImage(mesh=mesh, data=data)


# %%
# Plotting the DK atlas
# ---------------------
# To plot the DK atlas, we also fetch the lookup table (LUT) from TemplateFlow, to map
# region indices to the original region names and colors. We also fetch the background
# sulcal depth map to add more detail to the surface plot. To do this we map the density
# to the corresponding fsaverage template name.


import pandas as pd

from nilearn.datasets import load_fsaverage_data
from nilearn.plotting import plot_surf_roi, show

fs_density = {
    "3k": "fsaverage4",
    "10k": "fsaverage5",
    "41k": "fsaverage6",
    "164k": "fsaverage",
}

lut = tflow.get(
    template,
    atlas="Desikan2006",
    suffix="dseg",
    extension="tsv",
)

bg_data = load_fsaverage_data(mesh=fs_density[density], mesh_type="pial")
plot_surf_roi(
    roi_map=desikan,
    cmap=pd.read_csv(lut, sep="\t"),
    bg_map=bg_data,
    bg_on_data=True,
)

show()

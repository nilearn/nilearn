r"""

Using brain templates from TemplateFlow
=======================================

In this example, we see how to fetch and use brain templates from
`TemplateFlow <https://https://www.templateflow.org/>`. We see how to fetch and use the
Desikan-Killiany (DK) atlas in order to make surface plots.
"""

# %%
# DK Surface template
# -------------------
#
# get surface template from templateflow

import pandas as pd
from templateflow import api as tflow

from nilearn.plotting import plot_surf_roi, show
from nilearn.surface import SurfaceImage

template = "fsaverage"
# number of vertices per hemisphere
density = "10k"

mesh = {}
data = {}

for hemi in ["left", "right"]:
    mesh[hemi] = tflow.get(
        template,
        extension="surf.gii",
        suffix="pial",
        density=density,
        hemi=hemi[0].upper(),
    )

    data[hemi] = tflow.get(
        template,
        atlas="Desikan2006",
        density=density,
        hemi=hemi[0].upper(),
        extension="label.gii",
    )

lut = tflow.get(
    template,
    atlas="Desikan2006",
    suffix="dseg",
    extension="tsv",
)

desikan = SurfaceImage(mesh=mesh, data=data)

plot_surf_roi(roi_map=desikan, cmap=pd.read_csv(lut, sep="\t"))

show()

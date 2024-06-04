"""Drafting."""

import numpy as np
from rich import inspect

from nilearn.experimental import surface

labels_img, label_names = surface.fetch_destrieux()

label = 62

for part in labels_img.data.parts:
    tmp = labels_img.data.parts[part] == label
    tmp = np.zeros(labels_img.data.parts[part].shape, dtype=np.int8)
    tmp[labels_img.data.parts[part] == label] = 1
    labels_img.data.parts[part] = tmp

inspect(labels_img.data)

img = surface.fetch_nki()[0]
inspect(img)

masker = surface.SurfaceMasker(labels_img)
masked_data = masker.fit_transform(img)
report = masker.generate_report()
report.save_as_html("surface_masker_nki.html")

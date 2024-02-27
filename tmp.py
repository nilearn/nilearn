"""Test atlases masking."""

import numpy as np
from rich import print

from nilearn import datasets
from nilearn.datasets import (
    fetch_atlas_aal,
    fetch_atlas_basc_multiscale_2015,
    fetch_atlas_destrieux_2009,
    fetch_atlas_harvard_oxford,
    fetch_atlas_juelich,
    fetch_atlas_schaefer_2018,
    fetch_atlas_talairach,
    fetch_atlas_yeo_2011,
)
from nilearn.maskers import NiftiLabelsMasker

functions = {
    fetch_atlas_destrieux_2009: None,
    fetch_atlas_harvard_oxford: ["cort-maxprob-thr0-1mm"],
    fetch_atlas_juelich: ["maxprob-thr0-1mm"],
    fetch_atlas_talairach: ["gyrus"],
    fetch_atlas_aal: None,
    fetch_atlas_schaefer_2018: None,
    fetch_atlas_basc_multiscale_2015: [None, None, True, 1, 64, "sym"],
    fetch_atlas_yeo_2011: None,
}


def main():
    """Test atlases masking."""
    dataset = datasets.fetch_development_fmri(n_subjects=1)
    func_filename = dataset.func[0]

    for f, args in functions.items():
        print(f)

        atlas = f(*args) if args else f()

        if "labels" in atlas:
            labels = atlas.labels
        elif "rsn_indices" in atlas:
            labels = atlas.rsn_indices
        elif f == fetch_atlas_basc_multiscale_2015:
            labels = range(64)
        elif f == fetch_atlas_yeo_2011:
            labels = range(17)

        if f == fetch_atlas_schaefer_2018:
            labels = np.insert(labels, 0, "Background")

        labels_img = (
            atlas["thick_17"] if f == fetch_atlas_yeo_2011 else atlas.maps
        )

        masker = NiftiLabelsMasker(
            labels_img=labels_img,
            labels=labels,
            standardize="zscore_sample",
        )

        masker.fit_transform(func_filename)

        print(f"{masker.labels=}")
        print(f"{masker.labels_=}")
        print(f"{masker.region_ids_=}")
        print(f"{masker.region_names_=}")


if __name__ == "__main__":
    exit(main())

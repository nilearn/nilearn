"""Test generating NiftiLabelsMasker report.

This is done with ALL Nilearn deterministic atlases:
- with no labels
- with labels
- with look up table

Also checks some template flow atlases
to ensure our maskers can work with external resources.
"""

import sys
from pathlib import Path

import numpy as np
from nibabel import Nifti1Image
from rich import print
from templateflow import api as tflow

from nilearn import datasets
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
from nilearn.maskers import NiftiLabelsMasker

functions = {
    fetch_atlas_aal: None,
    fetch_atlas_basc_multiscale_2015: [None, None, True, 1, 64, "sym"],
    fetch_atlas_destrieux_2009: None,
    fetch_atlas_harvard_oxford: ["cort-maxprob-thr0-1mm"],
    fetch_atlas_juelich: ["maxprob-thr0-1mm"],
    fetch_atlas_pauli_2017: ["deterministic", None, 1],
    fetch_atlas_schaefer_2018: None,
    fetch_atlas_talairach: ["gyrus"],
    fetch_atlas_yeo_2011: [None, None, True, 1, 17],
}


def _check_atlas(
    labels_img: str | Path | Nifti1Image,
    atlas_name: str,
    stat_map: str | Path,
    output_dir: Path,
    background_label=0,
    **kwargs,
):
    """Check atlas is usable by nilearn label masker.

    - construct masker instance
    - use fit_transform
    - generate report and save it.
    """
    print(kwargs.keys())

    masker = NiftiLabelsMasker(
        labels_img=labels_img, background_label=background_label, **kwargs
    )

    masker.fit_transform(stat_map)
    try:
        masker.fit_transform(stat_map)
    except Exception as e:
        print("\n[red]" + "Could not fit atlas".upper() + f"\n{e!r}")
        return False

    try:
        report = masker.generate_report()
        report.save_as_html(
            output_dir / f"report_{atlas_name}{_suffix(**kwargs)}.html"
        )
    except Exception as e:
        print("\n[red]" + "Could not generate report".upper() + f"\n{e!r}")
        return False

    return True


def _suffix(**kwargs):
    """Return first key of kwargs if it has any."""
    return f"_{next(iter(kwargs.keys()))}" if kwargs.keys() else ""


def main():
    """Test atlases masking."""
    stat_map = datasets.load_sample_motor_activation_image()

    output_dir = Path(__file__).parent / "tmp"
    output_dir.mkdir(exist_ok=True, parents=True)

    failing_atlas = []

    # TEMPLATE FLOW ATLASES

    labels_img = tflow.get(
        "MNI152NLin2009cAsym",
        desc="100Parcels7Networks",
        atlas="Schaefer2018",
        resolution="01",
        suffix="dseg",
        extension="nii.gz",
    )
    lut = tflow.get(
        "MNI152NLin2009cAsym",
        desc="100Parcels7Networks",
        atlas="Schaefer2018",
        suffix="dseg",
        extension="tsv",
    )

    # check that the datalad get returns something
    assert labels_img
    assert lut

    atlas_name = "tflow_Schaefer2018_100Parcels7Networks"

    print(f"\n{atlas_name}")

    for kwargs in [{}, {"lut": lut}]:
        if not _check_atlas(
            labels_img,
            atlas_name,
            stat_map,
            output_dir,
            background_label=np.int16(0),
            **kwargs,
        ):
            failing_atlas.extend([f"{atlas_name}_{_suffix(**kwargs)}"])

    # NILEARN ATLASES

    for fn, args in functions.items():
        atlas_name = fn.__name__

        print(f"\n{atlas_name}")

        try:
            atlas = fn(*args) if args else fn()
        except Exception as e:
            print("\n[red]" + "Could not get atlas".upper() + f"\n{e!r}")
            continue

        labels = atlas.labels
        labels_img = atlas.maps
        lut = atlas.lut

        for kwargs in [{}, {"labels": labels}, {"lut": lut}]:
            if not _check_atlas(
                labels_img,
                atlas_name,
                stat_map,
                output_dir,
                **kwargs,
            ):
                failing_atlas.extend([f"{atlas_name}_{_suffix(**kwargs)}"])

    if failing_atlas:
        raise RuntimeError(
            f"The following atlases could not be used: {failing_atlas}."
        )


if __name__ == "__main__":
    sys.exit(main())

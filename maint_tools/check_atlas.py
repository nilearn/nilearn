"""Test generating NiftiLabelsMasker report.

This is done with ALL Nilearn deterministic atlases:
- with no labels
- with labels
- with look up table
- after just fit() and fit_transform on a given image

Also checks some template flow atlases
to ensure our maskers can work with external resources.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from nibabel import Nifti1Image
from rich import print
from templateflow import api as tflow

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
    load_fsaverage,
    load_sample_motor_activation_image,
)
from nilearn.maskers import NiftiLabelsMasker, SurfaceLabelsMasker
from nilearn.surface import SurfaceImage
from nilearn.typing import NiimgLike

functions = {
    fetch_atlas_aal: None,
    fetch_atlas_basc_multiscale_2015: [None, None, True, 1, 64, "sym"],
    fetch_atlas_destrieux_2009: None,
    fetch_atlas_harvard_oxford: ["cort-maxprob-thr0-1mm"],
    fetch_atlas_juelich: ["maxprob-thr0-1mm"],
    fetch_atlas_pauli_2017: ["deterministic", None, 1],
    fetch_atlas_schaefer_2018: None,
    fetch_atlas_talairach: ["gyrus"],
    fetch_atlas_yeo_2011: [None, None, True, 1, 7],
}


def _check_atlas(
    labels_img: str | Path | Nifti1Image | SurfaceImage,
    atlas_name: str,
    stat_map: str | Path | Nifti1Image | SurfaceImage,
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

    if isinstance(stat_map, NiimgLike):
        masker = NiftiLabelsMasker(
            labels_img=labels_img, background_label=background_label, **kwargs
        )
    else:
        masker = SurfaceLabelsMasker(
            labels_img=labels_img, background_label=background_label, **kwargs
        )

    try:
        masker.fit()
        if "lut" in kwargs and isinstance(kwargs["lut"], pd.DataFrame):
            assert list(masker.lut.columns) == list(masker.lut_.columns)

    except Exception as e:
        print("\n[red]" + "Could not fit atlas".upper() + f"\n{e!r}")
        return False

    try:
        report = masker.generate_report()
        report.save_as_html(
            output_dir
            / f"report_{atlas_name}{_suffix(**kwargs)}_fit_only.html"
        )

    except Exception as e:
        print("\n[red]" + "Could not generate report".upper() + f"\n{e!r}")
        return False

    try:
        signal = masker.fit_transform(stat_map)

        if isinstance(stat_map, NiimgLike):
            n_regions = signal.shape[0]
        else:
            n_regions = signal.shape[1]
        assert n_regions == len(masker.region_names_), (
            f"{n_regions=} == {len(masker.region_names_)=}"
        )
        assert len(masker.region_ids_) == len(masker.labels_), (
            f"{len(masker.region_ids_)=} == {len(masker.labels_)=}"
        )

    except Exception as e:
        print("\n[red]" + "Could not fit transform atlas".upper() + f"\n{e!r}")
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
    stat_map = load_sample_motor_activation_image()

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

    # NILEARN V0LUME ATLASES

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

    # NILEARN SURFACE ATLASES

    fsaverage = load_fsaverage("fsaverage5")
    destrieux = fetch_atlas_surf_destrieux()
    atlas_name = "surf_destrieux"

    labels_img = SurfaceImage(
        mesh=fsaverage["pial"],
        data={
            "left": destrieux.map_left,
            "right": destrieux.map_right,
        },
    )

    labels = destrieux.labels
    lut = destrieux.lut

    stat_map = SurfaceImage(
        mesh=fsaverage["pial"],
        data={
            "left": np.ones((fsaverage["pial"].parts["left"].n_vertices, 5)),
            "right": -1
            * np.ones((fsaverage["pial"].parts["right"].n_vertices, 5)),
        },
    )

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

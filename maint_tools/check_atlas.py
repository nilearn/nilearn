"""Test generating NiftiLabelsMasker report.

This is done with ALL Nilearn deterministic atlases:
- with no labels
- with labels
- with look up table
"""

import sys
from pathlib import Path

from rich import print

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

failing_atlas = []


def main():
    """Test atlases masking."""
    dataset = datasets.load_sample_motor_activation_image()

    output_dir = Path(__file__).parent / "tmp"
    output_dir.mkdir(exist_ok=True, parents=True)

    for fn, args in functions.items():
        print()
        print(fn.__name__)

        try:
            atlas = fn(*args) if args else fn()
        except Exception as e:
            print("\n[red]" + "Could not get atlas".upper() + f"\n{e!r}")
            continue

        labels = atlas.labels
        labels_img = atlas.maps
        lut = atlas.lut

        for kwargs in [{}, {"labels": labels}, {"lut": lut}]:
            print(kwargs.keys())

            masker = NiftiLabelsMasker(labels_img=labels_img, **kwargs)

            try:
                masker.fit_transform(dataset)
            except Exception as e:
                print("\n[red]" + "Could not fit atlas".upper() + f"\n{e!r}")
                failing_atlas.append(fn)
                continue

            try:
                report = masker.generate_report()
                suffix = (
                    f"_{next(iter(kwargs.keys()))}" if kwargs.keys() else ""
                )
                report.save_as_html(
                    output_dir / f"report_{fn.__name__}{suffix}.html"
                )
            except Exception as e:
                print(
                    "\n[red]"
                    + "Could not generate report".upper()
                    + f"\n{e!r}"
                )
                failing_atlas.append(fn)

    print(failing_atlas)


if __name__ == "__main__":
    sys.exit(main())

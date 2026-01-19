"""Marimo notebook to visualize reports in a notebook."""

import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Test rendering of masker reports in notebook

    This notebook is mostly for Nilearn developers to ensure
    that reports from the maskers look fine in a notebook.
    """)


@app.cell
def _():
    import time

    import marimo as mo
    from reporter_visual_inspection_suite import (
        cli_parser,
        report_nifti_labels_masker,
        report_nifti_maps_masker,
        report_nifti_masker,
        report_sphere_masker,
        report_surface_label_masker,
        report_surface_maps_masker,
        report_surface_masker,
    )

    return (
        cli_parser,
        mo,
        report_nifti_labels_masker,
        report_nifti_maps_masker,
        report_nifti_masker,
        report_sphere_masker,
        report_surface_label_masker,
        report_surface_maps_masker,
        report_surface_masker,
        time,
    )


@app.cell
def _(cli_parser):
    args = cli_parser().parse_args()
    BUILD_TYPE = args.build_type
    if isinstance(BUILD_TYPE, list):
        BUILD_TYPE = BUILD_TYPE[0]
    # BUILD_TYPE = "full"
    print(f"{BUILD_TYPE=}")
    return (BUILD_TYPE,)


@app.cell
def _(BUILD_TYPE, mo):
    mo.md(rf"""
    Generating masker reports for a build: {BUILD_TYPE}
    """)


@app.cell
def _(time):
    t0 = time.time()
    return (t0,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Nifti maskers
    """)


@app.cell
def _(BUILD_TYPE, report_nifti_masker):
    nifti_masker_reports = report_nifti_masker(build_type=BUILD_TYPE)
    return (nifti_masker_reports,)


@app.cell
def _(nifti_masker_reports):
    nifti_masker_reports[0]


@app.cell
def _(nifti_masker_reports):
    nifti_masker_reports[1]


@app.cell
def _(BUILD_TYPE, report_nifti_maps_masker):
    nifti_maps_masker_reports = report_nifti_maps_masker(build_type=BUILD_TYPE)
    return (nifti_maps_masker_reports,)


@app.cell
def _(nifti_maps_masker_reports):
    nifti_maps_masker_reports[0]


@app.cell
def _(nifti_maps_masker_reports):
    nifti_maps_masker_reports[1]


@app.cell
def _(BUILD_TYPE, report_nifti_labels_masker):
    nifti_labels_masker_reports = report_nifti_labels_masker(
        build_type=BUILD_TYPE
    )
    return (nifti_labels_masker_reports,)


@app.cell
def _(nifti_labels_masker_reports):
    nifti_labels_masker_reports[0]


@app.cell
def _(nifti_labels_masker_reports):
    nifti_labels_masker_reports[1]


@app.cell
def _(BUILD_TYPE, report_sphere_masker):
    sphere_masker_reports = report_sphere_masker(build_type=BUILD_TYPE)
    return (sphere_masker_reports,)


@app.cell
def _(sphere_masker_reports):
    sphere_masker_reports[0]


@app.cell
def _(sphere_masker_reports):
    sphere_masker_reports[1]


@app.cell
def _(mo):
    mo.md(r"""
    ## Surface maskers
    """)


@app.cell
def _(BUILD_TYPE, report_surface_masker):
    surface_masker_reports = report_surface_masker(build_type=BUILD_TYPE)
    return (surface_masker_reports,)


@app.cell
def _(surface_masker_reports):
    surface_masker_reports[0]


@app.cell
def _(surface_masker_reports):
    surface_masker_reports[1]


@app.cell
def _(BUILD_TYPE, report_surface_maps_masker):
    surface_maps_masker_reports = report_surface_maps_masker(
        build_type=BUILD_TYPE
    )
    return (surface_maps_masker_reports,)


@app.cell
def _(surface_maps_masker_reports):
    surface_maps_masker_reports[0]


@app.cell
def _(surface_maps_masker_reports):
    surface_maps_masker_reports[1]


@app.cell
def _(BUILD_TYPE, surface_maps_masker_reports):
    if BUILD_TYPE == "full":
        surface_maps_masker_reports[2]


@app.cell
def _(BUILD_TYPE, report_surface_label_masker):
    surface_label_masker_reports = report_surface_label_masker(
        build_type=BUILD_TYPE
    )
    return (surface_label_masker_reports,)


@app.cell
def _(surface_label_masker_reports):
    surface_label_masker_reports[0]


@app.cell
def _(surface_label_masker_reports):
    surface_label_masker_reports[1]


@app.cell
def _(t0, time):
    t1 = time.time()
    print(f"\nTook: {t1 - t0:0.2f} seconds\n")


if __name__ == "__main__":
    app.run()

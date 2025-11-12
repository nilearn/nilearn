"""Marimo notebook to visualize reports in a notebook."""

import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Test rendering of maskers in notebook
    """)


@app.cell
def _():
    BUILD_TYPE = "full"
    return (BUILD_TYPE,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This notebook allows us to visualize
    how the different reports look in a notebook.
    """)


@app.cell
def _():
    from reporter_visual_inspection_suite import (
        report_flm_adhd_dmn,
        report_flm_bids_features,
        report_flm_fiac,
        report_nifti_labels_masker,
        report_nifti_maps_masker,
        report_nifti_masker,
        report_slm_oasis,
    )

    return (
        report_flm_adhd_dmn,
        report_flm_bids_features,
        report_flm_fiac,
        report_nifti_labels_masker,
        report_nifti_maps_masker,
        report_nifti_masker,
        report_slm_oasis,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Maskers reports
    """)


@app.cell
def _(BUILD_TYPE, report_nifti_masker):
    nifti_masker_reports = report_nifti_masker(build_type=BUILD_TYPE)
    return (nifti_masker_reports,)


@app.cell
def _(nifti_masker_reports):
    nifti_masker_reports


@app.cell
def _(BUILD_TYPE, report_nifti_maps_masker):
    nifti_maps_masker_reports = report_nifti_maps_masker(build_type=BUILD_TYPE)
    return (nifti_maps_masker_reports,)


@app.cell
def _(nifti_maps_masker_reports):
    nifti_maps_masker_reports


@app.cell
def _(BUILD_TYPE, report_nifti_labels_masker):
    nifti_labels_masker_reports = report_nifti_labels_masker(
        build_type=BUILD_TYPE
    )
    return (nifti_labels_masker_reports,)


@app.cell
def _(nifti_labels_masker_reports):
    nifti_labels_masker_reports


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GLM reports
    """)


@app.cell
def _(BUILD_TYPE, report_flm_bids_features):
    flm_bids_report = report_flm_bids_features(build_type=BUILD_TYPE)
    return (flm_bids_report,)


@app.cell
def _(flm_bids_report):
    flm_bids_report


@app.cell
def _(BUILD_TYPE, report_flm_fiac):
    flm_fiac_report = report_flm_fiac(build_type=BUILD_TYPE)
    return (flm_fiac_report,)


@app.cell
def _(flm_fiac_report):
    flm_fiac_report


@app.cell
def _(BUILD_TYPE, report_flm_adhd_dmn):
    masker_report, flm_adhd_report = report_flm_adhd_dmn(build_type=BUILD_TYPE)
    return flm_adhd_report, masker_report


@app.cell
def _(masker_report):
    masker_report


@app.cell
def _(flm_adhd_report):
    flm_adhd_report


@app.cell
def _(BUILD_TYPE, report_slm_oasis):
    slm_reports = report_slm_oasis(build_type=BUILD_TYPE)
    return (slm_reports,)


@app.cell
def _(slm_reports):
    slm_reports


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

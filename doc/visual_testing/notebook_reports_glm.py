"""Marimo notebook to visualize reports in a notebook."""

import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Test rendering of GLM reports in notebook

    This notebook is mostly for Nilearn developers
    to ensure that reports from the GLM
    look fine in a notebook.
    """)


@app.cell
def _():
    import time

    import marimo as mo
    from reporter_visual_inspection_suite import (
        cli_parser,
        report_flm_adhd_dmn,
        report_flm_bids_features,
        report_flm_fiac,
        report_slm_oasis,
        report_surface_flm,
        report_surface_slm,
    )

    return (
        cli_parser,
        mo,
        report_flm_adhd_dmn,
        report_flm_bids_features,
        report_flm_fiac,
        report_slm_oasis,
        report_surface_flm,
        report_surface_slm,
        time,
    )


@app.cell
def _(cli_parser):
    args = cli_parser().parse_args()
    BUILD_TYPE = args.build_type
    if isinstance(BUILD_TYPE, list):
        BUILD_TYPE = BUILD_TYPE[0]
    # BUILD_TYPE="full"
    print(f"{BUILD_TYPE=}")
    return (BUILD_TYPE,)


@app.cell
def _(BUILD_TYPE, mo):
    mo.md(rf"""
    Generating GLM reports for a build: {BUILD_TYPE}
    """)


@app.cell
def _(time):
    t0 = time.time()
    return (t0,)


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
    flm_adhd_report = report_flm_adhd_dmn(build_type=BUILD_TYPE)
    return (flm_adhd_report,)


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
def _(BUILD_TYPE, report_surface_flm):
    surface_flm = report_surface_flm(build_type=BUILD_TYPE)
    return (surface_flm,)


@app.cell
def _(surface_flm):
    surface_flm[0]


@app.cell
def _(surface_flm):
    surface_flm[0].save_as_html("tmp.html")


@app.cell
def _(surface_flm):
    surface_flm[1]


@app.cell
def _(report_surface_slm):
    surface_slm = report_surface_slm()
    return (surface_slm,)


@app.cell
def _(surface_slm):
    surface_slm


@app.cell
def _(t0, time):
    t1 = time.time()
    print(f"\nTook: {t1 - t0:0.2f} seconds\n")


if __name__ == "__main__":
    app.run()

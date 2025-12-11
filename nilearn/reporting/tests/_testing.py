"""Testing utilities for reporting."""

from pathlib import Path

import pytest

from nilearn._utils.helpers import is_matplotlib_installed
from nilearn._utils.html_document import WIDTH_DEFAULT
from nilearn.reporting import HTMLReport
from nilearn.reporting.html_report import MISSING_ENGINE_MSG


def check_report(
    estimator,
    view=False,
    pth: Path | None = None,
    extend_includes: list[str] | None = None,
    extend_excludes: list[str] | None = None,
    **kwargs,
) -> HTMLReport:
    """Generate a report and run generic checks on it.

    Parameters
    ----------
    model : any estimator with a generate_report method
        Model that generated the report

    view: bool, default=False
        if True the report is open in browser
        only used for debugging locally

    pth: Path or None, default=None
        Where to save the report

    extend_includes : Iterable[str] | None, default=None
        The function will check
        for the presence in the report
        of each string in this iterable.

    extend_includes : Iterable[str] | None, default=None
        The function will check
        for the absence in the report
        of each string in this iterable.

    kwargs : dict
        Extra-parameters to pass to generate_report.
    """
    report = estimator.generate_report(**kwargs)

    assert isinstance(report, HTMLReport)

    # catches & raises UnicodeEncodeError in HTMLDocument.get_iframe()
    # in case certain unicode characters are mishandled,
    # like the greek alpha symbol.
    report.get_iframe()

    # resize width and height
    report.resize(1200, 800)
    assert report.width == 1200
    assert report.height == 800

    # invalid values fall back on default dimensions
    with pytest.warns(UserWarning, match="Using default instead"):
        report.width = "foo"
    assert report.width == WIDTH_DEFAULT

    assert report._repr_html_() == report.body

    # only for debugging
    if view:
        report.open_in_browser()

    if pth:
        # save to disk
        # useful for visual inspection
        # for manual checks or in case of test failure
        report.save_as_html(pth / "tmp.html")
        assert (pth / "tmp.html").exists()

    includes = []
    excludes = []

    if is_matplotlib_installed():
        excludes.extend(
            [MISSING_ENGINE_MSG, 'grey">No plotting engine found</p>']
        )
    else:
        includes.extend(
            [
                'id="warnings"',
                MISSING_ENGINE_MSG,
                'grey">No plotting engine found</p>',
            ]
        )

    if not estimator.__sklearn_is_fitted__():
        includes.extend(["This estimator has not been fit yet."])

    else:
        excludes.extend(["This estimator has not been fit yet."])

    if extend_includes is not None:
        includes.extend(extend_includes)
    for check in set(includes):
        assert check in str(report)

    if extend_excludes is not None:
        excludes.extend(extend_excludes)
    for check in set(excludes):
        assert check not in str(report)

    return report

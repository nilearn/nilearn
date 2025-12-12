"""Testing utilities for reporting."""

import warnings
from pathlib import Path

import pytest

from nilearn._utils.helpers import is_matplotlib_installed
from nilearn._utils.html_document import WIDTH_DEFAULT
from nilearn.reporting import HTMLReport
from nilearn.reporting.html_report import MISSING_ENGINE_MSG


def generate_and_check_report(
    estimator,
    title: str | None = None,
    view: bool = False,
    pth: Path | None = None,
    extend_includes: list[str] | None = None,
    extend_excludes: list[str] | None = None,
    warnings_msg_to_check: list[str] | None = None,
    extra_warnings_allowed: bool = False,
    duplicate_warnings_allowed: bool = False,
    **kwargs,
) -> HTMLReport:
    """Generate a report and run generic checks on it.

    Parameters
    ----------
    model : any estimator with a generate_report method
        Model that generated the report

    title : str | None
        Title to include in report.
        If None is passed the estimator name should be in report instead.

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

    warnings_msg_to_check : Iterable[str] | None, default=None
        List of warning messages that will be:
        - raised during report generation
        - AND will be included in the HTML report

    extra_warnings_allowed :  bool
        Allows extra warnings to be thrown during report generation
        without being included in the HTML of the report.

    duplicate_warnings_allowed : bool
        In general we want to avoid throwing the same warnings
        too many times when generating a report.

    kwargs : dict
        Extra-parameters to pass to generate_report.
    """
    if warnings_msg_to_check is None:
        warnings_msg_to_check = []

    includes = []
    excludes = []

    if is_matplotlib_installed():
        excludes.extend(
            [MISSING_ENGINE_MSG, 'grey">No plotting engine found</p>']
        )
    else:
        includes.extend(
            [
                MISSING_ENGINE_MSG,
                'grey">No plotting engine found</p>',
            ]
        )

        warnings_msg_to_check.append(MISSING_ENGINE_MSG)

    if not estimator.__sklearn_is_fitted__():
        warnings_msg_to_check.append("This estimator has not been fit yet.")

    else:
        excludes.append("This estimator has not been fit yet.")

    if len(warnings_msg_to_check) > 0:
        includes.extend(['id="warnings"', *warnings_msg_to_check])

    if title is None:
        title = estimator.__class__.__name__
    includes.append(title)

    if len(warnings_msg_to_check) > 0:
        with pytest.warns(UserWarning) as warnings_list:
            report = estimator.generate_report(title=title, **kwargs)
    else:
        with warnings.catch_warnings(record=True) as all_warnings:
            report = estimator.generate_report(title=title, **kwargs)
            warnings_msg = [str(x.message) for x in all_warnings]
            if not extra_warnings_allowed:
                assert len(warnings_msg) == 0
            else:
                assert len(warnings_msg) > 0, (
                    "You can set extra_warnings_allowed to False"
                )

        if not duplicate_warnings_allowed:
            # make sure that warnings are not thrown several times
            # during report generation
            assert len(warnings_msg) == len(set(warnings_msg)), warnings_msg

    # TODO
    # make sure all estimators with generate_report have '_report_content'
    if hasattr(estimator, "_report_content"):
        assert estimator._report_content["title"] == title

    assert isinstance(report, HTMLReport)

    # only for debugging
    if view:
        report.open_in_browser()

    if len(warnings_msg_to_check) > 0:
        warnings_msg = [str(x.message) for x in warnings_list]
        for msg in warnings_msg_to_check:
            assert any(msg in x for x in warnings_msg), warnings_msg

    if extend_includes is not None:
        includes.extend(extend_includes)
    for check in set(includes):
        assert check in str(report)

    if extend_excludes is not None:
        excludes.extend(extend_excludes)
    for check in set(excludes):
        assert check not in str(report)

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

    if pth:
        # save to disk
        # useful for visual inspection
        # for manual checks or in case of test failure
        report.save_as_html(pth / "tmp.html")
        assert (pth / "tmp.html").exists()

    return report

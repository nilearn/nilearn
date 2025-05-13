"""Reporting code for nilearn.

This module implements plotting functions useful to report analysis results.
"""

from nilearn._utils.helpers import set_mpl_backend
from nilearn.reporting.get_clusters_table import get_clusters_table
from nilearn.reporting.html_report import HTMLReport

try:
    warning = (
        "nilearn.reporting.glm_reporter and nilearn.reporting.html_report "
        "requires nilearn.plotting."
    )
    set_mpl_backend(warning)

    from nilearn.reporting.glm_reporter import make_glm_report

    __all__ = [
        "HTMLReport",
        "get_clusters_table",
        "make_glm_report",
    ]

except ImportError:
    __all__ = [
        "HTMLReport",
        "get_clusters_table",
    ]

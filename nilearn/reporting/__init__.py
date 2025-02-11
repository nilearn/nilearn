"""Reporting code for nilearn.

This module implements plotting functions useful to report analysis results.
"""

# Author: Martin Perez-Guevara, Elvis Dohmatob, 2017

from nilearn._utils.helpers import set_mpl_backend
from nilearn.reporting.get_clusters_table import get_clusters_table

try:
    warning = (
        "nilearn.reporting.glm_reporter and nilearn.reporting.html_report "
        "requires nilearn.plotting."
    )
    set_mpl_backend(warning)

    from nilearn.reporting.glm_reporter import make_glm_report
    from nilearn.reporting.html_report import HTMLReport

    __all__ = [
        "HTMLReport",
        "get_clusters_table",
        "make_glm_report",
    ]

except ImportError:
    __all__ = ["get_clusters_table"]

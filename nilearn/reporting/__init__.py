"""Reporting code for nilearn.

This module implements plotting functions useful to report analysis results.
"""

# Author: Martin Perez-Guevara, Elvis Dohmatob, 2017

from nilearn._utils.helpers import _set_mpl_backend
from nilearn.reporting.get_clusters_table import get_clusters_table

try:
    WARNING = (
        "nilearn.reporting.make_glm_report and nilearn.reporting.HTMLReport "
        "requires 'matplotlib' to be installed."
        "\nIt can be installed with:\n"
        " pip install 'matplotlib'"
    )
    _set_mpl_backend(WARNING)

    from nilearn.reporting.glm_reporter import make_glm_report
    from nilearn.reporting.html_report import HTMLReport

    __all__ = [
        "HTMLReport",
        "get_clusters_table",
        "make_glm_report",
    ]

except ImportError:
    __all__ = ["get_clusters_table"]

"""Reporting code for nilearn.

This module implements plotting functions useful to report analysis results.
"""

from nilearn.reporting.get_clusters_table import get_clusters_table
from nilearn.reporting.glm_reporter import make_glm_report
from nilearn.reporting.html_report import HTMLReport

# TODO (nilearn >= 0.15.0) remove make_glm_report

__all__ = [
    "HTMLReport",
    "get_clusters_table",
    "make_glm_report",
]

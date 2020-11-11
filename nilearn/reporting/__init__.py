"""
Reporting code for nilearn
"""
"""
This module implements plotting functions useful to report analysis results.

Author: Martin Perez-Guevara, Elvis Dohmatob, 2017
"""

from nilearn.reporting.html_report import HTMLReport
from nilearn.reporting._get_clusters_table import get_clusters_table
from nilearn.reporting.glm_reporter import make_glm_report

__all__ = ['HTMLReport', 'get_clusters_table', 'make_glm_report']

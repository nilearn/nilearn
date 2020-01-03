"""
This module implements plotting functions useful to report analysis results.

Author: Martin Perez-Guevara, Elvis Dohmatob, 2017
"""
import warnings

try:
    import matplotlib
except ImportError:
    warnings.warn("Nistats' reporting module requires matplotlib to work. "
                  "Install it using:\n"
                  "pip install matplotlib"
                  )
else:
    # matplotlib backend must be set before importing  any of its functions.
    matplotlib.use('Agg')

from ._compare_niimgs import compare_niimgs
from ._get_clusters_table import get_clusters_table
from ._plot_matrices import plot_contrast_matrix, plot_design_matrix
from .glm_reporter import make_glm_report
from .sphinx_report import _ReportScraper

__all__ = [compare_niimgs,
           get_clusters_table,
           make_glm_report,
           plot_contrast_matrix,
           plot_design_matrix,
           ]

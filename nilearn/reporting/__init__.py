"""
Reporting code for nilearn
"""
"""
This module implements plotting functions useful to report analysis results.

Author: Martin Perez-Guevara, Elvis Dohmatob, 2017
"""
import warnings

''' Ensure the html_report & html_document imports happen before the remaining 
nilearn.reporting imports to avoid circular imports. '''
from .html_report import (ReportMixin, HTMLReport)
from .html_document import (HTMLDocument, set_max_img_views_before_warning)

try:
    import matplotlib  # noqa: F401
except ImportError:
    warnings.warn("Nilearn's reporting module requires matplotlib to work. "
                  "Install it using:\n"
                  "pip install matplotlib"
                  )
else:
    # matplotlib backend must be set before importing  any of its functions.
    from nilearn.plotting import _set_mpl_backend
    _set_mpl_backend()

    from nilearn.reporting._compare_niimgs import compare_niimgs
    from nilearn.reporting._get_clusters_table import get_clusters_table
    from nilearn.reporting._plot_matrices import plot_contrast_matrix
    from nilearn.reporting._plot_matrices import plot_event
    from nilearn.reporting._plot_matrices import plot_design_matrix
    from nilearn.reporting.glm_reporter import make_glm_report


__all__ = ['ReportMixin', 'HTMLDocument', 'HTMLReport',
           'set_max_img_views_before_warning',
           'compare_niimgs',
           'get_clusters_table',
           'make_glm_report',
           'plot_contrast_matrix',
           'plot_event',
           'plot_design_matrix',
           ]

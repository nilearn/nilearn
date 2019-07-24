"""
Reporting code for nilearn
"""

from .html_report import (ReportMixin, HTMLDocument,
                          set_max_img_views_before_warning)

__all__ = ['ReportMixin', 'HTMLDocument', 'set_max_img_views_before_warning']

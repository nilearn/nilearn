"""
Reporting code for nilearn
"""

from .html_report import (ReportMixin, HTMLReport)

from .html_document import (HTMLDocument, set_max_img_views_before_warning)

from .sphinx_report import (_ReportScraper)

__all__ = ['ReportMixin', 'HTMLDocument', 'HTMLReport',
           'set_max_img_views_before_warning', '_ReportScraper']

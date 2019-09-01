# Scraper for sphinx-gallery
# Copied from https://github.com/mne-tools/mne-python/

import os.path as op
from shutil import copyfile
from nilearn.reporting import HTMLReport


_SCRAPER_TEXT = '''
.. only:: builder_html

    .. container:: row

        .. raw:: html

            <iframe class="sg_report" src="{0}"></iframe>
'''  # noqa: E501


class _ReportScraper(object):
    """Scrape Reports for Sphinx display.
    """

    def __init__(self):
        self.app = None
        self.files = dict()

    def __repr__(self):
        return '<ReportScraper>'

    def __call__(self, block, block_vars, gallery_conf):
        for report in block_vars['example_globals'].values():
            if (isinstance(report, HTMLReport) and
                    gallery_conf['builder_name'] == 'html'):
                # embed links/iframe
                data = _SCRAPER_TEXT.format(report.get_iframe())
                return data
        return ''

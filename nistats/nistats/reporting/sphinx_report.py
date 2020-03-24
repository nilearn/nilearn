# -*- coding: utf-8 -*-
"""
HTML Scraper for sphinx-gallery

Adapted by ELizabeth Dupre @emdupre for the Nilearn project
in https://github.com/nilearn/nilearn/pull/2019
Inspired from https://github.com/mne-tools/mne-python/
"""

import weakref

from nilearn.plotting.js_plotting_utils import HTMLDocument


_SCRAPER_TEXT = '''
.. only:: builder_html

    .. container:: row sg-report

        .. raw:: html

            {0}

'''  # noqa: E501


def indent_and_espace(text, amount=12):
    "Indent, skip empty lines, and escape string delimiters"
    return (''.join(amount * ' ' + line.replace("'", '"')
                    for line in text.splitlines(True)
                    if line) + amount * ' ')


class _ReportScraper(object):
    """Scrape Reports for Sphinx display.
    """

    def __init__(self):
        self.app = None
        self.displayed_reports = weakref.WeakSet()

    def __repr__(self):
        return '<ReportScraper>'

    def __call__(self, block, block_vars, gallery_conf):
        for report in block_vars['example_globals'].values():
            if (isinstance(report, HTMLDocument) and
                    gallery_conf['builder_name'] == 'html'):
                if report in self.displayed_reports:
                    continue
                report_str = report._repr_html_()
                data = _SCRAPER_TEXT.format(indent_and_espace(report_str))
                self.displayed_reports.add(report)
                return data
        return ''

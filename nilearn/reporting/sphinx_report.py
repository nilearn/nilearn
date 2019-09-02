# Scraper for sphinx-gallery
# Inspired from https://github.com/mne-tools/mne-python/

from nilearn.reporting import HTMLReport


_SCRAPER_TEXT = '''
.. only:: builder_html

    .. container:: row sg-report

        .. raw:: html

            <iframe class="sg-report" frameBorder="0" width="100%"
                srcdoc='{0}'></iframe>

'''  # noqa: E501

def indent_and_espace(text, amount=12):
    "Indent, skip empty lines, and escape string delimiters"
    return (''.join(amount * ' ' + line.replace("'", '"')
                for line in text.splitlines(True)
                if line)
            + 12 * ' ')


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
                report_str = report.get_standalone()
                data = _SCRAPER_TEXT.format(indent_and_espace(report_str))
                return data
        return ''

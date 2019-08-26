# Scraper for sphinx-gallery
# Copied from https://github.com/mne-tools/mne-python/

from shutil import copyfile

_SCRAPER_TEXT = '''
.. only:: builder_html
    .. container:: row
        .. rubric:: The `HTML document <{0}>`__ written by the :meth:`.generate_report` meethod:
        .. raw:: html
            <iframe class="sg_report" sandbox="allow-scripts" src="{0}"></iframe>
'''  # noqa: E501
# Adapted from fa-file-code
_FA_FILE_CODE = '<svg class="sg_report" role="img" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 384 512"><path fill="#dec" d="M149.9 349.1l-.2-.2-32.8-28.9 32.8-28.9c3.6-3.2 4-8.8.8-12.4l-.2-.2-17.4-18.6c-3.4-3.6-9-3.7-12.4-.4l-57.7 54.1c-3.7 3.5-3.7 9.4 0 12.8l57.7 54.1c1.6 1.5 3.8 2.4 6 2.4 2.4 0 4.8-1 6.4-2.8l17.4-18.6c3.3-3.5 3.1-9.1-.4-12.4zm220-251.2L286 14C277 5 264.8-.1 252.1-.1H48C21.5 0 0 21.5 0 48v416c0 26.5 21.5 48 48 48h288c26.5 0 48-21.5 48-48V131.9c0-12.7-5.1-25-14.1-34zM256 51.9l76.1 76.1H256zM336 464H48V48h160v104c0 13.3 10.7 24 24 24h104zM209.6 214c-4.7-1.4-9.5 1.3-10.9 6L144 408.1c-1.4 4.7 1.3 9.6 6 10.9l24.4 7.1c4.7 1.4 9.6-1.4 10.9-6L240 231.9c1.4-4.7-1.3-9.6-6-10.9zm24.5 76.9l.2.2 32.8 28.9-32.8 28.9c-3.6 3.2-4 8.8-.8 12.4l.2.2 17.4 18.6c3.3 3.5 8.9 3.7 12.4.4l57.7-54.1c3.7-3.5 3.7-9.4 0-12.8l-57.7-54.1c-3.5-3.3-9.1-3.2-12.4.4l-17.4 18.6c-3.3 3.5-3.1 9.1.4 12.4z" class=""></path></svg>'  # noqa: E501


class _ReportScraper(object):
    """Scrape Report outputs.
    Only works properly if conf.py is configured properly and the file
    is written to the same directory as the example script.
    """

    def __init__(self):
        self.app = None
        self.files = dict()

    def __repr__(self):
        return '<ReportScraper>'

    def __call__(self, block, block_vars, gallery_conf):
        for report in block_vars['example_globals'].values():
            if (isinstance(report, Report) and hasattr(report, 'fname') and
                    report.fname.endswith('.html') and
                    gallery_conf['builder_name'] == 'html'):
                # Thumbnail
                image_path_iterator = block_vars['image_path_iterator']
                img_fname = next(image_path_iterator)
                img_fname = img_fname.replace('.png', '.svg')
                with open(img_fname, 'w') as fid:
                    fid.write(_FA_FILE_CODE)
                # copy HTML file
                html_fname = op.basename(report.fname)
                out_fname = op.join(
                    self.app.builder.outdir,
                    op.relpath(op.dirname(block_vars['target_file']),
                               self.app.builder.srcdir), html_fname)
                self.files[report.fname] = out_fname
                # embed links/iframe
                data = _SCRAPER_TEXT.format(html_fname)
                return data
        return ''

    def copyfiles(self, *args, **kwargs):
        for key, value in self.files.items():
            copyfile(key, value)

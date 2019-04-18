import os
import io
import base64

from nilearn.externals import tempita
from . import js_plotting_utils as plot_utils


def _embed_img(display):
    """
    Parameters
    ----------
    display: obj
        A matplotlib object to display

    Returns
    -------
    embed : str
        Binary image string
    """

    io_buffer = io.BytesIO()
    display.savefig(io_buffer)

    io_buffer.seek(0)
    data = base64.b64encode(io_buffer.read())

    return '{}'.format(data.decode())


class ReportMixin():
    """
    Generate a report for Nilearn objects.

    Report is useful to visualize steps in a processing pipeline.
    Example use case: visualize the overlap of a mask and reference image
    in NiftiMasker.

    Returns
    -------
    report : HTML report with embedded content
    """

    def update_template(self, title, content, description=None):
        """
        Populate a report with content.

        Parameters
        ----------
        title: str
            The title for the report
        content: img
            The content to display
        description: str
            An optional description of the content

        Returns
        -------
        html : populated HTML report
        """
        template_name = 'report_template.html'
        template_path = os.path.join(
            os.path.dirname(__file__), 'data', 'html', template_name)
        tpl = tempita.HTMLTemplate.from_filename(template_path,
                                                 encoding='utf-8')

        html = tpl.substitute(title=title, content=content,
                              description=description)
        return plot_utils.HTMLDocument(html)

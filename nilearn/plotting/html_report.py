import os
import io
import base64
import matplotlib as mpl
from contextlib import contextmanager

from nilearn.externals import tempita
from . import js_plotting_utils as plot_utils


@contextmanager
def _update_mpl_backend():
    """
    Safely, temporarily set matplotlib backend to 'Agg'
    """
    backend = mpl.get_backend()
    mpl.use('Agg')
    try:
        yield
    finally:
        mpl.use(backend)


def _str_params(params):
    """
    Parameters
    ----------
    params: dict
        A dictionary of input values to a function
    """
    for k, v in params.items():
        if v is None:
            params[k] = 'None'
    return params


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
    display.close()

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

    def update_template(self, title, docstring, content,
                        parameters, description=None):
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
                              docstring=docstring,
                              parameters=parameters,
                              description=description)
        return plot_utils.HTMLDocument(html)

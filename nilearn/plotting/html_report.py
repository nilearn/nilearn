import os
import io
import base64
from contextlib import contextmanager
from string import Template

import matplotlib as mpl

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


class HTMLReport():
    """ A report written as html
    """
    def __init__(self, head_tpl, body):
        """ The head_tpl is meant for display as a full page, eg writing on
            disk. The body is used for embedding in an existing page.
        """
        self.head_tpl = head_tpl
        self.body = body

    def _repr_html_(self):
        """
        Used by the Jupyter notebook.

        Users normally won't call this method explicitely.
        """
        return self.body

    def __str__(self):
        return self.body

    def get_standalone(self):
        return self.head_tpl.substitute(body=self.body)

    def save_as_html(self, file_name):
        """
        Save the plot in an HTML file, that can later be opened in a browser.
        """
        html_document = plot_utils.HTMLDocument(self.get_standalone())
        html_document.save_as_html(file_name)
        return html_document

    def open_in_browser(self, file_name=None, temp_file_lifetime=30):
        """
        Save to a temporary HTML file and open it in a browser.

        Parameters
        ----------

        file_name : str, optional
            .html file to use as temporary file

        temp_file_lifetime : float, optional (default=30.)
            Time, in seconds, after which the temporary file is removed.
            If None, it is never removed.

        """
        html_document = plot_utils.HTMLDocument(self.get_standalone())
        html_document.open_in_browser(file_name=file_name,
                                      temp_file_lifetime=temp_file_lifetime)
        return html_document


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
        body_template_name = 'report_body_template.html'
        body_template_path = os.path.join(
            os.path.dirname(__file__), 'data', 'html', body_template_name)
        tpl = tempita.HTMLTemplate.from_filename(body_template_path,
                                                 encoding='utf-8')

        body = tpl.substitute(title=title, content=content,
                              docstring=docstring,
                              parameters=parameters,
                              description=description)
        head_template_name = 'report_head_template.html'
        head_template_path = os.path.join(
            os.path.dirname(__file__), 'data', 'html', head_template_name)
        with open(head_template_path, 'r') as head_file:
            head_tpl = Template(head_file.read())
        return HTMLReport(body=body, head_tpl=head_tpl)

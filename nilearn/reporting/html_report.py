import io
import copy
import base64
import warnings
from pathlib import Path
from string import Template

from .html_document import HTMLDocument
from nilearn.externals import tempita


def _embed_img(display):
    """
    Parameters
    ----------
    display: obj
        A Nilearn plotting object to display

    Returns
    -------
    embed : str
        Binary image string
    """
    if display is None:  # no image to display
        return None

    else:  # we were passed a matplotlib display
        io_buffer = io.BytesIO()
        display.frame_axes.figure.savefig(io_buffer, format='svg',
                                          facecolor='white',
                                          edgecolor='white')
        display.close()

        io_buffer.seek(0)
        data = base64.b64encode(io_buffer.read())

        return '{}'.format(data.decode())


def _str_params(params):
    """
    Convert NoneType values to the string 'None'
    for display.

    Parameters
    ----------
    params: dict
        A dictionary of input values to a function
    """
    params_str = copy.deepcopy(params)
    for k, v in params_str.items():
        if v is None:
            params_str[k] = 'None'
    return params_str


def _update_template(title, docstring, content, overlay,
                     parameters, description=None):
    """
    Populate a report with content.

    Parameters
    ----------
    title : str
        The title for the report
    docstring : str
        The introductory docstring for the reported object
    content : img
        The content to display
    overlay : img
        Overlaid content, to appear on hover
    parameters : dict
        A dictionary of object parameters and their values
    description : str
        An optional description of the content

    Returns
    -------
    HTMLReport : an instance of a populated HTML report
    """
    resource_path = Path(__file__).resolve().parent.joinpath('data', 'html')

    body_template_name = 'report_body_template.html'
    body_template_path = resource_path.joinpath(body_template_name)
    tpl = tempita.HTMLTemplate.from_filename(str(body_template_path),
                                             encoding='utf-8')
    body = tpl.substitute(title=title, content=content,
                          overlay=overlay,
                          docstring=docstring,
                          parameters=parameters,
                          description=description)

    head_template_name = 'report_head_template.html'
    head_template_path = resource_path.joinpath(head_template_name)
    with open(str(head_template_path), 'r') as head_file:
        head_tpl = Template(head_file.read())

    return HTMLReport(body=body, head_tpl=head_tpl)


class ReportMixin:
    """
    A class to provide general reporting functionality
    """

    def _define_overlay(self):
        """
        Determine whether an overlay was provided and
        update the report text as appropriate.

        Parameters
        ----------

        Returns
        -------
        """
        displays = self._reporting()

        if len(displays) == 1:  # set overlay to None
            overlay, image = None, displays[0]

        elif len(displays) == 2:
            overlay, image = displays[0], displays[1]

        return overlay, image

    def generate_report(self):
        """
        Generate a report for Nilearn objects.

        Reports are useful to visualize steps in a processing pipeline.
        Example use case: visualize the overlap of a mask and reference image
        in NiftiMasker.

        Returns
        -------
        report : HTMLReport
        """
        if not hasattr(self, '_reporting_data'):
            warnings.warn('This object has not been fitted yet ! '
                          'Make sure to run `fit` before inspecting reports.')
            report = _update_template(title='Empty Report',
                                      docstring=('This report was not '
                                                 'generated. Please `fit` the '
                                                 'object.'),
                                      content=_embed_img(None),
                                      overlay=None,
                                      parameters=dict())

        elif self._reporting_data is None:
            warnings.warn('Report generation not enabled ! '
                          'No visual outputs will be created.')
            report = _update_template(title='Empty Report',
                                      docstring=('This report was not '
                                                 'generated. Please check '
                                                 'that reporting is enabled.'),
                                      content=_embed_img(None),
                                      overlay=None,
                                      parameters=dict())

        else:  # We can create a report
            overlay, image = self._define_overlay()
            description = self._report_description
            parameters = _str_params(self.get_params())
            docstring = self.__doc__
            snippet = docstring.partition('Parameters\n    ----------\n')[0]
            report = _update_template(title=self.__class__.__name__,
                                      docstring=snippet,
                                      content=_embed_img(image),
                                      overlay=_embed_img(overlay),
                                      parameters=parameters,
                                      description=description)
        return report


class HTMLReport(HTMLDocument):
    """
    A report written as HTML.
    Methods such as save_as_html(), open_in_browser()
    are inherited from HTMLDocument
    """
    def __init__(self, head_tpl, body):
        """ The head_tpl is meant for display as a full page, eg writing on
            disk. The body is used for embedding in an existing page.
        """
        html = head_tpl.substitute(body=body)
        super(HTMLReport, self).__init__(html)
        self.head_tpl = head_tpl
        self.body = body

    def _repr_html_(self):
        """
        Used by the Jupyter notebook.
        Users normally won't call this method explicitly.
        """
        return self.body

    def __str__(self):
        return self.body

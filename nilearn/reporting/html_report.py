import io
import base64
import warnings
from pathlib import Path
from string import Template

from nilearn.externals import tempita
from nilearn.plotting import js_plotting_utils as plot_utils


def generate_report(obj):
    """
    Generate a report for Nilearn objects.

    Report is useful to visualize steps in a processing pipeline.
    Example use case: visualize the overlap of a mask and reference image
    in NiftiMasker.

    Returns
    -------
    report : HTMLDocument
    """
    if not hasattr(obj, 'input_'):
        warnings.warn('Report generation not enabled !'
                      'No visual outputs will be created.')
        report = update_template(title='Empty Report',
                                 docstring='This report was not generated.',
                                 content=_embed_img(None),
                                 parameters=dict())

    else:
        description = obj._report_description
        parameters = _str_params(obj.get_params())
        docstring = obj.__doc__.partition('Parameters\n    ----------\n')[0]
        report = update_template(title=obj.__class__.__name__,
                                 docstring=docstring,
                                 content=_embed_img(obj._reporting()),
                                 parameters=parameters,
                                 description=description)
    return report


def update_template(title, docstring, content,
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
    parameters : dict
        A dictionary of object parameters and their values
    description : str
        An optional description of the content

    Returns
    -------
    HTMLReport : an instance of a populated HTML report
    """
    resource_path = Path(__file__).resolve().parent.joinpath('data')

    body_template_name = 'report_body_template.html'
    body_template_path = resource_path.joinpath('html', body_template_name)
    tpl = tempita.HTMLTemplate.from_filename(body_template_path,
                                             encoding='utf-8')
    body = tpl.substitute(css=resource_path.joinpath('css'),
                          title=title, content=content,
                          docstring=docstring,
                          parameters=parameters,
                          description=description)

    head_template_name = 'report_head_template.html'
    head_template_path = resource_path.joinpath('html', head_template_name)
    with open(head_template_path, 'r') as head_file:
        head_tpl = Template(head_file.read())

    return HTMLReport(body=body, head_tpl=head_tpl)


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
    if display is not None:
        io_buffer = io.BytesIO()
        display.savefig(io_buffer)
        display.close()
        io_buffer.seek(0)

    else:
        logo_name = 'nilearn-logo-small.png'
        logo_dir = Path(__file__).resolve().parent.parent
        logo_path = logo_dir.joinpath('doc', 'logos', logo_name)
        io_buffer = open(logo_path, 'rb')

    data = base64.b64encode(io_buffer.read())

    return '{}'.format(data.decode())


class HTMLReport(plot_utils.HTMLDocument):
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

    # save_as_html, open_in_browser are inherited from HTMLDocument

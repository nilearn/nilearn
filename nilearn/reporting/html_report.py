import io
import os
import copy
import base64
import weakref
import warnings
import tempfile
import webbrowser
import subprocess
from html import escape
from pathlib import Path
from string import Template

from nilearn.externals import tempita

MAX_IMG_VIEWS_BEFORE_WARNING = 10


def set_max_img_views_before_warning(new_value):
    """Set the number of open views which triggers a warning.

    If `None` or a negative number, disable the memory warning.
    """
    global MAX_IMG_VIEWS_BEFORE_WARNING
    MAX_IMG_VIEWS_BEFORE_WARNING = new_value


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
    if display is None:  # no display, show single transparent pixel
        data = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAA" +
                "AAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")
        return data

    else:  # we were passed a matplotlib display
        io_buffer = io.BytesIO()
        display.frame_axes.figure.savefig(io_buffer, format='svg')
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


def _remove_after_n_seconds(file_name, n_seconds):
    script = os.path.join(os.path.dirname(__file__), 'rm_file.py')
    subprocess.Popen(['python', script, file_name, str(n_seconds)])


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

        Report is useful to visualize steps in a processing pipeline.
        Example use case: visualize the overlap of a mask and reference image
        in NiftiMasker.

        Returns
        -------
        report : HTMLReport
        """
        if not hasattr(self, 'input_'):
            warnings.warn('Report generation not enabled !'
                          'No visual outputs will be created.')
            report = _update_template(title='Empty Report',
                                      docstring=('This report was not '
                                                 'generated. Please check '
                                                 'that reporting is enabled.'),
                                      content=_embed_img(None),
                                      overlay=_embed_img(None),
                                      parameters=dict())

        else:
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


class HTMLDocument(object):
    """
    Embeds a plot in a web page.

    If you are running a Jupyter notebook, the plot will be displayed
    inline if this object is the output of a cell.
    Otherwise, use open_in_browser() to open it in a web browser (or
    save_as_html("filename.html") to save it as an html file).

    use str(document) or document.html to get the content of the web page,
    and document.get_iframe() to have it wrapped in an iframe.

    The head_tpl is meant for display as a full page, eg writing on
    disk. The body is used for embedding in an existing page.
    """
    _all_open_html_repr = weakref.WeakSet()

    def __init__(self, html, width=600, height=400):
        self.html = html
        self.width = width
        self.height = height
        self._temp_file = None
        self._check_n_open()

    def _check_n_open(self):
        HTMLDocument._all_open_html_repr.add(self)
        if MAX_IMG_VIEWS_BEFORE_WARNING is None:
            return
        if MAX_IMG_VIEWS_BEFORE_WARNING < 0:
            return
        if len(HTMLDocument._all_open_html_repr
               ) > MAX_IMG_VIEWS_BEFORE_WARNING - 1:
            warnings.warn('It seems you have created more than {} '
                          'nilearn views. As each view uses dozens '
                          'of megabytes of RAM, you might want to '
                          'delete some of them.'.format(
                              MAX_IMG_VIEWS_BEFORE_WARNING))

    def resize(self, width, height):
        """Resize the plot displayed in a Jupyter notebook."""
        self.width, self.height = width, height
        return self

    def get_iframe(self, width=None, height=None):
        """
        Get the document wrapped in an inline frame.

        For inserting in another HTML page of for display in a Jupyter
        notebook.

        """
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        escaped = escape(self.html, quote=True)
        wrapped = ('<iframe srcdoc="{}" width="{}" height="{}" '
                   'frameBorder="0"></iframe>').format(escaped, width, height)
        return wrapped

    def get_standalone(self):
        """ Get the plot in an HTML page."""
        return self.html

    def _repr_html_(self):
        """
        Used by the Jupyter notebook.

        Users normally won't call this method explicitely.
        """
        return self.get_iframe()

    def __str__(self):
        return self.html

    def save_as_html(self, file_name):
        """
        Save the plot in an HTML file, that can later be opened in a browser.
        """
        with open(file_name, 'wb') as f:
            f.write(self.get_standalone().encode('utf-8'))

    def open_in_browser(self, file_name=None, temp_file_lifetime=30):
        """
        Save the plot to a temporary HTML file and open it in a browser.

        Parameters
        ----------

        file_name : str, optional
            .html file to use as temporary file

        temp_file_lifetime : float, optional (default=30.)
            Time, in seconds, after which the temporary file is removed.
            If None, it is never removed.

        """
        if file_name is None:
            fd, file_name = tempfile.mkstemp('.html', 'nilearn_plot_')
            os.close(fd)
            named_file = False
        else:
            named_file = True
        self.save_as_html(file_name)
        self._temp_file = file_name
        file_size = os.path.getsize(file_name) / 1e6
        if temp_file_lifetime is None:
            if not named_file:
                warnings.warn(
                    ("Saved HTML in temporary file: {}\n"
                     "file size is {:.1f}M, delete it when you're done, "
                     "for example by calling this.remove_temp_file").format(
                         file_name, file_size))
        else:
            _remove_after_n_seconds(self._temp_file, temp_file_lifetime)
        webbrowser.open('file://{}'.format(file_name))

    def remove_temp_file(self):
        """
        Remove the temporary file created by `open_in_browser`, if necessary.
        """
        if self._temp_file is None:
            return
        if not os.path.isfile(self._temp_file):
            return
        os.remove(self._temp_file)
        print('removed {}'.format(self._temp_file))
        self._temp_file = None


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
        self.head_tpl = head_tpl
        self.body = body
        self.html = self.get_standalone()

    def _repr_html_(self):
        """
        Used by the Jupyter notebook.
        Users normally won't call this method explicitly.
        """
        return self.body

    def __str__(self):
        return self.body

    def get_standalone(self):
        return self.head_tpl.substitute(body=self.body)

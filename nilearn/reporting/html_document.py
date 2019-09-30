import os
import weakref
import warnings
import tempfile
import webbrowser
import subprocess
from html import escape

MAX_IMG_VIEWS_BEFORE_WARNING = 10


def set_max_img_views_before_warning(new_value):
    """Set the number of open views which triggers a warning.

    If `None` or a negative number, disable the memory warning.
    """
    global MAX_IMG_VIEWS_BEFORE_WARNING
    MAX_IMG_VIEWS_BEFORE_WARNING = new_value


def _remove_after_n_seconds(file_name, n_seconds):
    script = os.path.join(os.path.dirname(__file__), 'rm_file.py')
    subprocess.Popen(['python', script, file_name, str(n_seconds)])


class HTMLDocument(object):
    """
    Embeds a plot in a web page.

    If you are running a Jupyter notebook, the plot will be displayed
    inline if this object is the output of a cell.
    Otherwise, use open_in_browser() to open it in a web browser (or
    save_as_html("filename.html") to save it as an html file).

    use str(document) or document.html to get the content of the web page,
    and document.get_iframe() to have it wrapped in an iframe.

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

"""Handle HTML plotting."""

import os
import subprocess
import sys
import tempfile
import warnings
import weakref
import webbrowser
from html import escape

MAX_IMG_VIEWS_BEFORE_WARNING = 10


def set_max_img_views_before_warning(new_value):
    """Set the number of open views which triggers a warning.

    If `None` or a negative number, disable the memory warning.
    """
    global MAX_IMG_VIEWS_BEFORE_WARNING
    MAX_IMG_VIEWS_BEFORE_WARNING = new_value


def _remove_after_n_seconds(file_name, n_seconds):
    script = os.path.join(os.path.dirname(__file__), "rm_file.py")
    proc = subprocess.Popen(
        [sys.executable, script, file_name, str(n_seconds)]
    )
    return proc


class HTMLDocument:
    """Embeds a plot in a web page.

    If you are running a Jupyter notebook, the plot will be displayed
    inline if this object is the output of a cell.
    Otherwise, use ``open_in_browser()`` to open it in a web browser (or
    ``save_as_html("filename.html")`` to save it as an html file).

    Use ``str(document)`` or ``document.html`` to get the content of the
    web page, and ``document.get_iframe()`` to have it wrapped in an iframe.

    """

    _all_open_html_repr = weakref.WeakSet()

    def __init__(self, html, width=600, height=400):
        self.html = html
        self.width = width
        self.height = height
        self._temp_file = None
        self._check_n_open()
        self._temp_file_removing_proc = None

    def _check_n_open(self):
        HTMLDocument._all_open_html_repr.add(self)
        if MAX_IMG_VIEWS_BEFORE_WARNING is None:
            return
        if MAX_IMG_VIEWS_BEFORE_WARNING < 0:
            return
        if (
            len(HTMLDocument._all_open_html_repr)
            > MAX_IMG_VIEWS_BEFORE_WARNING - 1
        ):
            warnings.warn(
                "It seems you have created "
                f"more than {MAX_IMG_VIEWS_BEFORE_WARNING} "
                "nilearn views. As each view uses dozens "
                "of megabytes of RAM, you might want to "
                "delete some of them."
            )

    def resize(self, width, height):
        """Resize the plot displayed in a Jupyter notebook.

        Parameters
        ----------
        width : :obj:`int`
            New width of the plot.

        height : :obj:`int`
            New height of the plot.

        """
        self.width, self.height = width, height
        return self

    def get_iframe(self, width=None, height=None):
        """Get the document wrapped in an inline frame.

        For inserting in another HTML page of for display in a Jupyter
        notebook.

        Parameters
        ----------
        width : :obj:`int` or ``None``, optional
            Width of the inline frame. Default=None.

        height : :obj:`int` or ``None``, optional
            Height of the inline frame. Default=None.

        Returns
        -------
        wrapped : :obj:`str`
            Raw HTML code for the inline frame.

        """
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        escaped = escape(self.html, quote=True)
        wrapped = (
            f'<iframe srcdoc="{escaped}" '
            f'width="{width}" height="{height}" '
            'frameBorder="0"></iframe>'
        )
        return wrapped

    def get_standalone(self):
        """Return the plot in an HTML page."""
        return self.html

    def _repr_html_(self):
        """Return html representation of the plot.

        Used by the Jupyter notebook.

        Users normally won't call this method explicitly.
        """
        return self.get_iframe()

    def __str__(self):
        return self.html

    def save_as_html(self, file_name):
        """Save the plot in an HTML file, that can later be opened \
        in a browser.

        Parameters
        ----------
        file_name : :obj:`str`
            Path to the HTML file used for saving.

        """
        with open(file_name, "wb") as f:
            f.write(self.get_standalone().encode("utf-8"))

    def open_in_browser(self, file_name=None, temp_file_lifetime=30):
        """Save the plot to a temporary HTML file and open it in a browser.

        Parameters
        ----------
        file_name : :obj:`str` or ``None``, optional
            HTML file to use as a temporary file. Default=None.

        temp_file_lifetime : :obj:`float`, optional
            Time, in seconds, after which the temporary file is removed.
            If None, it is never removed.
            Default=30.

        """
        if file_name is None:
            fd, file_name = tempfile.mkstemp(".html", "nilearn_plot_")
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
                    f"Saved HTML in temporary file: {file_name}\n"
                    f"file size is {file_size:.1f}M, "
                    "delete it when you're done, "
                    "for example by calling this.remove_temp_file"
                )
        else:
            self._temp_file_removing_proc = _remove_after_n_seconds(
                self._temp_file, temp_file_lifetime
            )
        webbrowser.open(f"file://{file_name}")

    def remove_temp_file(self):
        """Remove the temporary file created by \
        ``open_in_browser``, if necessary."""
        if self._temp_file is None:
            return
        if not os.path.isfile(self._temp_file):
            return
        os.remove(self._temp_file)
        print(f"removed {self._temp_file}")
        self._temp_file = None

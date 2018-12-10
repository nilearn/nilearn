"""
Helper functions for views, i.e. interactive plots from html_surface and
html_connectome.
"""

import os
import base64
import webbrowser
import tempfile
import warnings
import subprocess
import weakref
try:
    from html import escape  # Unavailable in Py2
except ImportError:  # Can be removed once we EOL Py2 support for NiLearn
    from cgi import escape  # Deprecated in Py3, necessary for Py2

import matplotlib as mpl
import numpy as np
from matplotlib import cm as mpl_cm

from .._utils.extmath import fast_abs_percentile
from .._utils.param_validation import check_threshold
from .. import surface


def add_js_lib(html, embed_js=True):
    """
    Add javascript libraries to html template.

    if embed_js is True, jquery and plotly are embedded in resulting page.
    otherwise, they are loaded via CDNs.
    """
    js_dir = os.path.join(os.path.dirname(__file__), 'data', 'js')
    with open(os.path.join(js_dir, 'surface-plot-utils.js')) as f:
        js_utils = f.read()
    if not embed_js:
        js_lib = """
        <script
        src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js">
        </script>
        <script src="https://cdn.plot.ly/plotly-gl3d-latest.min.js"></script>
        <script>
        {}
        </script>
        """.format(js_utils)
    else:
        with open(os.path.join(js_dir, 'jquery.min.js')) as f:
            jquery = f.read()
        with open(os.path.join(js_dir, 'plotly-gl3d-latest.min.js')) as f:
            plotly = f.read()
        js_lib = """
        <script>{}</script>
        <script>{}</script>
        <script>
        {}
        </script>
        """.format(jquery, plotly, js_utils)
    return html.replace('INSERT_JS_LIBRARIES_HERE', js_lib)


def get_html_template(template_name):
    """Get an HTML file from package data"""
    template_path = os.path.join(
        os.path.dirname(__file__), 'data', 'html', template_name)
    with open(template_path, 'rb') as f:
        return f.read().decode('utf-8')


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
        if len(HTMLDocument._all_open_html_repr) > 9:
            warnings.warn('It seems you have created more than 10 '
                          'nilearn views. As each view uses dozens '
                          'of megabytes of RAM, you might want to '
                          'delete some of them.')

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
        wrapped = ('<iframe srcdoc="{}" width={} height={} '
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
            f.write(self.html.encode('utf-8'))

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
            fd, file_name = tempfile.mkstemp('.html', 'nilearn_surface_plot_')
            os.close(fd)
        self.save_as_html(file_name)
        self._temp_file = file_name
        file_size = os.path.getsize(file_name) / 1e6
        if temp_file_lifetime is None:
            print(("Saved HTML in temporary file: {}\n"
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


def colorscale(cmap, values, threshold=None, symmetric_cmap=True,
               vmax=None, vmin=None):
    """Normalize a cmap, put it in plotly format, get threshold and range."""
    cmap = mpl_cm.get_cmap(cmap)
    abs_values = np.abs(values)
    if not symmetric_cmap and (values.min() < 0):
        warnings.warn('you have specified symmetric_cmap=False '
                      'but the map contains negative values; '
                      'setting symmetric_cmap to True')
        symmetric_cmap = True
    if symmetric_cmap and vmin is not None:
        warnings.warn('vmin cannot be chosen when cmap is symmetric')
        vmin = None
    if threshold is not None:
        if vmin is not None:
            warnings.warn('choosing both vmin and a threshold is not allowed; '
                          'setting vmin to 0')
        vmin = 0
    if vmax is None:
        vmax = abs_values.max()
    if symmetric_cmap:
        vmin = - vmax
    if vmin is None:
        vmin = values.min()
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    abs_threshold = None
    if threshold is not None:
        abs_threshold = check_threshold(threshold, values, fast_abs_percentile)
        istart = int(norm(-abs_threshold, clip=True) * (cmap.N - 1))
        istop = int(norm(abs_threshold, clip=True) * (cmap.N - 1))
        for i in range(istart, istop):
            cmaplist[i] = (0.5, 0.5, 0.5, 1.)  # just an average gray color
    our_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    x = np.linspace(0, 1, 100)
    rgb = our_cmap(x, bytes=True)[:, :3]
    rgb = np.array(rgb, dtype=int)
    colors = []
    for i, col in zip(x, rgb):
        colors.append([np.round(i, 3), "rgb({}, {}, {})".format(*col)])
    return {
        'colors': colors, 'vmin': vmin, 'vmax': vmax, 'cmap': our_cmap,
        'norm': norm, 'abs_threshold': abs_threshold,
        'symmetric_cmap': symmetric_cmap
    }


def encode(a):
    """Base64 encode a numpy array"""
    try:
        data = a.tobytes()
    except AttributeError:
        # np < 1.9
        data = a.tostring()
    return base64.b64encode(data).decode('utf-8')


def decode(b, dtype):
    """Decode a numpy array encoded as Base64"""
    return np.frombuffer(base64.b64decode(b.encode('utf-8')), dtype)


def mesh_to_plotly(mesh):
    mesh = surface.load_surf_mesh(mesh)
    x, y, z = map(encode, np.asarray(mesh[0].T, dtype='<f4'))
    i, j, k = map(encode, np.asarray(mesh[1].T, dtype='<i4'))
    info = {
        "_x": x,
        "_y": y,
        "_z": z,
        "_i": i,
        "_j": j,
        "_k": k,
    }
    return info


def to_color_strings(colors):
    cmap = mpl.colors.ListedColormap(colors)
    colors = cmap(np.arange(cmap.N))[:, :3]
    colors = np.asarray(colors * 255, dtype='uint8')
    colors = ['#{:02x}{:02x}{:02x}'.format(*row) for row in colors]
    return colors

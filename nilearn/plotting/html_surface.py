import os
import base64
import json
import cgi
import webbrowser
import tempfile
import collections

import numpy as np
import matplotlib as mpl
from matplotlib import cm as mpl_cm

from .._utils.extmath import fast_abs_percentile
from .._utils.param_validation import check_threshold
from .. import datasets, surface
from . import cm


def get_html_template(template_name):
    """Get an HTML file from package data"""
    template_path = os.path.join(
        os.path.dirname(__file__), 'data', 'html', template_name)
    with open(template_path, 'rb') as f:
        return f.read().decode('utf-8')


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


class SurfaceView(object):
    """
    Represents a web page.

    use str(document) or document.html to get a web page,
    document.get_iframe() to have it wrapped in an iframe.

    """

    def __init__(self, html, width=600, height=400):
        self.html = html
        self.width = width
        self.height = height
        self._temp_file = None

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
        escaped = cgi.escape(self.html, quote=True)
        wrapped = '<iframe srcdoc="{}" width={} height={}></iframe>'.format(
            escaped, width, height)
        return wrapped

    def get_standalone(self):
        """ Get the surface plot in an HTML page."""
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

    def open_in_browser(self, file_name=None):
        """
        Save the plot to a temporary HTML file and open it in a browser.
        """
        if file_name is None:
            fd, file_name = tempfile.mkstemp('.html', 'nilearn_surface_plot_')
            os.close(fd)
        self.save_as_html(file_name)
        self._temp_file = file_name
        file_size = os.path.getsize(file_name) / 1e6
        print(("Saved HTML in temporary file: {}\n"
               "file size is {:.1f}M, delete it when you're done, "
               "for example by calling this.remove_temp_file").format(
                   file_name, file_size))
        webbrowser.open(file_name)

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


def colorscale(cmap, values, threshold=None, symmetric_cmap=True):
    """Normalize a cmap, put it in plotly format, get threshold and range"""
    cmap = mpl_cm.get_cmap(cmap)
    abs_values = np.abs(values)
    if symmetric_cmap:
        vmax = abs_values.max()
        vmin = - vmax
    else:
        vmin, vmax = values.min(), values.max()
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
    return colors, vmin, vmax, our_cmap, norm, abs_threshold


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


def to_plotly(mesh):
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


def _to_color_strings(colors):
    colors = np.asarray(colors)
    colors = np.asarray(colors * 255, dtype='uint8')
    colors = ['#{:02x}{:02x}{:02x}'.format(*row) for row in colors]
    return colors


def _get_vertexcolor(surf_map, cmap, norm,
                     absolute_threshold=None, bg_map=None):
    vertexcolor = cmap(norm(surf_map).data)
    if absolute_threshold is None:
        return _to_color_strings(vertexcolor)
    if bg_map is None:
        bg_map = np.ones(len(surf_map)) * .5
        bg_vmin, bg_vmax = 0, 1
    else:
        bg_map = surface.load_surf_data(bg_map)
        bg_vmin, bg_vmax = np.min(bg_map), np.max(bg_map)
    bg_norm = mpl.colors.Normalize(vmin=bg_vmin, vmax=bg_vmax)
    bg_color = mpl_cm.get_cmap('Greys')(bg_norm(bg_map))
    vertexcolor[np.abs(surf_map) < absolute_threshold] = bg_color[
        np.abs(surf_map) < absolute_threshold]
    return _to_color_strings(vertexcolor)


def one_mesh_info(surf_map, surf_mesh, threshold=None, cmap=cm.cold_hot,
                  black_bg=False, bg_map=None, symmetric_cmap=True):
    """Prepare info for plotting one surface map on a single mesh."""
    info = {}
    colors, cmin, cmax, cmap, norm, abs_threshold = colorscale(
        cmap, surf_map, threshold, symmetric_cmap=symmetric_cmap)
    info['inflated_left'] = to_plotly(surf_mesh)
    info['vertexcolor_left'] = _get_vertexcolor(
        surf_map, cmap, norm, abs_threshold, bg_map)
    info["cmin"], info["cmax"] = float(cmin), float(cmax)
    info['black_bg'] = black_bg
    info['full_brain_mesh'] = False
    info['colorscale'] = colors
    return info


def _check_mesh(mesh):
    if not isinstance(mesh, str):
        assert isinstance(mesh, collections.Mapping)
        assert {'pial_left', 'pial_right', 'sulc_left', 'sulc_right',
                'infl_left', 'infl_right'}.issubset(mesh.keys())
        return mesh
    mesh = datasets.fetch_surf_fsaverage(mesh)
    return mesh


def full_brain_info(volume_img, mesh='fsaverage5', threshold=None,
                    cmap=cm.cold_hot, black_bg=False, symmetric_cmap=True,
                    vol_to_surf_kwargs={}):
    """Project 3d map on cortex; prepare info to plot both hemispheres."""
    info = {}
    mesh = _check_mesh(mesh)
    surface_maps = {
        h: surface.vol_to_surf(volume_img, mesh['pial_{}'.format(h)],
                               **vol_to_surf_kwargs)
        for h in ['left', 'right']
    }
    colors, cmin, cmax, cmap, norm, abs_threshold = colorscale(
        cmap, np.asarray(list(surface_maps.values())).ravel(), threshold,
        symmetric_cmap=symmetric_cmap)

    for hemi, surf_map in surface_maps.items():
        bg_map = surface.load_surf_data(mesh['sulc_{}'.format(hemi)])
        info['pial_{}'.format(hemi)] = to_plotly(mesh['pial_{}'.format(hemi)])
        info['inflated_{}'.format(hemi)] = to_plotly(
            mesh['infl_{}'.format(hemi)])

        info['vertexcolor_{}'.format(hemi)] = _get_vertexcolor(
            surf_map, cmap, norm, abs_threshold, bg_map)
    info["cmin"], info["cmax"] = float(cmin), float(cmax)
    info['black_bg'] = black_bg
    info['full_brain_mesh'] = True
    info['colorscale'] = colors
    return info


def _fill_html_template(info, embed_js=True):
    as_json = json.dumps(info)
    as_html = get_html_template('surface_plot_template.html').replace(
        'INSERT_STAT_MAP_JSON_HERE', as_json)
    as_html = add_js_lib(as_html, embed_js=embed_js)
    return SurfaceView(as_html)


def view_img_on_surf(stat_map_img, mesh='fsaverage5',
                     threshold=None, cmap=cm.cold_hot,
                     black_bg=False):
    """
    Insert a surface plot of a statistical map into an HTML page.

    Parameters
    ----------
    stat_map_img : Niimg-like object, 3d
        See http://nilearn.github.io/manipulating_images/input_output.html

    mesh : str or dict, optional.
        if 'fsaverage5', use fsaverage5 mesh from nilearn.datasets
        if 'fsaverage', use fsaverage mesh from nilearn.datasets
        if a dictionary, keys should be 'infl_left', 'pial_left', 'sulc_left',
        'infl_right', 'pial_right', 'sulc_right', containing inflated and
        pial meshes, and sulcal depth values for left and right hemispheres.

    threshold : str, number or None, optional (default=None)
        If None, no thresholding.
        If it is a number only values of amplitude greater
        than threshold will be shown.
        If it is a string it must finish with a percent sign,
        e.g. "25.3%", and only values of amplitude above the
        given percentile will be shown.

    cmap : str or matplotlib colormap, optional

    black_bg : bool, optional (default=False)

    Returns
    -------
    SurfaceView : plot of the stat map.
        It can be saved as an html page or rendered (transparently) by the
        Jupyter notebook. Useful methods are :
         - 'resize' to resize the plot displayed in a Jupyter notebook
         - 'save_as_html' to save the plot to a file
         - 'open_in_browser' to save the plot and open it in a web browser.

    """
    info = full_brain_info(
        volume_img=stat_map_img, mesh=mesh, threshold=threshold,
        cmap=cmap, black_bg=black_bg)
    return _fill_html_template(info, embed_js=True)


def view_surf(surf_mesh, surf_map=None, bg_map=None, threshold=None,
              cmap=cm.cold_hot, black_bg=False, symmetric_cmap=True):
    """
    Insert a surface plot of a surface map into an HTML page.

    Parameters
    ----------
    surf_mesh: str or list of two numpy.ndarray
        Surface mesh geometry, can be a file (valid formats are
        .gii or Freesurfer specific files such as .orig, .pial,
        .sphere, .white, .inflated) or
        a list of two Numpy arrays, the first containing the x-y-z coordinates
        of the mesh vertices, the second containing the indices
        (into coords) of the mesh faces.

    surf_map: str or numpy.ndarray, optional.
        Data to be displayed on the surface mesh. Can be a file (valid formats
        are .gii, .mgz, .nii, .nii.gz, or Freesurfer specific files such as
        .thickness, .curv, .sulc, .annot, .label) or
        a Numpy array

    bg_map: Surface data, optional,
        Background image to be plotted on the mesh underneath the
        surf_data in greyscale, most likely a sulcal depth map for
        realistic shading.

    threshold : str, number or None, optional (default=None)
        If None, no thresholding.
        If it is a number only values of amplitude greater
        than threshold will be shown.
        If it is a string it must finish with a percent sign,
        e.g. "25.3%", and only values of amplitude above the
        given percentile will be shown.

    cmap : str or matplotlib colormap, optional
        You might want to change it to 'gnist_ncar' if plotting a
        surface atlas.

    black_bg : bool, optional (default=False)

    symmetric_cmap : bool, optional (default=True)
        Make colormap symmetric (ranging from -vmax to vmax).
        Set it to False if you are plotting a surface atlas.

    Returns
    -------
    SurfaceView : plot of the stat map.
        It can be saved as an html page or rendered (transparently) by the
        Jupyter notebook. Useful methods are :
         - 'resize' to resize the plot displayed in a Jupyter notebook
         - 'save_as_html' to save the plot to a file
         - 'open_in_browser' to save the plot and open it in a web browser.

    """
    surf_mesh = surface.load_surf_mesh(surf_mesh)
    if surf_map is None:
        surf_map = np.ones(len(surf_mesh[0]))
    if surf_map is not None:
        surface.check_mesh_and_data(surf_mesh, surf_map)
    if bg_map is not None:
        surface.check_mesh_and_data(surf_mesh, bg_map)
    info = one_mesh_info(
        surf_map=surf_map, surf_mesh=surf_mesh, threshold=threshold,
        cmap=cmap, black_bg=black_bg, bg_map=bg_map,
        symmetric_cmap=symmetric_cmap)
    return _fill_html_template(info, embed_js=True)

import os
import base64
import json
import cgi

import numpy as np
import matplotlib as mpl
from matplotlib import cm

from .. import datasets, plotting
from . import surface

HTML_TEMPLATE = """

<html>

<head>
    <title>surface plot</title>
    <meta charset="UTF-8" />
    INSERT_JS_LIBRARIES_HERE
    <script>
        var surfaceMapInfo = INSERT_STAT_MAP_JSON_HERE;
        var colorscale = INSERT_COLORSCALE_HERE;

        function decodeBase64(encoded, dtype) {

            let getter = {
                "float32": "getFloat32",
                "int32": "getInt32"
            }[dtype];

            let arrayType = {
                "float32": Float32Array,
                "int32": Int32Array
            }[dtype];

            let raw = atob(encoded)
            let buffer = new ArrayBuffer(raw.length);
            let asIntArray = new Uint8Array(buffer);
            for (let i = 0; i !== raw.length; i++) {
                asIntArray[i] = raw.charCodeAt(i);
            }

            let view = new DataView(buffer);
            let decoded = new arrayType(
                raw.length / arrayType.BYTES_PER_ELEMENT);
            for (let i = 0, off = 0; i !== decoded.length;
                i++, off += arrayType.BYTES_PER_ELEMENT) {
                decoded[i] = view[getter](off, true);
            }
            return decoded;
        }

        function addPlot() {

            let hemisphere = $("#select-hemisphere").val();
            let kind = $("#select-kind").val();
            makePlot(kind, hemisphere,
                "surface-plot", display = null, erase = true);
        }

        function getLayout() {

            let camera = getCamera();
            let axisConfig = getAxisConfig();

            let layout = {
                width: $(window).width() * .8,
                height: $(window).outerHeight() * .8,
                hovermode: false,
                paper_bgcolor: surfaceMapInfo['black_bg'] ? '#000': '#fff',
                axis_bgcolor: '#333',
                scene: {
                    camera: camera,
                    xaxis: axisConfig,
                    yaxis: axisConfig,
                    zaxis: axisConfig
                }
            };

            return layout;

        }

        function getConfig() {
            let config = {
                modeBarButtonsToRemove: ["hoverClosest3d"],
                displayLogo: false
            };

            return config;
        }

        function getAxisConfig() {
            let axisConfig = {
                showgrid: false,
                showline: false,
                ticks: '',
                title: '',
                showticklabels: false,
                 zeroline: false,
                showspikes: false,
                spikesides: false
            };

            return axisConfig;
        }

        function getLighting() {
            return {
                "ambient": 0.5,
                "diffuse": 1,
                "fresnel": .1,
                "specular": .05,
                "roughness": .1,
                "facenormalsepsilon": 1e-6,
                "vertexnormalsepsilon": 1e-12
            };

        }

        function addColorbar(divId, layout, config) {
            // hack to get a colorbar
            let dummy = {
                "opacity": 0,
                "type": "mesh3d",
                "colorscale": colorscale,
                "x": [1, 0, 0],
                "y": [0, 1, 0],
                "z": [0, 0, 1],
                "i": [0],
                "j": [1],
                "k": [2],
                "intensity": [0.],
                "cmin": surfaceMapInfo["cmin"],
                "cmax": surfaceMapInfo["cmax"]
            };

            Plotly.plot(divId, [dummy], layout, config);

        }

        function getCamera() {
            let view = $("#select-view").val();
            if (view === "custom") {
                try {
                    return $("#surface-plot")[0].layout.scene.camera;
                } catch (e) {
                    return {};
                }
            }
            let cameras = {
                "left": {eye: {x: -2, y: 0, z: 0},
                         up: {x: 0, y: 0, z: 1},
                         center: {x: 0, y: 0, z: 0}},
                "right": {eye: {x: 2, y: 0, z: 0},
                          up: {x: 0, y: 0, z: 1},
                          center: {x: 0, y: 0, z: 0}},
                "top": {eye: {x: 0, y: 0, z: 2},
                        up: {x: 0, y: 1, z: 0},
                        center: {x: 0, y: 0, z: 0}},
                "bottom": {eye: {x: 0, y: 0, z: -2},
                           up: {x: 0, y: 1, z: 0},
                           center: {x: 0, y: 0, z: 0}},
                "front": {eye: {x: 0, y: 2, z: 0},
                          up: {x: 0, y: 0, z: 1},
                          center: {x: 0, y: 0, z: 0}},
                "back": {eye: {x: 0, y: -2, z: 0},
                         up: {x: 0, y: 0, z: 1},
                         center: {x: 0, y: 0, z: 0}},
            };

            return cameras[view];

        }


        function makePlot(surface, hemisphere, divId) {

            info = surfaceMapInfo[surface + "_" + hemisphere];

            info["type"] = "mesh3d";


            for (let attribute of ["x", "y", "z"]) {
                if (!(attribute in info)) {
                    info[attribute] = decodeBase64(
                        info["_" + attribute], "float32");
                }
            }

            for (let attribute of ["i", "j", "k"]) {
                if (!(attribute in info)) {
                    info[attribute] = decodeBase64(
                        info["_" + attribute], "int32");
                }
            }

            info["vertexcolor"] = surfaceMapInfo["vertexcolor_" + hemisphere];

            let data = [info];

            info['lighting'] = getLighting();
            let layout = getLayout();
            let config = getConfig();

            Plotly.react(divId, data, layout, config);

            addColorbar(divId, layout, config);

        }
    </script>
    <script>
        $(document).ready(
            function() {
                addPlot();
                $("#select-hemisphere").change(addPlot);
                $("#select-kind").change(addPlot);
                $("#select-view").change(addPlot);
                $("#surface-plot").mouseup(function() {
                    $("#select-view").val("custom");
                });
                $(window).resize(addPlot);

            });
    </script>
</head>

<body>
    <div id="surface-plot"></div>
    <select id="select-hemisphere">
<option value="left">Left hemisphere</option>
<option value="right">Right hemisphere</option>
</select>

    <select id="select-kind">
<option value="inflated">Inflated</option>
<option value="pial">Pial</option>
</select>
    <select id="select-view">
<option value="left">view: Left</option>
<option value="right">view: Right</option>
<option value="front">view: Front</option>
<option value="back">view: Back</option>
<option value="top">view: Top</option>
<option value="bottom">view: Bottom</option>
<option value="custom">view: -</option>
</select>

</body>

</html>

"""


def add_js_lib(html, embed_js=True):
    if not embed_js:
        js_lib = """
        <script
        src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js">
        </script>
        <script src="https://cdn.plot.ly/plotly-gl3d-latest.min.js"></script>
        """
    else:
        js_dir = os.path.join(os.path.dirname(__file__), 'data', 'js')
        with open(os.path.join(js_dir, 'jquery.min.js')) as f:
            jquery = f.read()
        with open(os.path.join(js_dir, 'plotly-gl3d-latest.min.js')) as f:
            plotly = f.read()
        js_lib = '<script>{}</script>\n<script>{}</script>'.format(
            jquery, plotly)
    return html.replace('INSERT_JS_LIBRARIES_HERE', js_lib)


class HTMLDocument(object):
    """
    Represents a web page.

    use str(document) or document.html to get a web page,
    document.iframe() to have it wrapped in an iframe.

    """

    def __init__(self, html, width=600, height=600):
        self.html = html
        self.width = width
        self.height = height

    def iframe(self, width=None, height=None):
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        escaped = cgi.escape(self.html, quote=True)
        wrapped = '<iframe srcdoc="{}" width={} height={}></iframe>'.format(
            escaped, width, height)
        return wrapped

    def standalone(self):
        return self.html

    def _repr_html_(self):
        return self.iframe()

    def __str__(self):
        return self.html


def colorscale(cmap, values, threshold=None):
    cmap = cm.get_cmap(cmap)
    abs_values = np.abs(values)
    abs_max = abs_values.max()
    norm = mpl.colors.Normalize(vmin=-abs_max, vmax=abs_max)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    abs_threshold = None
    if threshold is not None:
        abs_threshold = np.percentile(abs_values, threshold)
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
    return json.dumps(colors), abs_max, our_cmap, norm, abs_threshold


def _encode(a):
    return base64.b64encode(a.tobytes()).decode('utf-8')


def to_plotly(mesh):
    mesh = surface.load_surf_mesh(mesh)
    x, y, z = map(_encode, np.asarray(mesh[0].T, dtype='<f4'))
    i, j, k = map(_encode, np.asarray(mesh[1].T, dtype='<i4'))
    info = {
        "_x": x,
        "_y": y,
        "_z": z,
        "_i": i,
        "_j": j,
        "_k": k,
    }
    return info


def full_brain_info(stat_map=None, surface_maps=None, mesh='fsaverage5',
                    threshold=None, cmap=plotting.cm.cold_hot, black_bg=False):
    info = {}
    if mesh == 'fsaverage5':
        mesh = datasets.fetch_surf_fsaverage5()
    if surface_maps is None:
        assert stat_map is not None
        surface_maps = {
            h: surface.vol_to_surf(stat_map, mesh['pial_{}'.format(h)])
            for h in ['left', 'right']
        }
    colors, cmax, cmap, norm, at = colorscale(
        cmap, np.asarray(list(surface_maps.values())).ravel(), threshold)

    for hemi, surf_map in surface_maps.items():
        sulc_depth_map = surface.load_surf_data(mesh['sulc_{}'.format(hemi)])
        sulc_depth_map -= sulc_depth_map.min()
        sulc_depth_map /= sulc_depth_map.max()
        info['pial_{}'.format(hemi)] = to_plotly(mesh['pial_{}'.format(hemi)])
        info['inflated_{}'.format(hemi)] = to_plotly(
            mesh['infl_{}'.format(hemi)])
        vertexcolor = cmap(norm(surf_map).data)
        if threshold is not None:
            anat_color = cm.get_cmap('Greys')(sulc_depth_map)
            vertexcolor[np.abs(surf_map) < at] = anat_color[
                np.abs(surf_map) < at]
        vertexcolor = np.asarray(vertexcolor * 255, dtype='uint8')
        info['vertexcolor_{}'.format(hemi)] = [
            '#{:02x}{:02x}{:02x}'.format(*row) for row in vertexcolor
        ]
    info["cmin"], info["cmax"] = -cmax, cmax
    info['black_bg'] = black_bg
    return info, colors


def brain_to_html(stat_map=None, surface_maps=None, mesh='fsaverage5',
                  threshold=None, cmap=plotting.cm.cold_hot,
                  embed_js=True, output_file=None, black_bg=False):
    """
    Insert a surface plot of a statistical map into an HTML page.

    Parameters
    ----------
    stat_map : Niimg-like object, 3d, optional
        See http://nilearn.github.io/manipulating_images/input_output.html
        If None, surface_maps should be provided.

    surface_maps : dict, optional
       if specified, keys should be 'left' and 'right' and values should be
       1-d arrays containing intensities at mesh nodes in the left and right
       hemispheres. Ignored if stat_map is provided.

    mesh : str or dict, optional.
        if 'fsaverage5', use fsaverage5 mesh from nilearn.datasets
        if a dictionary, keys should be 'infl_left', 'pial_left', 'sulc_left',
        'infl_right', 'pial_right', 'sulc_right', containing inflated and
        pial meshes, and sulcal depth values for left and right hemispheres.

    threshold : int, optional
        int in [0, 100]: percentage of values to be thresholded.

    cmap : str or matplotlib colormap, optional

    embed_js : bool, optional (default=True)
        if True, jquery and plotly are embedded in resulting page.
        otherwise, they are loaded via CDNs.

    output_file : str, optional
        path to file in which resulting html page is written.

    black_bg : bool, optional (default=False)

    Returns
    -------
    HTMLDocument : html page containing a plot of the stat map.

    """
    info, colors = full_brain_info(
        stat_map=stat_map, surface_maps=surface_maps,
        mesh=mesh, threshold=threshold, cmap=cmap, black_bg=black_bg)
    as_json = json.dumps(info)
    as_html = HTML_TEMPLATE.replace('INSERT_STAT_MAP_JSON_HERE', as_json)
    as_html = as_html.replace('INSERT_COLORSCALE_HERE', colors)
    as_html = add_js_lib(as_html, embed_js=embed_js)
    if output_file is not None:
        with open(output_file, 'w') as f:
            f.write(as_html)
    return HTMLDocument(as_html)

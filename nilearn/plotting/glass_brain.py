"""
Brain schematics plotting for glass brain functionality
"""

import json
import os

from distutils.version import LooseVersion

import matplotlib
from matplotlib.path import Path
from matplotlib import patches
from matplotlib import colors
from matplotlib import transforms


def _codes_bezier(pts):
    bezier_num = len(pts)
    # Next two lines are meant to handle both Bezier 3 and 4
    path_attr = 'CURVE{0}'.format(bezier_num)
    codes = [getattr(Path, path_attr)] * (bezier_num - 1)
    return [Path.MOVETO] + codes


def _codes_segment(pts):
    return [Path.MOVETO, Path.LINETO]


def _codes(atype, pts):
    dispatch = {'bezier': _codes_bezier,
                'segment': _codes_segment}

    return dispatch[atype](pts)


def _invert_color(color):
    """Return inverted color

    If color is (R, G, B) it returns (1 - R, 1 - G, 1 - B). If
    'color' can not be converted to a color it is returned
    unmodified.

    """
    try:
        color_converter = colors.ColorConverter()
        color_rgb = color_converter.to_rgb(color)
        return tuple(1 - level for level in color_rgb)
    except ValueError:
        return color


def _get_mpl_patches(json_content, transform=None,
                     invert_color=False, **kwargs):
    """Walks over the json content and builds a list of matplotlib patches
    """
    mpl_patches = []
    kwargs_edgecolor = kwargs.pop('edgecolor', None)
    kwargs_linewidth = kwargs.pop('linewidth', None)
    for path in json_content['paths']:
        if kwargs_edgecolor is not None:
            edgecolor = kwargs_edgecolor
        else:
            edgecolor = path['edgecolor']
            if invert_color:
                edgecolor = _invert_color(edgecolor)
        linewidth = kwargs_linewidth or path['linewidth']
        path_id = path['id']

        for item in path['items']:
            type = item['type']
            pts = item['pts']
            codes = _codes(type, pts)
            path = Path(pts, codes)
            patch = patches.PathPatch(path,
                                      edgecolor=edgecolor,
                                      linewidth=linewidth,
                                      facecolor='none',
                                      gid=path_id,
                                      transform=transform,
                                      **kwargs)

            mpl_patches.append(patch)

    return mpl_patches


def _get_json_and_transform(direction):
    """Returns the json filename and and an affine transform, which has
    been tweaked by hand to fit the MNI template
    """
    direction_to_view_name = {'x': 'side',
                              'y': 'front',
                              'z': 'top',
                              'l': 'side',
                              'r': 'side'}

    direction_to_transform_params = {
        'x': [0.38, 0, 0, 0.38, -108, -70],
        'y': [0.39, 0, 0, 0.39, -72, -73],
        'z': [0.36, 0, 0, 0.37, -71, -107],
        'l': [0.38, 0, 0, 0.38, -108, -70],
        'r': [0.38, 0, 0, 0.38, -108, -70]}

    dirname = os.path.dirname(os.path.abspath(__file__))
    dirname = os.path.join(dirname, 'glass_brain_files')
    direction_to_filename = dict([
        (_direction, os.path.join(
            dirname,
            'brain_schematics_{0}.json'.format(view_name)))
        for _direction, view_name in direction_to_view_name.items()])

    direction_to_transforms = dict([
        (_direction, transforms.Affine2D.from_values(*params))
        for _direction, params in direction_to_transform_params.items()])

    direction_to_json_and_transform = dict([
        (_direction, (direction_to_filename[_direction],
                      direction_to_transforms[_direction]))
        for _direction in direction_to_filename])

    filename_and_transform = direction_to_json_and_transform.get(direction)

    if filename_and_transform is None:
        message = ("No glass brain view associated with direction '{0}'. "
                   "Possible directions are {1}").format(
                       direction,
                       list(direction_to_json_and_transform.keys()))
        raise ValueError(message)

    return filename_and_transform


def _get_object_bounds(json_content, transform):
    xmin, xmax, ymin, ymax = json_content['metadata']['bounds']
    x0, y0 = transform.transform((xmin, ymin))
    x1, y1 = transform.transform((xmax, ymax))

    xmin, xmax = min(x0, x1), max(x0, x1)
    ymin, ymax = min(y0, y1), max(y0, y1)

    # A combination of a proportional factor (fraction of the drawing)
    # and a guestimate of the linewidth
    xmargin = (xmax - xmin) * 0.025 + .1
    ymargin = (ymax - ymin) * 0.025 + .1
    return xmin - xmargin, xmax + xmargin, ymin - ymargin, ymax + ymargin


def plot_brain_schematics(ax, direction, **kwargs):
    """Creates matplotlib patches from a json custom format and plots them
       on a matplotlib Axes.

       Parameters
       ----------
           ax: a MPL axes instance
                The axes in which the plots will be drawn
            direction: {'x', 'y', 'z', 'l', 'r'}
                The directions of the view
            **kwargs:
                Passed to the matplotlib patches constructor

       Returns
       -------
       object_bounds: (xmin, xmax, ymin, ymax) tuple
           Useful for the caller to be able to set axes limits

    """
    if LooseVersion(matplotlib.__version__) >= LooseVersion("2.0"):
        get_axis_bg_color = ax.get_facecolor()
    else:
        get_axis_bg_color = ax.get_axis_bgcolor()

    black_bg = colors.colorConverter.to_rgba(get_axis_bg_color) \
                    == colors.colorConverter.to_rgba('k')

    json_filename, transform = _get_json_and_transform(direction)
    with open(json_filename) as json_file:
        json_content = json.loads(json_file.read())

    mpl_patches = _get_mpl_patches(json_content,
                                   transform=transform + ax.transData,
                                   invert_color=black_bg,
                                   **kwargs)

    for mpl_patch in mpl_patches:
        ax.add_patch(mpl_patch)

    object_bounds = _get_object_bounds(json_content, transform)

    return object_bounds

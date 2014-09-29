# coding: utf-8

import sys
import json

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib import patches
from matplotlib import colors

class JSONReader(object):
    """Reads path coordinates and metadata from a custom JSON format and
    can transform that into a list of matplotlib patches

    """
    def __init__(self, filename):
        self.filename = filename
        with open(self.filename) as f:
            json_content = f.read()

        self.json_content = json.loads(json_content)

    def _codes(self, type, pts):
        method = getattr(self, '_codes_{}'.format(type))
        return method(pts)

    def _codes_bezier(self, pts):
        bezier_num = len(pts)
        # Next two lines are meant to handle both Bezier 3 and 4
        path_attr = 'CURVE{}'.format(bezier_num)
        codes = [getattr(Path, path_attr)] * (bezier_num - 1)
        return [Path.MOVETO] + codes

    def _codes_segment(self, pts):
        return [Path.MOVETO, Path.LINETO]

    @staticmethod
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

    def to_mpl(self, transform=None, invert_color=False):
        """Returns a list of matplotlib patches
        """
        mpl_patches = []

        for path in self.json_content['paths']:
            edgecolor = path['edgecolor']
            if invert_color:
                edgecolor = JSONReader._invert_color(edgecolor)
            linewidth = path['linewidth']
            path_id = path['id']

            for item in path['items']:
                type = item['type']
                pts = item['pts']
                codes = self._codes(type, pts)
                path = Path(pts, codes)
                patch = patches.PathPatch(path,
                                          edgecolor=edgecolor,
                                          linewidth=linewidth,
                                          facecolor='none',
                                          gid=path_id,
                                          transform=transform)

                mpl_patches.append(patch)

        return mpl_patches

    def get_object_bounds(self):
        alist = self.json_content['metadata']['bounds']
        return tuple(alist)


class BrainPlotter(object):
    def __init__(self, json_filename, transform):
        self.json_filename = json_filename
        self.reader = JSONReader(self.json_filename)
        self.transform = transform

    def plot(self, ax, transform=None, invert_color=False):
        mpl_patches = self.reader.to_mpl(self.transform + ax.transData,
                                         invert_color)
        for mpl_patch in mpl_patches:
            ax.add_patch(mpl_patch)

    def get_object_bounds(self):
        xmin, xmax, ymin, ymax = self.reader.get_object_bounds()
        xmin, ymin = self.transform.transform((xmin, ymin))
        xmax, ymax = self.transform.transform((xmax, ymax))

        return xmin, xmax, ymin, ymax


if __name__ == '__main__':
    fig, ax = plt.subplots()
    ax.set_xlim((0, 1000))
    ax.set_ylim((0, 1000))
    plotter = BrainPlotter(sys.argv[1])
    plotter.plot(ax)
    plt.show()

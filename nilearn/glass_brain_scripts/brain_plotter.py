# coding: utf-8

import sys
import json

import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches


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

    def to_mpl(self, transform=None):
        """Returns a list of matplotlib patches
        """
        mpl_patches = []

        for path in self.json_content:
            edgecolor = path['edgecolor']
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
        pts = [pt for path in self.json_content for item in path['items']
               for pt in item['pts']]
        xmin = min(pt[0] for pt in pts)
        xmax = max(pt[0] for pt in pts)
        ymin = min(pt[1] for pt in pts)
        ymax = max(pt[1] for pt in pts)

        return xmin, xmax, ymin, ymax


class BrainPlotter(object):
    def __init__(self, json_filename, transform):
        self.json_filename = json_filename
        self.reader = JSONReader(self.json_filename)
        self.transform = transform

    def plot(self, ax, transform=None):
        mpl_patches = self.reader.to_mpl(self.transform + ax.transData)
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

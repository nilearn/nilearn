import json
import os

from matplotlib.path import Path
from matplotlib import patches
from matplotlib import colors
from matplotlib import transforms


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

    def to_mpl(self, transform=None, invert_color=False, **kwargs):
        """Returns a list of matplotlib patches
        """
        mpl_patches = []
        kwargs_edgecolor = kwargs.pop('edgecolor', None)
        kwargs_linewidth = kwargs.pop('linewidth', None)
        for path in self.json_content['paths']:
            if kwargs_edgecolor is not None:
                edgecolor = kwargs_edgecolor
            else:
                edgecolor = path['edgecolor']
                if invert_color:
                    edgecolor = JSONReader._invert_color(edgecolor)
            linewidth = kwargs_linewidth or path['linewidth']
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
                                          transform=transform,
                                          **kwargs)

                mpl_patches.append(patch)

        return mpl_patches

    def get_object_bounds(self):
        alist = self.json_content['metadata']['bounds']
        return tuple(alist)


class BrainSchematics(object):
    """Plot brain schematics on a matplotlib axis
    """
    def __init__(self, json_filename, transform):
        self.json_filename = json_filename
        self.reader = JSONReader(self.json_filename)
        self.transform = transform

    @classmethod
    def from_direction(cls, direction):
        json_filename, transform = cls._get_json_and_transform(direction)
        return cls(json_filename, transform)

    @staticmethod
    def _get_json_and_transform(direction):
        direction_to_view_name = {'x': 'side',
                                  'y': 'front',
                                  'z': 'top'}

        direction_to_transform_params = {
            'x': [0.38, 0, 0, 0.38, -108, -70],
            'y': [0.39, 0, 0, 0.39, -72, -73],
            'z': [0.36, 0, 0, 0.37, -71, -107]}

        dirname = os.path.dirname(os.path.abspath(__file__))
        dirname = os.path.join(dirname, 'glass_brain_files')
        direction_to_filename = {
            direction: os.path.join(
                dirname,
                'brain_schematics_{}.json'.format(view_name))
            for direction, view_name in direction_to_view_name.iteritems()}

        direction_to_transforms = {
            direction: transforms.Affine2D.from_values(*params)
            for direction, params in direction_to_transform_params.iteritems()}

        direction_to_json_and_transform = {
            direction: (direction_to_filename[direction],
                        direction_to_transforms[direction])
            for direction in direction_to_filename}

        filename_and_transform = direction_to_json_and_transform.get(direction)

        if filename_and_transform is None:
            message = ("No glass brain view associated with direction '{}'. "
                       "Possible directions are {}").format(
                           direction,
                           direction_to_json_and_transform.keys())
            raise ValueError(message)

        return filename_and_transform

    def plot(self, ax, transform=None, invert_color=False, **kwargs):
        mpl_patches = self.reader.to_mpl(self.transform + ax.transData,
                                         invert_color, **kwargs)
        for mpl_patch in mpl_patches:
            ax.add_patch(mpl_patch)

    def get_object_bounds(self):
        xmin, xmax, ymin, ymax = self.reader.get_object_bounds()
        xmin, ymin = self.transform.transform((xmin, ymin))
        xmax, ymax = self.transform.transform((xmax, ymax))

        xmargin = (xmax - xmin) * 0.05
        ymargin = (ymax - ymin) * 0.05
        return xmin - xmargin, xmax + xmargin, ymin - ymargin, ymax + ymargin


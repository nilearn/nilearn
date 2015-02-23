"""SVG to JSON converter.

The main assumption is that the SVG only contains Bezier curves and
segments. The output JSON is used for plotting glass brain schematics.
"""

import re
import sys
import json


class SVGToJSONConverter(object):
    """Reads an svg file and exports paths to a JSON format

    Only segments and Bezier curves are supported
    """
    def __init__(self, filename):
        self.filename = filename
        self.svg = svg.parse(filename)
        self.paths = self.svg.flatten()

    def _get_style_attr(self, style, attr):
        pat = r'{}:([^;]+)'.format(attr)
        match = re.search(pat, style)
        return match.group(1) if match is not None else None

    def _type_and_pts(self, obj):
        if isinstance(obj, svg.Bezier):
            my_type = 'bezier'
            pts = [p.coord() for p in obj.pts]
        elif isinstance(obj, svg.Segment):
            my_type = 'segment'
            pts = [p.coord() for p in (obj.start, obj.end)]
        else:
            msg = '{0} is not a supported class'.format(obj.__class__)
            raise TypeError(msg)

        # svg has its origin in the top left whereas
        # matplotlib has its origin at the bottom left
        # need to apply a mirror symmetry
        pt_min, pt_max = self.svg.bbox()
        y_range = pt_max.y - pt_min.y

        pts = [(x, y_range - y) for x, y in pts]

        return {'type': my_type, 'pts': pts}

    def _get_paths(self):
        result = []
        for path in self.paths:
            style = path.style
            edgecolor = self._get_style_attr(style, 'stroke')
            linewidth = float(self._get_style_attr(style, 'stroke-width'))
            path_id = path.id
            path_dict = {'edgecolor': edgecolor,
                         'linewidth': linewidth,
                         'id': path_id,
                         'items': []}

            # svg.MoveTo instances do not hold any information since they
            # just contain the first point of the next item
            filtered_items = [i for i in path.items
                              if not isinstance(i, svg.MoveTo)]
            for geom in filtered_items:
                path_dict['items'].append(self._type_and_pts(geom))

            result.append(path_dict)

        return result

    def _get_bounds(self, paths):
        points = [pt for path in paths for item in path['items']
                  for pt in item['pts']]
        x_coords = [pt[0] for pt in points]
        y_coords = [pt[1] for pt in points]

        xmin, xmax = min(x_coords), max(x_coords)
        ymin, ymax = min(y_coords), max(y_coords)

        return xmin, xmax, ymin, ymax

    def to_json(self):
        """Exports the svg paths into json.

        The json format looks like this:
        {
          "paths": [
            {
              "edgecolor": "#b3b3b3",
              "linewidth": 3.03045774,
              "id": "path3943",
              "items": [
                {
                  "pts": [
                    [ 571.83955, 751.5887290000001 ],
                    [ 571.57463, 750.8480390000001 ],
                    [ 571.44965, 747.969189 ],
                    [ 571.56178, 745.191269 ]
                  ],
                  "type": "bezier"
                },
                {
                  "pts": [
                    [ 566.41278, 705.415739 ],
                    [ 566.7642900000001, 696.532339 ]
                  ],
                  "type": "segment"
                },
                .
                .
                .
              ]
            },
            .
            .
            .
          ],
          "metadata": {
            "bounds": [
              1.3884929999999542, 398.60061299999995,
              -0.9977599999999711, 490.82066700000007
            ]
          }
        }
        """
        paths = self._get_paths()
        bounds = self._get_bounds(paths)
        metadata = {'bounds': bounds}
        result = {'metadata': metadata,
                  'paths': paths}

        return json.dumps(result, indent=2)

    def save_json(self, filename):
        json_content = self.to_json()

        with open(filename, 'w') as f:
            f.write(json_content)


def _import_svg():
    try:
        import svg
        return svg
    except ImportError as exc:
        exc.args += ('Could not import svg (https://github.com/cjlano/svg)'
                     ' which is required to parse the svg file', )
        raise

if __name__ == '__main__':
    svg = _import_svg()

    svg_filename = sys.argv[1]
    json_filename = sys.argv[2]
    converter = SVGToJSONConverter(svg_filename)
    converter.save_json(json_filename)

"""SVG to JSON converter.

Exports the coordinates of the stroked paths of an SVG
together with a few path attributes (id, stroke-color, stroke-width, etc.).

The output JSON is used for plotting glass brain schematics.

This only depends on the standard library.
The SVG paths must only be made of
(relative or absolute) moves, lines, cubic Bezier curves and closed paths,
and can only use translations or matrices as transforms.
Paths without a stroke are invisible and are ignored.

USAGE::

  python svg_to_json_converter.py input.svg output.json

By default the glass brain uses the transform tuned for the MNI template
to align the drawing on the image.
For other templates (for example non human brains),
the transform can be stored in the JSON with the ``--transform`` option
(the 6 parameters ``a b c d e f`` passed to
:class:`matplotlib.transforms.Affine2D`).
See ``maint_tools/fit_glass_brain_transform.py``
to find them.

EXAMPLE::

  python maint_tools/svg_to_json_converter.py \
    nilearn/plotting/glass_brain_files/brain_schematics_back.svg \
    foo.json

  python maint_tools/svg_to_json_converter.py \
    rat_brain.svg rat_brain.json \
    --transform 0.78 0 0 0.78 -36.5 25.3
"""

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

SVG_NAMESPACE: str = "{http://www.w3.org/2000/svg}"

IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

# number of coordinates for each supported path command
N_COORDINATES = {"m": 2, "l": 2, "c": 6}


def _get_style_attribute(style, attribute):
    """Return an attribute from the style of an SVG element (or None)."""
    match = re.search(f"{attribute}:([^;]+)", style)
    return match[1] if match is not None else None


def _parse_transform(transform):
    """Convert an SVG translate or matrix transform to (a, b, c, d, e, f)."""
    match = re.fullmatch(r"(translate|matrix)\(([^)]*)\)", transform.strip())
    if match is None:
        raise ValueError(f"Unsupported SVG transform: {transform}")
    values = [float(value) for value in re.split(r"[,\s]+", match[2].strip())]
    if match[1] == "translate":
        return (1.0, 0.0, 0.0, 1.0, values[0], values[1] if values[1:] else 0)
    return tuple(values)


def _compose(parent, child):
    """Return the transform that applies ``child`` and then ``parent``."""
    a1, b1, c1, d1, e1, f1 = parent
    a2, b2, c2, d2, e2, f2 = child
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def _iter_paths(element, matrix=IDENTITY):
    """Yield the paths of an SVG element with their transform."""
    transform = element.get("transform")
    if transform is not None:
        matrix = _compose(matrix, _parse_transform(transform))

    if element.tag == f"{SVG_NAMESPACE}path":
        yield element, matrix

    for child in element:
        yield from _iter_paths(child, matrix)


def _path_to_items(path_data, matrix):
    """Convert the data of an SVG path to segments and Bezier curves."""
    tokens = re.findall(
        r"[a-zA-Z]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", path_data
    )

    items = []
    current = start = (0.0, 0.0)
    command = None
    position = 0
    while position < len(tokens):
        if tokens[position].isalpha():
            command = tokens[position]
            position += 1
            if command in "zZ":
                if current != start:
                    items.append({"type": "segment", "pts": [current, start]})
                current = start
                continue
        if command is None or command.lower() not in N_COORDINATES:
            raise ValueError(f"Unsupported SVG path command: {command}")

        n = N_COORDINATES[command.lower()]
        values = [float(token) for token in tokens[position : position + n]]
        position += n
        points = list(zip(values[::2], values[1::2], strict=True))
        if command.islower():
            points = [(x + current[0], y + current[1]) for x, y in points]

        if command in "mM":
            current = start = points[0]
            # coordinates that follow a move are implicit line segments
            command = "l" if command == "m" else "L"
            continue
        if command in "lL":
            items.append({"type": "segment", "pts": [current, *points]})
        else:
            items.append({"type": "bezier", "pts": [current, *points]})
        current = points[-1]

    return [
        {
            "pts": [
                (
                    matrix[0] * x + matrix[2] * y + matrix[4],
                    matrix[1] * x + matrix[3] * y + matrix[5],
                )
                for x, y in item["pts"]
            ],
            "type": item["type"],
        }
        for item in items
    ]


def svg_to_json(svg_file, json_file, transform=None) -> None:
    """Convert the stroked paths of an SVG file to a glass brain JSON file.

    The JSON format looks like this:

    .. code-block:: json

        {
          "paths": [
            {
              "edgecolor": "#b3b3b3",
              "linewidth": 3.03045774,
              "id": "path3943",
              "items": [
                {
                  "pts": [
                    [571.83955, 751.5887290000001],
                    [571.57463, 750.8480390000001],
                    [571.44965, 747.969189],
                    [571.56178, 745.191269]
                  ],
                  "type": "bezier"
                },
                {
                  "pts": [
                    [566.41278, 705.415739],
                    [566.7642900000001, 696.532339]
                  ],
                  "type": "segment"
                }
              ]
            }
          ],
          "metadata": {
            "bounds": [
              1.3884929999999542, 398.60061299999995,
              -0.9977599999999711, 490.82066700000007
            ]
          }
        }

    Parameters
    ----------
    svg_file : :obj:`str` or :obj:`pathlib.Path`
        SVG file to convert.

    json_file : :obj:`str` or :obj:`pathlib.Path`
        JSON file to create.

    transform : sequence of 6 :obj:`float`, default=None
        The parameters ``(a, b, c, d, e, f)`` passed to
        :class:`matplotlib.transforms.Affine2D`:
        it maps the coordinates of the drawing
        to the coordinates of the template.
        If ``None``, the transform tuned for the MNI template is used.
    """
    root = ET.parse(svg_file).getroot()

    paths = []
    for path, matrix in _iter_paths(root):
        style = path.get("style", "")
        edgecolor = _get_style_attribute(style, "stroke")
        if edgecolor is None:
            continue
        paths.append(
            {
                "edgecolor": edgecolor,
                "linewidth": float(
                    _get_style_attribute(style, "stroke-width")
                ),
                "id": path.get("id"),
                "items": _path_to_items(path.get("d"), matrix),
            }
        )

    # SVG has its origin in the top left whereas matplotlib
    # has its origin at the bottom left:
    # mirror the y coordinates using the height of the drawing.
    y_values = [
        y for path in paths for item in path["items"] for _, y in item["pts"]
    ]
    y_range = max(y_values) - min(y_values)
    for path in paths:
        for item in path["items"]:
            item["pts"] = [(x, y_range - y) for x, y in item["pts"]]

    points = [
        pt for path in paths for item in path["items"] for pt in item["pts"]
    ]
    x_coordinates, y_coordinates = zip(*points, strict=True)
    metadata = {
        "bounds": [
            min(x_coordinates),
            max(x_coordinates),
            min(y_coordinates),
            max(y_coordinates),
        ]
    }
    if transform is not None:
        metadata["transform"] = list(transform)

    content = json.dumps(
        {"paths": paths, "metadata": metadata},
        indent=2,
        separators=(",", ": "),
    )
    Path(json_file).write_text(content + "\n")


def main():
    """Convert an SVG file to JSON from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("svg_file", type=Path, help="SVG file to convert.")
    parser.add_argument("json_file", type=Path, help="JSON file to create.")
    parser.add_argument(
        "--transform",
        type=float,
        nargs=6,
        metavar=("A", "B", "C", "D", "E", "F"),
        help="Transform used to align the drawing on the template "
        "(default: the transform tuned for the MNI template).",
    )
    args = parser.parse_args()

    svg_to_json(args.svg_file, args.json_file, args.transform)


if __name__ == "__main__":
    main()

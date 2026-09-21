#!/usr/bin/env python

# /// script
# requires-python = ">=3.11"
# dependencies = [
#    "templateflow",
#    "nilearn[plotting]>=0.12",
# ]
# ///

"""Fit the transform that aligns glass brain schematics on a brain template.

The schematics used by glass brain plots are stored as JSON files
that are drawn in their own (arbitrary) units.
The default transform is tuned by hand for the MNI template
and does not fit non human brains.

For each view (side, back, top),
this script finds the scale and the translation that maximize the overlap
(intersection over union) between
the outline of the schematic
and the silhouette of the brain mask of a template.
The result can be stored in the ``transform`` field
of the ``metadata`` of the JSON file,
which is then used by ``plot_brain_schematics``.

USAGE::

  python maint_tools/fit_glass_brain_transform.py \
    examples/07_advanced/glass_brain_files --template WHS

Add ``--write`` to update the JSON files (otherwise it only reports the fit).
The "front" view is not used by any direction of the glass brain,
so it gets the transform of the "back" view.

To check the result visually,
see ``maint_tools/plot_align_glass_brain_svg.py``.
"""

import argparse
import json
from math import comb
from pathlib import Path

import nibabel as nib
import numpy as np
from matplotlib.path import Path as MplPath
from scipy.optimize import minimize

# Axes of the template (x, y, z) displayed
# horizontally and vertically for each view.
VIEW_AXES = {"side": (1, 2), "back": (0, 2), "top": (0, 1)}

# Resolution (in template units) of the grid used to compute the overlap.
STEP = 0.1


def _get_brain_mask_coordinates(template, resolution):
    """Return the (x, y, z) coordinates of the voxels of the brain mask."""
    import templateflow.api as tflow

    mask_file = tflow.get(
        template,
        resolution=resolution,
        desc="brain",
        suffix="mask",
        atlas=None,
    )
    img = nib.load(mask_file)
    ijk = np.argwhere(np.asanyarray(img.dataobj) > 0)
    return nib.affines.apply_affine(img.affine, ijk)


def _sample_outline(json_file, n_samples=25):
    """Return points densely sampled along all the paths of a schematic."""
    with json_file.open() as f:
        paths = json.load(f)["paths"]

    t = np.linspace(0, 1, n_samples)[:, None]
    points = []
    for path in paths:
        for item in path["items"]:
            pts = np.array(item["pts"])
            if item["type"] == "segment":
                points.append(pts)
                continue
            n = len(pts) - 1  # Bezier curve of order n
            points.append(
                sum(
                    comb(n, k) * (1 - t) ** (n - k) * t**k * pts[k]
                    for k in range(n + 1)
                )
            )
    return np.vstack(points)


def _fit_view(outline, mask_xyz, axes):
    """Fit scale and translation of an outline on a brain silhouette.

    Returns
    -------
    (a, b, c, d, e, f) transform parameters in the order expected by
    matplotlib.transforms.Affine2D.from_values, and the best overlap.
    """
    silhouette_pts = np.unique(
        np.round(mask_xyz[:, axes] / STEP).astype(int), axis=0
    )
    pad = int(4 / STEP)
    origin = silhouette_pts.min(0) - pad
    shape = silhouette_pts.max(0) - origin + pad + 1
    silhouette = np.zeros(shape, dtype=bool)
    silhouette[tuple((silhouette_pts - origin).T)] = True
    silhouette = silhouette.ravel()

    grid_h, grid_v = np.meshgrid(
        (np.arange(shape[0]) + origin[0]) * STEP,
        (np.arange(shape[1]) + origin[1]) * STEP,
        indexing="ij",
    )
    grid = np.c_[grid_h.ravel(), grid_v.ravel()]

    def neg_overlap(params):
        sx, sy, tx, ty = params
        inside = MplPath(outline * [sx, sy] + [tx, ty]).contains_points(grid)
        return -(inside & silhouette).sum() / max(
            (inside | silhouette).sum(), 1
        )

    # initialize by matching the bounding boxes
    lower, upper = silhouette_pts.min(0) * STEP, silhouette_pts.max(0) * STEP
    scale = (upper - lower) / (outline.max(0) - outline.min(0))
    translation = lower - outline.min(0) * scale
    result = minimize(
        neg_overlap,
        [*scale, *translation],
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 600},
    )
    sx, sy, tx, ty = (float(x) for x in result.x)
    transform = [round(sx, 4), 0, 0, round(sy, 4), round(tx, 2), round(ty, 2)]
    return transform, -float(result.fun)


def _write_transform(json_file, transform):
    raw = json_file.read_text()
    content = json.loads(raw)
    content["metadata"]["transform"] = transform
    json_file.write_text(
        json.dumps(content, indent=2, separators=(",", ": "))
        + ("\n" if raw.endswith("\n") else "")
    )


def main():
    """Fit the transforms and optionally store them in the JSON files."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "assets",
        type=Path,
        help="Folder with the brain_schematics_<view>.json files.",
    )
    parser.add_argument("--template", default="WHS", help="TemplateFlow name.")
    parser.add_argument(
        "--resolution", type=int, default=2, help="TemplateFlow resolution."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Store the transforms in the metadata of the JSON files.",
    )
    args = parser.parse_args()

    mask_xyz = _get_brain_mask_coordinates(args.template, args.resolution)

    transforms = {}
    for view, axes in VIEW_AXES.items():
        json_file = args.assets / f"brain_schematics_{view}.json"
        outline = _sample_outline(json_file)
        transforms[view], overlap = _fit_view(outline, mask_xyz, list(axes))
        print(f"{view}: {transforms[view]} (overlap: {overlap:.3f})")

    transforms["front"] = transforms["back"]

    if args.write:
        for view, transform in transforms.items():
            _write_transform(
                args.assets / f"brain_schematics_{view}.json", transform
            )


if __name__ == "__main__":
    main()

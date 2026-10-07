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

  python maint_tools/fit_glass_brain_transform.py path/to/json --template WHS

The folder must contain the brain_schematics_<view>.json files:
they can be generated from SVG files
with ``maint_tools/svg_to_json_converter.py``.

Add ``--write`` to update the JSON files (otherwise it only reports the fit).

The same can be done from python
with :func:`fit_transforms` and :func:`write_transforms`.

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

from nilearn.image import load_img

# Axes of the template (x, y, z) displayed
# horizontally and vertically for each view.
VIEW_AXES = {"side": (1, 2), "back": (0, 2), "top": (0, 1)}

# Resolution (in template units) of the grid used to compute the overlap:
# coarser grids are faster but give less stable fits.
STEP: float = 0.1

# Number of points sampled on each curve of the outline.
N_SAMPLES: int = 4


def _get_brain_mask_coordinates(brain_mask):
    """Return the (x, y, z) coordinates of the voxels of the brain mask."""
    img = load_img(brain_mask)
    ijk = np.argwhere(np.asanyarray(img.dataobj) > 0)
    return nib.affines.apply_affine(img.affine, ijk)


def _sample_outline(json_file: Path, n_samples: int = N_SAMPLES) -> np.ndarray:
    """Return points densely sampled along all the paths of a schematic."""
    with json_file.open(encoding="utf8") as f:
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


def fit_transforms(json_folder: str | Path, brain_mask, verbose: bool = False):
    """Fit the transform of each view of a set of glass brain schematics.

    Parameters
    ----------
    json_folder : :obj:`str` or :obj:`pathlib.Path`
        Folder with the ``brain_schematics_<view>.json`` files
        for the side, back and top views.

    brain_mask : Niimg-like object
        Brain mask of the template on which to align the schematics.

    verbose : :obj:`bool`, default=False
        If ``True``, report the overlap obtained for each view.

    Returns
    -------
    transforms : :obj:`dict` of :obj:`list`
        For each view, the 6 parameters ``(a, b, c, d, e, f)``
        passed to :class:`matplotlib.transforms.Affine2D`.
    """
    mask_xyz = _get_brain_mask_coordinates(brain_mask)

    transforms = {}
    for view, axes in VIEW_AXES.items():
        outline = _sample_outline(
            Path(json_folder) / f"brain_schematics_{view}.json"
        )
        transforms[view], overlap = _fit_view(outline, mask_xyz, list(axes))
        if verbose:
            print(f"{view}: {transforms[view]} (overlap: {overlap:.3f})")

    return transforms


def write_transforms(json_folder, transforms):
    """Store transforms in the metadata of the glass brain JSON files.

    Parameters
    ----------
    json_folder : :obj:`str` or :obj:`pathlib.Path`
        Folder with the ``brain_schematics_<view>.json`` files.

    transforms : :obj:`dict` of :obj:`list`
        Transform of each view, as returned by :func:`fit_transforms`.
    """
    for view, transform in transforms.items():
        _write_transform(
            Path(json_folder) / f"brain_schematics_{view}.json", transform
        )


def _write_transform(json_file: Path, transform) -> None:
    raw = json_file.read_text()
    content = json.loads(raw)
    content["metadata"]["transform"] = transform
    json_file.write_text(
        json.dumps(content, indent=2, separators=(",", ": "))
        + ("\n" if raw.endswith("\n") else "")
    )


def main():
    """Fit the transforms and optionally store them in the JSON files."""
    import templateflow.api as tflow

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

    brain_mask = tflow.get(
        args.template,
        resolution=args.resolution,
        desc="brain",
        suffix="mask",
        atlas=None,
    )
    transforms = fit_transforms(args.assets, brain_mask, verbose=True)

    if args.write:
        write_transforms(args.assets, transforms)


if __name__ == "__main__":
    main()

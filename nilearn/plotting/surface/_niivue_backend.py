"""Functions specific to "niivue" backend for surface visualization."""

import base64
import warnings
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from nilearn._utils.extmath import fast_abs_percentile
from nilearn._utils.logger import find_stack_level
from nilearn._utils.param_validation import check_threshold
from nilearn.plotting import cm

from nilearn.surface.surface import _data_to_gifti, _mesh_to_gifti


def colorscale_niivue(values, vmax, threshold=None):
    """Normalize a cmap, put it in plotly format, get threshold and range."""
    if vmax is None:
        abs_values = np.abs(values)
        vmax = abs_values.max()
    vmax = float(vmax)

    if threshold is not None:
        threshold = check_threshold(threshold, values, fast_abs_percentile)

    return vmax, threshold


def matplotlib_cm_to_niivue_cm(
    cmap: str | mpl.colors.Colormap,
) -> None | dict[str, dict[str, list[int]]]:
    """Convert matplotlib colormap to niivue colormap.

    Parameters
    ----------
    cmap_name : str or Colormap
        Name of the colormap to convert.

    Returns
    -------
    cmap
        Converted positive "pos" and negative "neg" colormaps,
        with keys "R", "G", "B", "A".
    """
    if not isinstance(cmap, (mpl.colors.Colormap, str)):
        warnings.warn(
            f"'cmap' must be a str or a Colormap. Got {type(cmap)}",
            stacklevel=find_stack_level(),
        )
        return None

    name = None
    spec = None
    reverse = False

    if isinstance(cmap, str):
        name = cmap
        spec = plt.get_cmap(name)
    else:
        spec = cmap
        name = cmap.name

    if name.endswith("_r"):
        name = name[:-2]
        reverse = True

    n_nodes = 255
    colors = spec(np.linspace(0, 1, 2 * n_nodes))

    js: dict[str, list[int]] = {"R": [], "G": [], "B": [], "A": []}
    js["R"] = (255 * colors[..., 0]).astype(int).tolist()
    js["G"] = (255 * colors[..., 1]).astype(int).tolist()
    js["B"] = (255 * colors[..., 2]).astype(int).tolist()
    js["A"] = [64 for _ in range(2 * n_nodes)]

    if reverse:
        for k in js:
            js[k] = js[k][::-1]

    js_pos = {k: v[n_nodes:] for k, v in js.items()}
    js_neg = {k: v[:n_nodes][::-1] for k, v in js.items()}

    return {"pos": js_pos, "neg": js_neg}


def _one_mesh_info(
    surf_map,
    surf_mesh,
    threshold=None,
    cmap=cm.cold_hot,  # type: ignore[attr-defined]
    black_bg: bool = False,
    bg_map=None,
    symmetric_cmap: bool = True,
    bg_on_data: bool = False,
    vmax=None,
    vmin=None,
    engine: Literal["niivue", "plotly"] = "plotly",
    **colorbar_kwargs,
) -> dict[str, Any]:
    """Prepare info for plotting one surface map on a single mesh.

    This computes the dictionary that gets inserted in the web page,
    which contains the encoded mesh, colors, min and max values, and
    background color.

    """
    info: dict[str, Any] = {}

    # Handle mesh
    surf_mesh_gifti = _mesh_to_gifti(
        surf_mesh.coordinates, surf_mesh.faces
    )
    info["surf_mesh"] = base64.b64encode(
        surf_mesh_gifti.to_bytes()
    ).decode("UTF-8")

    # Handle surface data
    gii = _data_to_gifti(surf_map)
    info["surf_map"] = base64.b64encode(gii.to_bytes()).decode("UTF-8")

    info["cmap"] = matplotlib_cm_to_niivue_cm(cmap)

    vmax, threshold = colorscale_niivue(surf_map, vmax, threshold)
    info["threshold"] = threshold
    info["vmax"] = vmax

    # Handle background map
    if bg_map is not None:
        gii = _data_to_gifti(bg_map)
        info["bg_map"] = base64.b64encode(gii.to_bytes()).decode("UTF-8")
    else:
        info["bg_map"] = "null"

    info["bg_color"] = "[0, 0, 0, 1]" if black_bg else "[1, 1, 1, 1]"
    info["bg_theme"] = "black" if black_bg else "white"

    info["colorbar"] = str(colorbar_kwargs.get("colorbar", True)).lower()

    return info

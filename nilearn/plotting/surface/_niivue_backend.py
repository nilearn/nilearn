"""Functions specific to "niivue" backend for surface visualization."""

import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from nilearn._utils.extmath import fast_abs_percentile
from nilearn._utils.logger import find_stack_level
from nilearn._utils.param_validation import check_threshold


def colorscale_niivue(values, vmax, threshold=None):
    """Normalize a cmap, put it in plotly format, get threshold and range."""
    if vmax is None:
        abs_values = np.abs(values)
        vmax = abs_values.max()
    vmax = float(vmax)

    if threshold is not None:
        threshold = check_threshold(threshold, values, fast_abs_percentile)
        return vmax, float(threshold)

    return vmax, threshold


def matplotlib_cm_to_niivue_cm(cmap):
    """Convert matplotlib colormap to niivue colormap.

    Parameters
    ----------
    cmap_name : str or Colormap
        Name of the colormap to convert.

    Returns
    -------
    cmap : dict of dict of list
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

    js = {"R": [], "G": [], "B": [], "A": []}
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

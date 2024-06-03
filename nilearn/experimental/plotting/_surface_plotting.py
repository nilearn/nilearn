from __future__ import annotations

from nilearn import plotting as old_plotting
from nilearn.experimental.surface import SurfaceImage


def plot_surf(
    img, part: str | None = None, mesh=None, view: str | None = None, **kwargs
):
    """Plot a SurfaceImage.

    TODO: docstring.
    """
    if not isinstance(img, SurfaceImage):
        return old_plotting.plot_surf(
            surf_mesh=mesh,
            surf_map=img,
            hemi=part,
            **kwargs,
        )

    if mesh is None:
        mesh = img.mesh
    if part is None:
        # only take the first hemisphere by default
        part = list(img.data.parts.keys())[0]
    if view is None:
        view = "lateral"

    return old_plotting.plot_surf(
        surf_mesh=mesh.parts[part],
        surf_map=img.data.parts[part],
        hemi=part,
        view=view,
        **kwargs,
    )

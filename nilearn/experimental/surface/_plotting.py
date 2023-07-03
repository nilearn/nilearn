from typing import Optional, Sequence

from matplotlib import pyplot as plt

from nilearn import plotting
from nilearn.experimental.surface._surface_image import PolyMesh, SurfaceImage


def plot_surf_img(
    img: SurfaceImage,
    parts: Optional[Sequence[str]] = None,
    mesh: Optional[PolyMesh] = None,
    **kwargs,
) -> plt.Figure:
    if mesh is None:
        mesh = img.mesh
    if parts is None:
        parts = list(img.data.keys())
    fig, axes = plt.subplots(
        1,
        len(parts),
        subplot_kw={"projection": "3d"},
        figsize=(4 * len(parts), 4),
    )
    for ax, mesh_part in zip(axes, parts):
        plotting.plot_surf(
            mesh[mesh_part],
            img.data[mesh_part],
            axes=ax,
            title=mesh_part,
            **kwargs,
        )
    assert isinstance(fig, plt.Figure)
    return fig

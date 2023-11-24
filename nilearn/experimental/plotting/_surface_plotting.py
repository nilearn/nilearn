from nilearn import plotting as old_plotting
from matplotlib import pyplot as plt


def plot_surf(img, parts=None, mesh=None, views=["lateral"], **kwargs):
    if mesh is None:
        mesh = img.mesh
    if parts is None:
        parts = list(img.data.keys())
    fig, axes = plt.subplots(
        len(views),
        len(parts),
        subplot_kw={"projection": "3d"},
        figsize=(4 * len(parts), 4),
    )
    for ax, mesh_part in zip(axes, parts):
        old_plotting.plot_surf(
            mesh[mesh_part],
            img.data[mesh_part],
            axes=ax,
            title=mesh_part,
            **kwargs,
        )
    return fig

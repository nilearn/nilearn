#!/usr/bin/env python
"""The goal of this script is to align the glass brain SVGs on top of the
anatomy.

This is only useful for internal purposes especially when the SVG is modified.
"""

from nilearn.plotting.glass_brain import plot_brain_schematics
from nilearn.plotting.image import plot_anat, plot_glass_brain, show
from nilearn.plotting.image.utils import load_anat

if __name__ == "__main__":
    # plotting anat for coarse alignment
    bg_img, _, _, _ = load_anat()
    plot_glass_brain(bg_img, threshold=0, black_bg=True, title="anat", alpha=1)
    plot_glass_brain(
        bg_img,
        threshold=0,
        black_bg=True,
        title="anat",
        alpha=1,
        display_mode="ortho",
    )
    plot_glass_brain(bg_img, threshold=0, title="anat", alpha=1)

    # checking hemispheres plotting
    plot_glass_brain(
        bg_img,
        threshold=0,
        black_bg=True,
        title="anat",
        alpha=1,
        display_mode="lyrz",
    )

    def add_brain_schematics(display):
        """Plot slices for finer alignment.

        e.g. parieto-occipital sulcus
        """
        for axes in display.axes.values():
            kwargs = {"alpha": 0.5, "linewidth": 1, "edgecolor": "orange"}
            object_bounds = plot_brain_schematics(
                axes.ax, axes.direction, **kwargs
            )
            axes.add_object_bounds(object_bounds)

    # side
    display = plot_anat(display_mode="x", cut_coords=[-2])
    add_brain_schematics(display)

    # top
    display = plot_anat(display_mode="z", cut_coords=[20])
    add_brain_schematics(display)

    # front
    display = plot_anat(display_mode="y", cut_coords=[-20])
    add_brain_schematics(display)

    # all in one
    display = plot_anat(display_mode="ortho", cut_coords=(-2, -20, 20))
    add_brain_schematics(display)

    # Plot multiple slices
    display = plot_anat(display_mode="x")
    add_brain_schematics(display)

    display = plot_anat(display_mode="y")
    add_brain_schematics(display)

    display = plot_anat(display_mode="z")
    add_brain_schematics(display)

    show()

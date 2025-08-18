"""Functions to compare volume or surface images."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from scipy import stats

from nilearn import DEFAULT_SEQUENTIAL_CMAP
from nilearn._utils.docs import fill_doc
from nilearn._utils.helpers import constrained_layout_kwargs
from nilearn._utils.logger import find_stack_level
from nilearn._utils.masker_validation import (
    check_compatibility_mask_and_images,
)
from nilearn._utils.niimg_conversions import check_niimg_3d
from nilearn.maskers import NiftiMasker, SurfaceMasker
from nilearn.plotting._utils import save_figure_if_needed
from nilearn.surface.surface import SurfaceImage
from nilearn.surface.utils import check_polymesh_equal
from nilearn.typing import NiimgLike


@fill_doc
def plot_img_comparison(
    ref_imgs,
    src_imgs,
    masker=None,
    plot_hist=True,
    log=True,
    ref_label="image set 1",
    src_label="image set 2",
    output_dir=None,
    axes=None,
    colorbar=True,
):
    """Create plots to compare two lists of images and measure correlation.

    The first plot displays linear correlation between :term:`voxel` values.
    The second plot superimposes histograms to compare values distribution.

    Parameters
    ----------
    ref_img : 3D Niimg-like object or :obj:`~nilearn.surface.SurfaceImage` \
              or a :obj:`list` of \
              3D Niimg-like object or :obj:`~nilearn.surface.SurfaceImage`
        Reference image.

    src_img : 3D Niimg-like object or :obj:`~nilearn.surface.SurfaceImage` \
              or a :obj:`list` of \
              3D Niimg-like object or :obj:`~nilearn.surface.SurfaceImage`
        Source image.
        Its type must match that of the ``ref_img``.
        If the source image is Niimg-Like,
        it will be resampled to match that or the source image.

    masker : 3D Niimg-like binary mask or \
            :obj:`~nilearn.maskers.NiftiMasker` or \
            binary :obj:`~nilearn.surface.SurfaceImage` or \
            or :obj:`~nilearn.maskers.SurfaceMasker` or \
            None, default = None
        Mask to be used on data.
        Its type must be compatible with that of the ``ref_img``.
        If ``None`` is passed,
        an appropriate masker will be fitted
        on the first reference image.

    plot_hist : :obj:`bool`, default=True
        If True then histograms of each img in ref_imgs will be plotted
        along-side the histogram of the corresponding image in src_imgs.

    log : :obj:`bool`, default=True
        Passed to plt.hist.

    ref_label : :obj:`str`, default='image set 1'
        Name of reference images.

    src_label : :obj:`str`, default='image set 2'
        Name of source images.

    output_dir : :obj:`str` or None, default=None
        Directory where plotted figures will be stored.

    axes : :obj:`list` of two matplotlib Axes objects, or None, default=None
        Can receive a list of the form [ax1, ax2] to render the plots.
        By default new axes will be created.

    %(colorbar)s
        default=True

    Returns
    -------
    corrs : :class:`numpy.ndarray`
        Pearson correlation between the images.

    """
    # Cast to list
    if isinstance(ref_imgs, (*NiimgLike, SurfaceImage)):
        ref_imgs = [ref_imgs]
    if isinstance(src_imgs, (*NiimgLike, SurfaceImage)):
        src_imgs = [src_imgs]
    if not isinstance(ref_imgs, list) or not isinstance(src_imgs, list):
        raise TypeError(
            "'ref_imgs' and 'src_imgs' "
            "must both be list of 3D Niimg-like or SurfaceImage.\n"
            f"Got {type(ref_imgs)=} and {type(src_imgs)=}."
        )

    if all(isinstance(x, NiimgLike) for x in ref_imgs) and all(
        isinstance(x, NiimgLike) for x in src_imgs
    ):
        image_type = "volume"

    elif all(isinstance(x, SurfaceImage) for x in ref_imgs) and all(
        isinstance(x, SurfaceImage) for x in src_imgs
    ):
        image_type = "surface"
    else:
        types_ref_imgs = {type(x) for x in ref_imgs}
        types_src_imgs = {type(x) for x in src_imgs}
        raise TypeError(
            "'ref_imgs' and 'src_imgs' "
            "must both be list of only 3D Niimg-like or SurfaceImage.\n"
            f"Got {types_ref_imgs=} and {types_src_imgs=}."
        )

    masker = _sanitize_masker(masker, image_type, ref_imgs[0])

    corrs = []

    for i, (ref_img, src_img) in enumerate(zip(ref_imgs, src_imgs)):
        if axes is None:
            fig, (ax1, ax2) = plt.subplots(
                1,
                2,
                figsize=(12, 5),
                **constrained_layout_kwargs(),
            )
        else:
            (ax1, ax2) = axes
            fig = ax1.get_figure()

        ref_data, src_data = _extract_data_2_images(
            ref_img, src_img, masker=masker
        )

        if ref_data.shape != src_data.shape:
            warnings.warn(
                "Images are not shape-compatible",
                stacklevel=find_stack_level(),
            )
            return

        corr = stats.pearsonr(ref_data, src_data)[0]
        corrs.append(corr)

        # when plot_hist is False creates two empty axes
        # and doesn't plot anything
        if plot_hist:
            gridsize = 100

            lims = [
                np.min(ref_data),
                np.max(ref_data),
                np.min(src_data),
                np.max(src_data),
            ]

            hb = ax1.hexbin(
                ref_data,
                src_data,
                bins="log",
                cmap=DEFAULT_SEQUENTIAL_CMAP,
                gridsize=gridsize,
                extent=lims,
            )

            if colorbar:
                cb = fig.colorbar(hb, ax=ax1)
                cb.set_label("log10(N)")

            x = np.linspace(*lims[:2], num=gridsize)

            ax1.plot(x, x, linestyle="--", c="grey")
            ax1.set_title(f"Pearson's R: {corr:.2f}")
            ax1.grid("on")
            ax1.set_xlabel(ref_label)
            ax1.set_ylabel(src_label)
            ax1.axis(lims)

            ax2.hist(
                ref_data, alpha=0.6, bins=gridsize, log=log, label=ref_label
            )
            ax2.hist(
                src_data, alpha=0.6, bins=gridsize, log=log, label=src_label
            )

            ax2.set_title("Histogram of imgs values")
            ax2.grid("on")
            ax2.legend(loc="best")

            output_file = (
                output_dir / f"{int(i):04}.png" if output_dir else None
            )
            save_figure_if_needed(ax1, output_file)

    return corrs


@fill_doc
def plot_bland_altman(
    ref_img,
    src_img,
    masker=None,
    ref_label="reference image",
    src_label="source image",
    figure=None,
    title=None,
    cmap=DEFAULT_SEQUENTIAL_CMAP,
    colorbar=True,
    gridsize=100,
    lims=None,
    output_file=None,
):
    """Create a Bland-Altman plot between 2 images.

    Plot the the 2D distribution of voxel-wise differences
    as a function of the voxel-wise mean,
    along with an histogram for the distribution of each.

    .. note::

        Bland-Altman plots show
        the difference between the statistic values (y-axis)
        against the mean statistic value (x-axis) for all voxels.

        The plots provide an assessment of the level of agreement
        between two images about the magnitude of the statistic value
        observed at each voxel.

        If two images were in perfect agreement,
        all points on the Bland-Altman plot would lie on the x-axis,
        since the difference between the statistic values
        at each voxel would be zero.

        The degree of disagreement is therefore evaluated
        by the perpendicular distance of points from the x-axis.

    Parameters
    ----------
    ref_img : 3D Niimg-like object or :obj:`~nilearn.surface.SurfaceImage`
        Reference image.

    src_img : 3D Niimg-like object or :obj:`~nilearn.surface.SurfaceImage`
        Source image.
        Its type must match that of the ``ref_img``.
        If the source image is Niimg-Like,
        it will be resampled to match that or the source image.

    masker : 3D Niimg-like binary mask or \
            :obj:`~nilearn.maskers.NiftiMasker` or \
            binary :obj:`~nilearn.surface.SurfaceImage` or \
            or :obj:`~nilearn.maskers.SurfaceMasker` or \
            None, default = None
        Mask to be used on data.
        Its type must be compatible with that of the ``ref_img``.
        If ``None`` is passed,
        an appropriate masker will be fitted on the reference image.

    ref_label : :obj:`str`, default='reference image'
        Name of reference image.

    src_label : :obj:`str`, default='source image'
        Name of source image.

    %(figure)s

    %(title)s

    %(cmap)s
        default="inferno"

    %(colorbar)s
        default=True

    gridsize : :obj:`int` or :obj:`tuple` of 2 :obj:`int`, default=100
        Dimension of the grid on which to display the main plot.
        If a single value is passed, then the grid is square.
        If a tuple is passed, the first value corresponds
        to the length of the x axis,
        and the second value corresponds to the length of the y axis.

    lims : A :obj:`list` or :obj:`tuple` of 4 :obj:`int` or None, default=None
        Determines the limit the central hexbin plot
        and the marginal histograms.
        Values in the list or tuple are: [-lim_x, lim_x, -lim_y, lim_y].
        If ``None`` is passed values are determined based on the data.

    %(output_file)s

    Notes
    -----
    This function and the plot description was adapted
    from :footcite:t:`Bowring2019`
    and its associated `code base <https://github.com/AlexBowring/Software_Comparison/blob/master/figures/lib/bland_altman.py>`_.


    References
    ----------

    .. footbibliography::

    """
    data_ref, data_src = _extract_data_2_images(
        ref_img, src_img, masker=masker
    )

    mean = np.mean([data_ref, data_src], axis=0)
    diff = data_ref - data_src

    if lims is None:
        lim_x = np.max(np.abs(mean))
        if lim_x == 0:
            lim_x = 1
        lim_y = np.max(np.abs(diff))
        if lim_y == 0:
            lim_y = 1
        lims = [-lim_x, lim_x, -lim_y, lim_y]

    if (
        not isinstance(lims, (list, tuple))
        or len(lims) != 4
        or any(x == 0 for x in lims)
    ):
        raise TypeError(
            "'lims' must be a list or tuple of length == 4, "
            "with all values different from 0."
        )

    if isinstance(gridsize, int):
        gridsize = (gridsize, gridsize)

    if figure is None:
        figure = plt.figure(figsize=(6, 6))

    gs0 = gridspec.GridSpec(1, 1)

    gs = gridspec.GridSpecFromSubplotSpec(
        5, 6, subplot_spec=gs0[0], hspace=0.5, wspace=0.8
    )

    ax1 = figure.add_subplot(gs[:-1, 1:5])
    hb = ax1.hexbin(
        mean,
        diff,
        bins="log",
        cmap=cmap,
        gridsize=gridsize,
        extent=lims,
    )
    ax1.axis(lims)
    ax1.axhline(linewidth=1, color="r")
    ax1.axvline(linewidth=1, color="r")
    if title:
        ax1.set_title(title)

    ax2 = figure.add_subplot(gs[:-1, 0], xticklabels=[], sharey=ax1)
    ax2.set_ylim(lims[2:])
    ax2.hist(
        diff,
        bins=gridsize[0],
        range=lims[2:],
        histtype="stepfilled",
        orientation="horizontal",
        color="gray",
    )
    ax2.invert_xaxis()
    ax2.set_ylabel(f"Difference : ({ref_label} - {src_label})")

    ax3 = figure.add_subplot(gs[-1, 1:5], yticklabels=[], sharex=ax1)
    ax3.hist(
        mean,
        bins=gridsize[1],
        range=lims[:2],
        histtype="stepfilled",
        orientation="vertical",
        color="gray",
    )
    ax3.set_xlim(lims[:2])
    ax3.invert_yaxis()
    ax3.set_xlabel(f"Average :  mean({ref_label}, {src_label})")

    if colorbar:
        ax4 = figure.add_subplot(gs[:-1, 5])
        ax4.set_aspect(20)
        pos1 = ax4.get_position()
        ax4.set_position([pos1.x0 - 0.025, pos1.y0, pos1.width, pos1.height])

        cb = figure.colorbar(hb, cax=ax4)
        cb.set_label("log10(N)")

    return save_figure_if_needed(figure, output_file)


def _extract_data_2_images(ref_img, src_img, masker=None):
    """Return data of 2 images as 2 vectors.

    Parameters
    ----------
    ref_img : 3D Niimg-like object or :obj:`~nilearn.surface.SurfaceImage`
        Reference image.

    src_img : 3D Niimg-like object or :obj:`~nilearn.surface.SurfaceImage`
        Source image. Its type must match that of the ``ref_img``.
        If the source image is Niimg-Like,
        it will be resampled to match that or the source image.

    masker : 3D Niimg-like binary mask or \
            :obj:`~nilearn.maskers.NiftiMasker` or \
            binary :obj:`~nilearn.surface.SurfaceImage` or \
            or :obj:`~nilearn.maskers.SurfaceMasker` or \
            None
        Mask to be used on data.
        Its type must be compatible with that of the ``ref_img``.
        If None is passed,
        an appropriate masker will be fitted on the reference image.

    """
    if isinstance(ref_img, NiimgLike) and isinstance(src_img, NiimgLike):
        image_type = "volume"
        ref_img = check_niimg_3d(ref_img)
        src_img = check_niimg_3d(src_img)

    elif isinstance(ref_img, (SurfaceImage)) and isinstance(
        src_img, (SurfaceImage)
    ):
        image_type = "surface"
        ref_img.data._check_ndims(1)
        src_img.data._check_ndims(1)
        check_polymesh_equal(ref_img.mesh, src_img.mesh)

    else:
        raise TypeError(
            "'ref_img' and 'src_img' "
            "must both be Niimg-like or SurfaceImage.\n"
            f"Got {type(src_img)=} and {type(ref_img)=}."
        )

    masker = _sanitize_masker(masker, image_type, ref_img)

    data_ref = masker.transform(ref_img)
    data_src = masker.transform(src_img)

    data_ref = data_ref.ravel()
    data_src = data_src.ravel()

    return data_ref, data_src


def _sanitize_masker(masker, image_type, ref_img):
    """Return an appropriate fitted masker.

    Raise exception
    if there is type mismatch between the masker and ref_img.
    """
    if masker is None:
        if image_type == "volume":
            masker = NiftiMasker(
                target_affine=ref_img.affine,
                target_shape=ref_img.shape,
            )
        else:
            masker = SurfaceMasker()

    check_compatibility_mask_and_images(masker, ref_img)

    if isinstance(masker, NiimgLike):
        masker = NiftiMasker(
            mask_img=masker,
            target_affine=ref_img.affine,
            target_shape=ref_img.shape,
        )
    elif isinstance(masker, SurfaceImage):
        check_polymesh_equal(ref_img.mesh, masker.mesh)
        masker = SurfaceMasker(
            mask_img=masker,
        )

    if not masker.__sklearn_is_fitted__():
        masker.fit(ref_img)

    return masker

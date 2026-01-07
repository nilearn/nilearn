import warnings
from html import escape  # TODO this should be removed from here
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils import Bunch

from nilearn import DEFAULT_DIVERGING_CMAP
from nilearn._utils.docs import fill_doc
from nilearn._utils.helpers import is_matplotlib_installed
from nilearn._utils.logger import find_stack_level
from nilearn._utils.niimg import load_niimg, safe_get_data
from nilearn._utils.param_validation import (
    check_parameter_in_allowed,
    check_params,
)
from nilearn._utils.tags import (
    accept_niimg_input,
    is_masker,
)
from nilearn.glm.thresholding import DEFAULT_Z_THRESHOLD, threshold_stats_img
from nilearn.image import check_niimg
from nilearn.reporting._utils import dataframe_to_html
from nilearn.reporting.get_clusters_table import (
    clustering_params_to_dataframe,
    get_clusters_table,
)
from nilearn.reporting.utils import figure_to_png_base64
from nilearn.surface.surface import SurfaceImage
from nilearn.surface.surface import get_data as get_surface_data


def check_generate_report_input(
    height_control, cluster_threshold, min_distance, plot_type
):
    height_control_methods = [
        "fpr",
        "fdr",
        "bonferroni",
        None,
    ]
    check_parameter_in_allowed(
        height_control,
        height_control_methods,
        "height_control",
    )

    if cluster_threshold < 0:
        raise ValueError(
            f"'cluster_threshold' must be > 0. Got {cluster_threshold=}"
        )

    if min_distance < 0:
        raise ValueError(f"'min_distance' must be > 0. Got {min_distance=}")

    if plot_type not in {"slice", "glass"}:
        raise ValueError(
            "'plot_type' must be one of {'slice', 'glass'}. "
            f"Got {plot_type=}"
        )


def sanitize_generate_report_input(
    height_control,
    threshold,
    cut_coords,
    plot_type,
    first_level_contrast,
    is_first_level_glm: bool,
):
    warning_messages = []

    if is_first_level_glm and first_level_contrast is not None:
        warnings.warn(
            "'first_level_contrast' is ignored for FirstLevelModel."
            "Setting first_level_contrast=None.",
            stacklevel=find_stack_level(),
        )
        first_level_contrast = None

    if height_control is None:
        # TODO (nilearn >= 0.15.0) update to DEFAULT_Z_THRESHOLD
        if threshold is None:
            threshold = 3.09

        # TODO (nilearn >= 0.15.0) remove
        if threshold == 3.09:
            warnings.warn(
                "\nFrom nilearn version>=0.15, "
                "the default 'threshold' will be set to "
                f"{DEFAULT_Z_THRESHOLD}.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )

    elif threshold is not None:
        threshold = float(threshold)
        warning_messages.append(
            f"\n'{threshold=}' is not used with '{height_control=}'."
            "\n'threshold' is only used when 'height_control=None'. "
            "\n'threshold' was set to 'None'. "
        )
        threshold = None

    if cut_coords is not None and plot_type == "glass":
        warning_messages.append(
            f"\n'{cut_coords=}' is not used with '{plot_type=}'."
            "\n'cut_coords' is only used when 'plot_type='slice''. "
            "\n'cut_coords' was set to None. "
        )
        cut_coords = None

    return threshold, cut_coords, first_level_contrast, warning_messages


def turn_into_full_path(bunch, dir: Path) -> str | Bunch:
    """Recursively turns str values of a dict into path.

    Used to turn relative paths into full paths.
    """
    if isinstance(bunch, str) and not bunch.startswith(str(dir)):
        return str(dir / bunch)
    tmp = Bunch()
    for k in bunch:
        if isinstance(bunch[k], (dict, str, Bunch)):
            tmp[k] = turn_into_full_path(bunch[k], dir)
        else:
            tmp[k] = bunch[k]
    return tmp


def glm_model_attributes_to_dataframe(model):
    """Return a pandas dataframe with pertinent model attributes & information.

    Parameters
    ----------
    model : FirstLevelModel or SecondLevelModel object.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the pertinent attributes of the model.
    """
    model_attributes = pd.DataFrame.from_dict(
        model._attributes_to_dict(),
        orient="index",
    )

    if len(model_attributes) == 0:
        return model_attributes

    attribute_units = {
        "t_r": "seconds",
        "high_pass": "Hertz",
        "smoothing_fwhm": "mm",
    }
    attribute_names_with_units = {
        attribute_name_: attribute_name_ + f" ({attribute_unit_})"
        for attribute_name_, attribute_unit_ in attribute_units.items()
    }
    model_attributes = model_attributes.rename(
        index=attribute_names_with_units
    )
    model_attributes.index.names = ["Parameter"]
    model_attributes.columns = ["Value"]

    return model_attributes


def load_bg_img(bg_img, is_volume_glm):
    if bg_img == "MNI152TEMPLATE":
        try:
            from nilearn.plotting.image.utils import (  # type: ignore[assignment]
                MNI152TEMPLATE,
            )

            bg_img = MNI152TEMPLATE if is_volume_glm else None
        except ImportError:
            bg_img = None
    if not is_volume_glm and bg_img and not isinstance(bg_img, SurfaceImage):
        raise TypeError(
            "'bg_img' must a SurfaceImage instance. "
            f"Got {bg_img.__class__.__name__}"
        )


def mask_to_plot(model, bg_img):
    """Plot a mask image and creates PNG code of it.

    Parameters
    ----------
    model : FirstLevelModel or SecondLevelModel object.

    bg_img : Niimg-like object
        See :ref:`extracting_data`.
        The background image that the mask will be plotted on top of.
        To turn off background image, just pass "bg_img=None".


    Returns
    -------
    mask_plot : str
        PNG Image for the mask plot.

    """
    if not is_matplotlib_installed():
        return None

    from matplotlib import pyplot as plt

    from nilearn.plotting import plot_roi

    # Select mask_img to use for plotting
    if not model._is_volume_glm():
        fig = model.masker_._create_figure_for_report()
        mask_plot = figure_to_png_base64(fig)
        # prevents sphinx-gallery & jupyter from scraping & inserting plots
        plt.close()
        return mask_plot

    if is_masker(model.mask_img) and accept_niimg_input(model.mask_img):
        mask_img = model.masker_.mask_img_
    else:
        try:
            # check that mask_img is a niiimg-like object
            check_niimg(model.mask_img)
            mask_img = model.mask_img
        except Exception:
            mask_img = model.masker_.mask_img_

    plot_roi(
        roi_img=mask_img,
        bg_img=bg_img,
        display_mode="z",
        cmap="Set1",
        colorbar=False,
    )
    mask_plot = figure_to_png_base64(plt.gcf())
    # prevents sphinx-gallery & jupyter from scraping & inserting plots
    plt.close()

    return mask_plot


@fill_doc
def make_stat_maps_contrast_clusters(
    stat_img,
    threshold_orig,
    alpha,
    cluster_threshold,
    height_control,
    two_sided,
    min_distance,
    bg_img,
    cut_coords,
    display_mode,
    plot_type,
    # clusters_tsvs,
):
    """Populate a smaller HTML sub-template with the proper values, \
    make a list containing one or more of such components \
    & return the list to be inserted into the HTML Report Template.

    Each component contains the HTML code for
    a contrast & its corresponding statistical maps & cluster table;

    Parameters
    ----------
    stat_img : dictionary of Niimg-like object or SurfaceImage, or None
       Statistical image (presumably in z scale)
       whenever height_control is 'fpr' or None,
       stat_img=None is acceptable.
       If it is 'fdr' or 'bonferroni',
       an error is raised if stat_img is None.

    contrasts_plots : Dict[str, str]
        Contains contrast names & HTML code of the contrast's PNG plot.

    threshold_orig : float
       Desired threshold in z-scale.
       This is used only if height_control is None

    alpha : float
        Number controlling the thresholding (either a p-value or q-value).
        Its actual meaning depends on the height_control parameter.
        This function translates alpha to a z-scale threshold.

    %(cluster_threshold)s

    height_control : string
        False positive control meaning of cluster forming
        threshold: 'fpr' or 'fdr' or 'bonferroni' or None.

    two_sided : `bool`, default=False
        Whether to employ two-sided thresholding or to evaluate positive values
        only.

    min_distance : float, default=8
        For display purposes only.
        Minimum distance between subpeaks in mm.

    bg_img : Niimg-like object
        Only used when plot_type is 'slice'.
        See :ref:`extracting_data`.
        The background image for stat maps to be plotted on upon.
        If nothing is specified, the MNI152 template will be used.
        To turn off background image, just pass "bg_img=False".

    %(cut_coords)s

    display_mode : string
        Choose the direction of the cuts:
        'x' - sagittal, 'y' - coronal, 'z' - axial,
        'l' - sagittal left hemisphere only,
        'r' - sagittal right hemisphere only,
        'ortho' - three cuts are performed in orthogonal directions.

        Possible values are:
        'ortho', 'x', 'y', 'z', 'xz', 'yx', 'yz',
        'l', 'r', 'lr', 'lzr', 'lyr', 'lzry', 'lyrz'.

    plot_type : string {'slice', 'glass'}
        The type of plot to be drawn.

    clusters_tsvs : dictionary of path of to tsv files

    Returns
    -------
    results : dict
        Each key contains
        contrast name, contrast plot, statistical map, cluster table.

    """
    check_params(locals())
    if not display_mode:
        display_mode_selector = {"slice": "z", "glass": "lzry"}
        display_mode = display_mode_selector[plot_type]

    results = {}
    for contrast_name, stat_map_img in stat_img.items():
        # Only use threshold_stats_img to adjust the threshold
        # that we will pass to clustering_params_to_dataframe
        # and _stat_map_to_png
        # Necessary to avoid :
        # https://github.com/nilearn/nilearn/issues/4192

        thresholded_img, threshold = threshold_stats_img(
            stat_img=stat_map_img,
            threshold=threshold_orig,
            alpha=alpha,
            cluster_threshold=cluster_threshold,
            height_control=height_control,
            two_sided=two_sided,
        )

        table_details = clustering_params_to_dataframe(
            threshold,
            cluster_threshold,
            min_distance,
            height_control,
            alpha,
            is_volume_glm=not isinstance(stat_map_img, SurfaceImage),
        )
        table_details_html = dataframe_to_html(
            table_details,
            precision=3,
            header=False,
        )

        # FIXME
        # The commented code below was there to reuse
        # cluster tables generated by save_glm_to_bids
        # to save time.
        # However cluster tables may have been computed
        # with different threshold, cluster_threshol...
        # by save_glm_to_bids than those requested in
        # generate_report.
        # So we are skipping this for now.

        # if clusters_tsvs:
        #     # try to reuse results saved to disk by
        #     # save_glm_to_bids
        #     try:
        #         cluster_table = pd.read_csv(
        #             clusters_tsvs[contrast_name], sep="\t"
        #         )
        #     except Exception:
        #         cluster_table = get_clusters_table(
        #             thresholded_img,
        #             stat_threshold=threshold,
        #             cluster_threshold=cluster_threshold,
        #             min_distance=min_distance,
        #             two_sided=two_sided,
        #         )
        # else:

        cluster_table = get_clusters_table(
            thresholded_img,
            stat_threshold=threshold,
            cluster_threshold=cluster_threshold,
            min_distance=min_distance,
            two_sided=two_sided,
        )
        cluster_table_html = dataframe_to_html(
            cluster_table,
            precision=2,
            index=False,
        )

        stat_map_png, _ = _stat_map_to_png(
            stat_img=thresholded_img,
            threshold=threshold,
            bg_img=bg_img,
            cut_coords=cut_coords,
            display_mode=display_mode,
            plot_type=plot_type,
            table_details=table_details,
            two_sided=two_sided,
        )

        if len(cluster_table) < 2:
            # do not pass anything when nothing survives thresholding
            cluster_table_html = None
            stat_map_png = None

        results[escape(contrast_name)] = Bunch(
            stat_map_img=stat_map_png,
            cluster_table_details=table_details_html,
            cluster_table=cluster_table_html,
        )

    return results


@fill_doc
def _stat_map_to_png(
    stat_img,
    threshold,
    bg_img,
    cut_coords,
    display_mode,
    plot_type,
    table_details,
    two_sided,
):
    """Generate PNG code for a statistical map, \
    including its clustering parameters.

    Parameters
    ----------
    stat_img : Niimg-like object or None
       Statistical image (presumably in z scale),
       to be plotted as slices or glass brain.

    threshold : float
       Desired threshold in z-scale.

    bg_img : Niimg-like object
        Only used when plot_type is 'slice'.
        See :ref:`extracting_data`.
        The background image for stat maps to be plotted on upon.
        If nothing is specified, the MNI152 template will be used.
        To turn off background image, just pass "bg_img=False".

    %(cut_coords)s

    display_mode : string
        Choose the direction of the cuts:
        'x' - sagittal, 'y' - coronal, 'z' - axial,
        'l' - sagittal left hemisphere only,
        'r' - sagittal right hemisphere only,
        'ortho' - three cuts are performed in orthogonal directions.

        Possible values are:
        'ortho', 'x', 'y', 'z', 'xz', 'yx', 'yz',
        'l', 'r', 'lr', 'lzr', 'lyr', 'lzry', 'lyrz'.

    plot_type : string {'slice', 'glass'}
        The type of plot to be drawn.

    table_details : pandas.Dataframe
        Dataframe listing the parameters used for clustering,
        to be included in the plot.

    two_sided : `bool`, default=False
        Whether to employ two-sided thresholding or to evaluate positive values
        only.

    Returns
    -------
    stat_map_png : string
        PNG Image Data representing a statistical map.

    fig : matplotlib figure
        only used for testing

    """
    if not is_matplotlib_installed():
        return None, None

    from matplotlib import pyplot as plt

    from nilearn.plotting import (
        plot_glass_brain,
        plot_stat_map,
        plot_surf_stat_map,
    )

    cmap = DEFAULT_DIVERGING_CMAP

    if two_sided:
        symmetric_cbar = True
        vmin = vmax = None

    else:
        symmetric_cbar = False

        if isinstance(stat_img, SurfaceImage):
            data = get_surface_data(stat_img)
        else:
            stat_img = load_niimg(stat_img)
            data = safe_get_data(stat_img, ensure_finite=True)

        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
        if vmin >= 0.0:
            vmin = 0
            cmap = "Reds"
        elif vmax <= 0.0:
            vmax = 0
            cmap = "Blues_r"

    if isinstance(stat_img, SurfaceImage):
        surf_mesh = bg_img.mesh if bg_img else None
        stat_map_plot = plot_surf_stat_map(
            stat_map=stat_img,
            hemi="left",
            bg_map=bg_img,
            surf_mesh=surf_mesh,
            cmap=cmap,
            symmetric_cbar=symmetric_cbar,
            threshold=abs(threshold),
        )

        x_label_color = "black"

    else:
        check_parameter_in_allowed(plot_type, ["slice", "glass"], "plot_type")

        if plot_type == "slice":
            stat_map_plot = plot_stat_map(
                stat_img,
                bg_img=bg_img,
                cut_coords=cut_coords,
                display_mode=display_mode,
                cmap=cmap,
                symmetric_cbar=symmetric_cbar,
                draw_cross=False,
                threshold=abs(threshold),
            )
        elif plot_type == "glass":
            stat_map_plot = plot_glass_brain(
                stat_img,
                display_mode=display_mode,
                plot_abs=False,
                symmetric_cbar=symmetric_cbar,
                cmap=cmap,
                threshold=abs(threshold),
            )

        x_label_color = "white" if plot_type == "slice" else "black"

    if hasattr(stat_map_plot, "_cbar"):
        cbar_ax = stat_map_plot._cbar.ax
        cbar_ax.set_xlabel(
            "Z score",
            labelpad=5,
            fontweight="bold",
            loc="right",
            color=x_label_color,
        )

    with pd.option_context("display.precision", 2):
        _add_params_to_plot(table_details, stat_map_plot)

    fig = plt.gcf()
    stat_map_png = figure_to_png_base64(fig)
    # prevents sphinx-gallery & jupyter from scraping & inserting plots
    plt.close()

    # the fig is returned for testing
    return stat_map_png, fig


def _add_params_to_plot(table_details, stat_map_plot):
    """Insert thresholding parameters into the stat map plot \
    as figure suptitle.

    Parameters
    ----------
    table_details : Dict[String, Any]
        Dict of parameters and values used in thresholding.

    stat_map_plot : matplotlib.Axes
        Axes object of the stat map plot.

    Returns
    -------
    stat_map_plot : matplotlib.Axes
        Axes object of the stat map plot, with the added suptitle.

    """
    from matplotlib import pyplot as plt

    from nilearn._utils.plotting import resize_plot_inches

    thresholding_params = [
        ":".join([name, str(val)]) for name, val in table_details[0].items()
    ]
    thresholding_params = "  ".join(thresholding_params)
    suptitle_text = plt.suptitle(
        thresholding_params,
        fontsize=11,
        x=0.45,
        wrap=True,
    )
    fig = plt.gcf()
    resize_plot_inches(
        plot=fig,
        width_change=0.2,
        height_change=1,
    )

    if hasattr(stat_map_plot, "_black_bg") and stat_map_plot._black_bg:
        suptitle_text.set_color("w")

    return stat_map_plot

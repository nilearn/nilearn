"""
Functionality to create an HTML report using a fitted GLM & contrasts.

Functions
---------

make_glm_report(model, contrasts):
    Creates an HTMLReport Object which can be viewed or saved as a report.

"""

import datetime
import uuid
from html import escape
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from nilearn._utils import check_niimg, fill_doc
from nilearn._utils.niimg import safe_get_data
from nilearn._version import __version__
from nilearn.externals import tempita
from nilearn.glm import threshold_stats_img
from nilearn.glm.first_level import FirstLevelModel
from nilearn.maskers import NiftiMasker, SurfaceMasker
from nilearn.plotting import (
    plot_contrast_matrix,
    plot_design_matrix,
    plot_glass_brain,
    plot_roi,
    plot_stat_map,
    plot_surf_stat_map,
)
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
from nilearn.plotting.img_plotting import MNI152TEMPLATE
from nilearn.reporting._utils import (
    check_report_dims,
    clustering_params_to_dataframe,
    coerce_to_dict,
    dataframe_to_html,
    make_stat_maps,
    model_attributes_to_dataframe,
    return_model_type,
)
from nilearn.reporting.get_clusters_table import get_clusters_table
from nilearn.reporting.html_report import (
    HTMLReport,
    _render_warnings_partial,
)
from nilearn.reporting.utils import (
    CSS_PATH,
    HTML_TEMPLATE_PATH,
    TEMPLATE_ROOT_PATH,
    figure_to_png_base64,
    figure_to_svg_quoted,
)
from nilearn.surface import SurfaceImage

HTML_TEMPLATE_ROOT_PATH = Path(__file__).parent / "glm_reporter_templates"


@fill_doc
def make_glm_report(
    model,
    contrasts=None,
    title=None,
    bg_img="MNI152TEMPLATE",
    threshold=3.09,
    alpha=0.001,
    cluster_threshold=0,
    height_control="fpr",
    two_sided=False,
    min_distance=8.0,
    plot_type="slice",
    cut_coords=None,
    display_mode=None,
    report_dims=(1600, 800),
):
    """Return HTMLReport object \
    for a report which shows all important aspects of a fitted GLM.

    The object can be opened in a browser, displayed in a notebook,
    or saved to disk as a standalone HTML file.

    Examples
    --------
    report = make_glm_report(model, contrasts)
    report.open_in_browser()
    report.save_as_html(destination_path)

    Parameters
    ----------
    model : FirstLevelModel or SecondLevelModel object
        A fitted first or second level model object.
        Must have the computed design matrix(ces).

    contrasts : :obj:`dict` with :obj:`str` - ndarray key-value pairs \
        or :obj:`str` \
        or :obj:`list` of :obj:`str` \
        or ndarray or \
        :obj:`list` of ndarray

        Contrasts information for a first or second level model.

        Example:

            Dict of :term:`contrast` names and coefficients,
            or list of :term:`contrast` names
            or list of :term:`contrast` coefficients
            or :term:`contrast` name
            or :term:`contrast` coefficient

            Each :term:`contrast` name must be a string.
            Each :term:`contrast` coefficient must be a list
            or numpy array of ints.

        Contrasts are passed to ``contrast_def`` for FirstLevelModel
        (:func:`nilearn.glm.first_level.FirstLevelModel.compute_contrast`)
        & second_level_contrast for SecondLevelModel
        (:func:`nilearn.glm.second_level.SecondLevelModel.compute_contrast`)

    title : :obj:`str`, default=None
        If string, represents the web page's title and primary heading,
        model type is sub-heading.
        If None, page titles and headings are autogenerated
        using :term:`contrast` names.

    bg_img : Niimg-like object, default='MNI152TEMPLATE'
        See :ref:`extracting_data`.
        The background image for mask and stat maps to be plotted on upon.
        To turn off background image, just pass "bg_img=None".

    threshold : :obj:`float`, default=3.09
        Cluster forming threshold in same scale as `stat_img` (either a
        t-scale or z-scale value). Used only if height_control is None.

    alpha : :obj:`float`, default=0.001
        Number controlling the thresholding (either a p-value or q-value).
        Its actual meaning depends on the height_control parameter.
        This function translates alpha to a z-scale threshold.

    cluster_threshold : :obj:`int`, default=0
        Cluster size threshold, in voxels.

    height_control :  :obj:`str`, default='fpr'
        false positive control meaning of cluster forming
        threshold: 'fpr' or 'fdr' or 'bonferroni' or None.

    two_sided : :obj:`bool`, default=False
        Whether to employ two-sided thresholding or to evaluate positive values
        only.

    min_distance : :obj:`float`, default=8.0
        For display purposes only.
        Minimum distance between subpeaks in mm.

    plot_type : :obj:`str`, {'slice', 'glass'}, default='slice'
        Specifies the type of plot to be drawn for the statistical maps.

    %(cut_coords)s

    display_mode :  :obj:`str`, default=None
        Default is 'z' if plot_type is 'slice'; '
        ortho' if plot_type is 'glass'.

        Choose the direction of the cuts:
        'x' - sagittal, 'y' - coronal, 'z' - axial,
        'l' - sagittal left hemisphere only,
        'r' - sagittal right hemisphere only,
        'ortho' - three cuts are performed in orthogonal directions.

        Possible values are:
        'ortho', 'x', 'y', 'z', 'xz', 'yx', 'yz',
        'l', 'r', 'lr', 'lzr', 'lyr', 'lzry', 'lyrz'.

    report_dims : Sequence[:obj:`int`, :obj:`int`], default=(1600, 800)
        Specifies width, height (in pixels) of report window within a notebook.
        Only applicable when inserting the report into a Jupyter notebook.
        Can be set after report creation using report.width, report.height.

    Returns
    -------
    report_text : HTMLReport Object
        Contains the HTML code for the :term:`GLM` Report.

    """
    is_volume_glm = True
    if isinstance(model.mask_img, (SurfaceMasker, SurfaceImage)) or (
        hasattr(model, "masker_") and isinstance(model.masker_, SurfaceMasker)
    ):
        is_volume_glm = False

    unique_id = str(uuid.uuid4()).replace("-", "")

    title = f"<br>{title}" if title else ""
    title = f"Statistical Report - {return_model_type(model)}{title}"

    docstring = model.__doc__
    snippet = docstring.partition("Parameters\n    ----------\n")[0]

    date = datetime.datetime.now().replace(microsecond=0).isoformat()

    css_file_path = CSS_PATH / "masker_report.css"
    with css_file_path.open(encoding="utf-8") as css_file:
        css = css_file.read()

    model_attributes = model_attributes_to_dataframe(
        model, is_volume_glm=is_volume_glm
    )
    with pd.option_context("display.max_colwidth", 100):
        model_attributes_html = dataframe_to_html(
            model_attributes,
            precision=2,
            header=True,
            sparsify=False,
        )

    body_template_path = HTML_TEMPLATE_PATH / "glm_report.html"
    tpl = tempita.HTMLTemplate.from_filename(
        str(body_template_path),
        encoding="utf-8",
    )

    warning_messages = []
    if not model.__sklearn_is_fitted__():
        warning_messages.append("The model has not been fit yet.")

        body = tpl.substitute(
            css=css,
            title=title,
            docstring=snippet,
            warning_messages=_render_warnings_partial(warning_messages),
            design_matrices_dict=None,
            parameters=model_attributes_html,
            contrasts_dict=None,
            statistical_maps=None,
            cluster_table_details=None,
            mask_plot=None,
            cluster_table=None,
            component=None,
            date=date,
            unique_id=unique_id,
        )

    else:
        design_matrices = (
            model.design_matrices_
            if isinstance(model, FirstLevelModel)
            else [model.design_matrix_]
        )

        design_matrices_dict = _return_design_matrices_dict(design_matrices)

        contrasts = coerce_to_dict(contrasts)
        contrasts_dict = _return_contrasts_dict(design_matrices, contrasts)

        if bg_img == "MNI152TEMPLATE":
            bg_img = MNI152TEMPLATE if is_volume_glm else None
        if (
            not is_volume_glm
            and bg_img
            and not isinstance(bg_img, SurfaceImage)
        ):
            raise TypeError(
                f"'bg_img' must a SurfaceImage instance.Got {type(bg_img)=}"
            )

        mask_plot = _mask_to_plot(model, bg_img, cut_coords, is_volume_glm)

        statistical_maps = make_stat_maps(model, contrasts)

        results = _make_stat_maps_contrast_clusters(
            stat_img=statistical_maps,
            threshold=threshold,
            alpha=alpha,
            cluster_threshold=cluster_threshold,
            height_control=height_control,
            two_sided=two_sided,
            min_distance=min_distance,
            bg_img=bg_img,
            cut_coords=cut_coords,
            display_mode=display_mode,
            plot_type=plot_type,
        )

        body = tpl.substitute(
            css=css,
            title=title,
            docstring=snippet,
            warning_messages=_render_warnings_partial(warning_messages),
            parameters=model_attributes_html,
            contrasts_dict=contrasts_dict,
            mask_plot=mask_plot,
            results=results,
            design_matrices_dict=design_matrices_dict,
            unique_id=unique_id,
            date=date,
        )

    # revert HTML safe substitutions in CSS sections
    body = body.replace(".pure-g &gt; div", ".pure-g > div")

    head_template_path = (
        TEMPLATE_ROOT_PATH / "html" / "report_head_template.html"
    )
    with head_template_path.open() as head_file:
        head_tpl = Template(head_file.read())

    head_css_file_path = CSS_PATH / "head.css"
    with head_css_file_path.open(encoding="utf-8") as head_css_file:
        head_css = head_css_file.read()

    report = HTMLReport(
        body=body,
        head_tpl=head_tpl,
        head_values={
            "head_css": head_css,
            "version": __version__,
            "page_title": title,
        },
    )

    # setting report size for better visual experience in Jupyter Notebooks.
    report.width, report.height = check_report_dims(report_dims)

    return report


def _plot_to_svg(plot):
    """Create an SVG image as a data URL \
    from a Matplotlib Axes or Figure object.

    Parameters
    ----------
    plot : Matplotlib Axes or Figure object
        Contains the plot information.

    Returns
    -------
    url_plot_svg : String
        SVG Image Data URL.

    """
    try:
        return figure_to_svg_quoted(plot)
    except AttributeError:
        return figure_to_svg_quoted(plot.figure)


def _resize_plot_inches(plot, width_change=0, height_change=0):
    """Accept a matplotlib figure or axes object and resize it (in inches).

    Returns the original object.

    Parameters
    ----------
    plot : matplotlib.Figure() or matplotlib.Axes()
        The matplotlib Figure/Axes object to be resized.

    width_change : float, default=0
        The amount of change to be added on to original width.
        Use negative values for reducing figure dimensions.

    height_change : float, default=0
        The amount of change to be added on to original height.
        Use negative values for reducing figure dimensions.

    Returns
    -------
    plot : matplotlib.Figure() or matplotlib.Axes()
        The matplotlib Figure/Axes object after being resized.

    """
    try:
        orig_size = plot.figure.get_size_inches()
    except AttributeError:
        orig_size = plot.get_size_inches()
    new_size = (
        orig_size[0] + width_change,
        orig_size[1] + height_change,
    )
    try:
        plot.figure.set_size_inches(new_size, forward=True)
    except AttributeError:
        plot.set_size_inches(new_size)
    return plot


def _mask_to_plot(model, bg_img, cut_coords, is_volume_glm):
    """Plot cuts of an mask image and creates SVG code of it.

    Parameters
    ----------
    model

    bg_img : Niimg-like object
        See :ref:`extracting_data`.
        The background image that the mask will be plotted on top of.
        To turn off background image, just pass "bg_img=None".

    cut_coords

    is_volume_glm : bool

    Returns
    -------
    mask_plot : str
        SVG Image Data URL for the mask plot.

    """
    # Select mask_img to use for plotting
    if not is_volume_glm:
        fig = model.masker_._create_figure_for_report()
        mask_plot = figure_to_png_base64(fig)
        # prevents sphinx-gallery & jupyter from scraping & inserting plots
        plt.close()
        return mask_plot

    if isinstance(model.mask_img, NiftiMasker):
        mask_img = model.masker_.mask_img_
    else:
        try:
            # check that mask_img is a niiimg-like object
            check_niimg(model.mask_img)
            mask_img = model.mask_img
        except Exception:
            mask_img = model.masker_.mask_img_

    if not mask_img:
        return None  # HTML image tag's alt attribute is used.

    plot_roi(
        roi_img=mask_img,
        bg_img=bg_img,
        display_mode="z",
        cmap="Set1",
        cut_coords=cut_coords,
    )
    mask_plot = _plot_to_svg(plt.gcf())
    # prevents sphinx-gallery & jupyter from scraping & inserting plots
    plt.close()

    return mask_plot


@fill_doc
def _make_stat_maps_contrast_clusters(
    stat_img,
    threshold,
    alpha,
    cluster_threshold,
    height_control,
    two_sided,
    min_distance,
    bg_img,
    cut_coords,
    display_mode,
    plot_type,
):
    """Populate a smaller HTML sub-template with the proper values, \
    make a list containing one or more of such components \
    & return the list to be inserted into the HTML Report Template.

    Each component contains the HTML code for
    a contrast & its corresponding statistical maps & cluster table;

    Parameters
    ----------
    stat_img : Niimg-like object or None
       Statistical image (presumably in z scale)
       whenever height_control is 'fpr' or None,
       stat_img=None is acceptable.
       If it is 'fdr' or 'bonferroni',
       an error is raised if stat_img is None.

    contrasts_plots : Dict[str, str]
        Contains contrast names & HTML code of the contrast's SVG plot.

    threshold : float
       Desired threshold in z-scale.
       This is used only if height_control is None

    alpha : float
        Number controlling the thresholding (either a p-value or q-value).
        Its actual meaning depends on the height_control parameter.
        This function translates alpha to a z-scale threshold.

    cluster_threshold : float
        Cluster size threshold. In the returned thresholded map,
        sets of connected voxels (`clusters`) with size smaller
        than this number will be removed.

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

    Returns
    -------
    results : dict
        Each key contains
        contrast name, contrast plot, statistical map, cluster table.

    """
    if not display_mode:
        display_mode_selector = {"slice": "z", "glass": "lzry"}
        display_mode = display_mode_selector[plot_type]

    results = {}
    for contrast_name, stat_map_img in stat_img.items():
        if isinstance(stat_map_img, SurfaceImage):
            surf_mesh = bg_img.mesh if bg_img else None
            fig = plot_surf_stat_map(
                stat_map=stat_map_img,
                hemi="left",
                colorbar=True,
                threshold=threshold,
                bg_map=bg_img,
                surf_mesh=surf_mesh,
            )
            stat_map_svg = figure_to_png_base64(fig)

            # prevents sphinx-gallery & jupyter from scraping & inserting plots
            plt.close("all")

            table_details_html = None
            cluster_table_html = None

        else:
            # Only use threshold_stats_img to adjust the threshold
            # that we will pass to clustering_params_to_dataframe
            # and _stat_map_to_svg
            # Necessary to avoid :
            # https://github.com/nilearn/nilearn/issues/4192
            thresholded_img, threshold = threshold_stats_img(
                stat_img=stat_map_img,
                threshold=threshold,
                alpha=alpha,
                cluster_threshold=cluster_threshold,
                height_control=height_control,
            )
            table_details = clustering_params_to_dataframe(
                threshold,
                cluster_threshold,
                min_distance,
                height_control,
                alpha,
            )
            stat_map_svg = _stat_map_to_svg(
                stat_img=thresholded_img,
                threshold=threshold,
                bg_img=bg_img,
                cut_coords=cut_coords,
                display_mode=display_mode,
                plot_type=plot_type,
                table_details=table_details,
            )

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

            table_details_html = dataframe_to_html(
                table_details,
                precision=3,
                header=False,
            )

        results[escape(contrast_name)] = tempita.bunch(
            stat_map_img=stat_map_svg,
            cluster_table_details=table_details_html,
            cluster_table=cluster_table_html,
        )

    return results


@fill_doc
def _stat_map_to_svg(
    stat_img,
    threshold,
    bg_img,
    cut_coords,
    display_mode,
    plot_type,
    table_details,
):
    """Generate SVG code for a statistical map, \
    including its clustering parameters.

    Parameters
    ----------
    stat_img : Niimg-like object or None
       Statistical image (presumably in z scale),
       to be plotted as slices or glass brain.
       Does not perform any thresholding.

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

    Returns
    -------
    stat_map_svg : string
        SVG Image Data URL representing a statistical map.

    """
    data = safe_get_data(stat_img, ensure_finite=True)
    stat_map_min = np.nanmin(data)
    stat_map_max = np.nanmax(data)
    symmetric_cbar = True
    cmap = "RdBu_r"
    if stat_map_min >= 0.0:
        symmetric_cbar = False
        cmap = "red_transparent_full_alpha_range"
    elif stat_map_max <= 0.0:
        symmetric_cbar = False
        cmap = "blue_transparent_full_alpha_range"
        cmap = nilearn_cmaps[cmap].reversed()

    if plot_type == "slice":
        stat_map_plot = plot_stat_map(
            stat_img,
            bg_img=bg_img,
            cut_coords=cut_coords,
            display_mode=display_mode,
            colorbar=True,
            cmap=cmap,
            symmetric_cbar=symmetric_cbar,
            threshold=threshold,
        )
    elif plot_type == "glass":
        stat_map_plot = plot_glass_brain(
            stat_img,
            display_mode=display_mode,
            colorbar=True,
            plot_abs=False,
            symmetric_cbar=symmetric_cbar,
            cmap=cmap,
            threshold=threshold,
        )
    else:
        raise ValueError(
            "Invalid plot type provided. "
            "Acceptable options are 'slice' or 'glass'."
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
    stat_map_svg = _plot_to_svg(fig)
    # prevents sphinx-gallery & jupyter from scraping & inserting plots
    plt.close()
    return stat_map_svg


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
    fig = next(iter(stat_map_plot.axes.values())).ax.figure
    _resize_plot_inches(
        plot=fig,
        width_change=0.2,
        height_change=1,
    )
    if stat_map_plot._black_bg:
        suptitle_text.set_color("w")
    return stat_map_plot


def _return_design_matrices_dict(design_matrices):
    if design_matrices is None:
        return None

    design_matrices_dict = {}
    for dmtx_count, design_matrix in enumerate(design_matrices, start=1):
        dmtx_plot = plot_design_matrix(design_matrix)
        dmtx_title = f"Run {dmtx_count}"
        if len(design_matrices) > 1:
            plt.title(dmtx_title, y=1.025, x=-0.1)
        dmtx_plot = _resize_plot_inches(dmtx_plot, height_change=0.3)
        url_design_matrix_svg = _plot_to_svg(dmtx_plot)
        # prevents sphinx-gallery & jupyter from scraping & inserting plots
        plt.close("all")

        design_matrices_dict[dmtx_title] = url_design_matrix_svg

    return design_matrices_dict


def _return_contrasts_dict(design_matrices, contrasts):
    if design_matrices is None or not contrasts:
        return None

    contrasts_dict = {}
    for design_matrix in design_matrices:
        for contrast_name, contrast_data in contrasts.items():
            contrast_plot = plot_contrast_matrix(
                contrast_data, design_matrix, colorbar=True
            )
            contrast_plot.set_xlabel(contrast_name)
            contrast_plot.figure.set_figheight(2)
            url_contrast_plot_svg = _plot_to_svg(contrast_plot)
            # prevents sphinx-gallery & jupyter
            # from scraping & inserting plots
            plt.close("all")
            contrasts_dict[contrast_name] = url_contrast_plot_svg

    return contrasts_dict

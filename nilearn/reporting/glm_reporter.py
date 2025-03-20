"""
Functionality to create an HTML report using a fitted GLM & contrasts.

Functions
---------

make_glm_report(model, contrasts):
    Creates an HTMLReport Object which can be viewed or saved as a report.

"""

import datetime
import uuid
import warnings
from html import escape
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd

from nilearn import DEFAULT_DIVERGING_CMAP
from nilearn._utils import check_niimg, fill_doc, logger
from nilearn._utils.glm import coerce_to_dict, make_stat_maps
from nilearn._utils.helpers import is_matplotlib_installed
from nilearn._utils.niimg import safe_get_data
from nilearn._version import __version__
from nilearn.externals import tempita
from nilearn.glm import threshold_stats_img
from nilearn.maskers import NiftiMasker
from nilearn.reporting._utils import (
    check_report_dims,
    clustering_params_to_dataframe,
    dataframe_to_html,
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
)
from nilearn.surface.surface import SurfaceImage
from nilearn.surface.surface import get_data as get_surface_data

MNI152TEMPLATE = None
if is_matplotlib_installed():
    from matplotlib import pyplot as plt

    from nilearn._utils.plotting import (
        generate_constrat_matrices_figures,
        generate_design_matrices_figures,
        resize_plot_inches,
    )
    from nilearn.plotting import (
        plot_glass_brain,
        plot_roi,
        plot_stat_map,
        plot_surf_stat_map,
    )
    from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
    from nilearn.plotting.img_plotting import (  # type: ignore[assignment]
        MNI152TEMPLATE,
    )


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
    if not is_matplotlib_installed():
        warnings.warn(
            ("No plotting back-end detected. Output will be missing figures."),
            UserWarning,
            stacklevel=2,
        )

    unique_id = str(uuid.uuid4()).replace("-", "")

    title = f"<br>{title}" if title else ""
    title = f"Statistical Report - {model.__str__()}{title}"

    docstring = model.__doc__
    snippet = docstring.partition("Parameters\n    ----------\n")[0]

    date = datetime.datetime.now().replace(microsecond=0).isoformat()

    smoothing_fwhm = getattr(model, "smoothing_fwhm", 0)
    if smoothing_fwhm == 0:
        smoothing_fwhm = None

    model_attributes = _glm_model_attributes_to_dataframe(model)
    with pd.option_context("display.max_colwidth", 100):
        model_attributes_html = dataframe_to_html(
            model_attributes,
            precision=2,
            header=True,
            sparsify=False,
        )

    contrasts = coerce_to_dict(contrasts)

    # If some contrasts are passed
    # we do not rely on filenames stored in the model.
    output = None
    if contrasts is None:
        output = model._reporting_data.get("filenames", None)

    design_matrices = None
    mask_plot = None
    results = None
    warning_messages = ["The model has not been fit yet."]
    if model.__sklearn_is_fitted__():
        warning_messages = []

        if model.__str__() == "Second Level Model":
            design_matrices = [model.design_matrix_]
        else:
            design_matrices = model.design_matrices_

        if bg_img == "MNI152TEMPLATE":
            bg_img = MNI152TEMPLATE if model._is_volume_glm() else None
        if (
            not model._is_volume_glm()
            and bg_img
            and not isinstance(bg_img, SurfaceImage)
        ):
            raise TypeError(
                f"'bg_img' must a SurfaceImage instance. Got {type(bg_img)=}"
            )

        mask_plot = _mask_to_plot(model, bg_img, cut_coords)

        if output is not None:
            # we try to rely on the content of glm object only
            try:
                statistical_maps = {
                    contrast_name: output["dir"]
                    / output["statistical_maps"][contrast_name]["z_score"]
                    for contrast_name in output["statistical_maps"]
                }
            except KeyError:  # pragma: no cover
                statistical_maps = make_stat_maps(
                    model, contrasts, output_type="z_score"
                )
        else:
            statistical_maps = make_stat_maps(
                model, contrasts, output_type="z_score"
            )

        logger.log(
            "Generating contrast-level figures...", verbose=model.verbose
        )
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

    design_matrices_dict = tempita.bunch()
    contrasts_dict = tempita.bunch()
    if output is not None:
        design_matrices_dict = output["design_matrices_dict"]
        # FIXME only contrast of first run are displayed
        # contrasts_dict[i_run] = tempita.bunch(**input["contrasts_dict"][i_run]) # noqa: E501
        contrasts_dict = output["contrasts_dict"]

    if is_matplotlib_installed():
        logger.log(
            "Generating design matrices figures...", verbose=model.verbose
        )
        design_matrices_dict = generate_design_matrices_figures(
            design_matrices,
            design_matrices_dict=design_matrices_dict,
            output=output,
        )

        logger.log(
            "Generating contrast matrices figures...", verbose=model.verbose
        )
        contrasts_dict = generate_constrat_matrices_figures(
            design_matrices,
            contrasts,
            contrasts_dict=contrasts_dict,
            output=output,
        )

        # FIXME
        # temporaty hack as the GLM reports only
        # show the contrast matrices of a single run
        contrasts_dict = contrasts_dict[0]

    # for methods writing, only keep the contrast expressed as strings
    if contrasts is not None:
        contrasts = [x for x in contrasts.values() if isinstance(x, str)]
    method_section_template_path = HTML_TEMPLATE_PATH / "method_section.html"
    method_tpl = tempita.HTMLTemplate.from_filename(
        str(method_section_template_path),
        encoding="utf-8",
    )
    method_section = method_tpl.substitute(
        version=__version__,
        model_type=model.__str__(),
        reporting_data=tempita.bunch(**model._reporting_data),
        smoothing_fwhm=smoothing_fwhm,
        contrasts=contrasts,
    )

    body_template_path = HTML_TEMPLATE_PATH / "glm_report.html"
    tpl = tempita.HTMLTemplate.from_filename(
        str(body_template_path),
        encoding="utf-8",
    )

    css_file_path = CSS_PATH / "masker_report.css"
    with css_file_path.open(encoding="utf-8") as css_file:
        css = css_file.read()

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
        method_section=method_section,
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


def _glm_model_attributes_to_dataframe(model):
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


def _mask_to_plot(model, bg_img, cut_coords):
    """Plot a mask image and creates PNG code of it.

    Parameters
    ----------
    model

    bg_img : Niimg-like object
        See :ref:`extracting_data`.
        The background image that the mask will be plotted on top of.
        To turn off background image, just pass "bg_img=None".

    cut_coords


    Returns
    -------
    mask_plot : str
        PNG Image for the mask plot.

    """
    if not is_matplotlib_installed():
        return None
    # Select mask_img to use for plotting
    if not model._is_volume_glm():
        model.masker_._create_figure_for_report()
        fig = plt.gcf()
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

    plot_roi(
        roi_img=mask_img,
        bg_img=bg_img,
        display_mode="z",
        cmap="Set1",
        cut_coords=cut_coords,
        colorbar=False,
    )
    mask_plot = figure_to_png_base64(plt.gcf())
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
        Contains contrast names & HTML code of the contrast's PNG plot.

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
        # Only use threshold_stats_img to adjust the threshold
        # that we will pass to clustering_params_to_dataframe
        # and _stat_map_to_png
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
            is_volume_glm=not isinstance(stat_map_img, SurfaceImage),
        )
        table_details_html = dataframe_to_html(
            table_details,
            precision=3,
            header=False,
        )

        cluster_table_html = None
        if not isinstance(thresholded_img, SurfaceImage):
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

        stat_map_png = _stat_map_to_png(
            stat_img=thresholded_img,
            threshold=threshold,
            bg_img=bg_img,
            cut_coords=cut_coords,
            display_mode=display_mode,
            plot_type=plot_type,
            table_details=table_details,
        )

        results[escape(contrast_name)] = tempita.bunch(
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
):
    """Generate PNG code for a statistical map, \
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
    stat_map_png : string
        PNG Image Data representing a statistical map.

    """
    if not is_matplotlib_installed():
        return None

    cmap = DEFAULT_DIVERGING_CMAP

    if isinstance(stat_img, SurfaceImage):
        data = get_surface_data(stat_img)
    else:
        data = safe_get_data(stat_img, ensure_finite=True)

    stat_map_min = np.nanmin(data)
    stat_map_max = np.nanmax(data)
    symmetric_cbar = True
    if stat_map_min >= 0.0:
        symmetric_cbar = False
        cmap = "red_transparent_full_alpha_range"
    elif stat_map_max <= 0.0:
        symmetric_cbar = False
        cmap = "blue_transparent_full_alpha_range"
        cmap = nilearn_cmaps[cmap].reversed()

    if isinstance(stat_img, SurfaceImage):
        surf_mesh = bg_img.mesh if bg_img else None
        stat_map_plot = plot_surf_stat_map(
            stat_map=stat_img,
            hemi="left",
            threshold=threshold,
            bg_map=bg_img,
            surf_mesh=surf_mesh,
            cmap=cmap,
        )

        x_label_color = "black"

    else:
        if plot_type == "slice":
            stat_map_plot = plot_stat_map(
                stat_img,
                bg_img=bg_img,
                cut_coords=cut_coords,
                display_mode=display_mode,
                cmap=cmap,
                symmetric_cbar=symmetric_cbar,
                threshold=threshold,
            )
        elif plot_type == "glass":
            stat_map_plot = plot_glass_brain(
                stat_img,
                display_mode=display_mode,
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
    stat_map_png = figure_to_png_base64(fig)
    # prevents sphinx-gallery & jupyter from scraping & inserting plots
    plt.close()

    return stat_map_png


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
    fig = plt.gcf()
    resize_plot_inches(
        plot=fig,
        width_change=0.2,
        height_change=1,
    )

    if hasattr(stat_map_plot, "_black_bg") and stat_map_plot._black_bg:
        suptitle_text.set_color("w")

    return stat_map_plot

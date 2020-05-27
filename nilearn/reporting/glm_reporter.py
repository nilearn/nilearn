# -*- coding: utf-8 -*-
"""
Functionality to create an HTML report using a fitted GLM & contrasts.

Functions
---------

make_glm_report(model, contrasts):
    Creates an HTMLDocument Object which can be viewed or saved as a report.

"""

import io
import os
import string
import warnings

from collections import OrderedDict
from collections.abc import Iterable
from html import escape
from urllib.parse import quote

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from nilearn.plotting import (plot_glass_brain,
                              plot_roi,
                              plot_stat_map,
                              )
from nilearn.plotting.img_plotting import MNI152TEMPLATE
from nilearn.plotting.js_plotting_utils import HTMLDocument
from nilearn import stats
from nilearn.reporting import (plot_contrast_matrix,
                               plot_design_matrix,
                               get_clusters_table,
                               )
from nilearn.stats.thresholding import threshold_stats_img


HTML_TEMPLATE_ROOT_PATH = os.path.join(os.path.dirname(__file__),
                                       'glm_reporter_templates')


def make_glm_report(model,
                    contrasts,
                    title=None,
                    bg_img=MNI152TEMPLATE,
                    threshold=3.09,
                    alpha=0.001,
                    cluster_threshold=0,
                    height_control='fpr',
                    min_distance=8.,
                    plot_type='slice',
                    display_mode=None,
                    report_dims=(1600, 800),
                    ):
    """ Returns HTMLDocument object
    for a report which shows all important aspects of a fitted GLM.
    The object can be opened in a browser, displayed in a notebook,
    or saved to disk as a standalone HTML file.

    Examples:
        report = make_glm_report(model, contrasts)
        report.open_in_browser()
        report.save_as_html(destination_path)

    Parameters
    ----------
    model: FirstLevelModel or SecondLevelModel object
        A fitted first or second level model object.
        Must have the computed design matrix(ces).

    contrasts: Dict[string, ndarray] or String or List[String] or ndarray or
        List[ndarray]

        Contrasts information for a first or second level model.

        Example:

            Dict of contrast names and coefficients,
            or list of contrast names
            or list of contrast coefficients
            or contrast name
            or contrast coefficient

            Each contrast name must be a string.
            Each contrast coefficient must be a list or numpy array of ints.

        Contrasts are passed to ``contrast_def`` for FirstLevelModel
        (:func:`nilearn.stats.first_level_model.FirstLevelModel.compute_contrast`)
        & second_level_contrast for SecondLevelModel
        (:func:`nilearn.stats.second_level_model.SecondLevelModel.compute_contrast`)

    title: String, optional
        If string, represents the web page's title and primary heading,
        model type is sub-heading.
        If None, page titles and headings are autogenerated
        using contrast names.

    bg_img: Niimg-like object
        Default is the MNI152 template
        See http://nilearn.github.io/manipulating_images/input_output.html
        The background image for mask and stat maps to be plotted on upon.
        To turn off background image, just pass "bg_img=None".

    threshold: float
        Default is 3.09
        Cluster forming threshold in same scale as `stat_img` (either a
        t-scale or z-scale value). Used only if height_control is None.

    alpha: float
        Default is 0.001
        Number controlling the thresholding (either a p-value or q-value).
        Its actual meaning depends on the height_control parameter.
        This function translates alpha to a z-scale threshold.

    cluster_threshold: int, optional
        Default is 0
        Cluster size threshold, in voxels.

    height_control: string or None
        false positive control meaning of cluster forming
        threshold: 'fpr' (default) or 'fdr' or 'bonferroni' or None

    min_distance: `float`
        For display purposes only.
        Minimum distance between subpeaks in mm. Default is 8 mm.

    plot_type: String. ['slice' (default) or  'glass']
        Specifies the type of plot to be drawn for the statistical maps.

    display_mode: string
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

    report_dims: Sequence[int, int]
        Default is (1600, 800) pixels.
        Specifies width, height (in pixels) of report window within a notebook.
        Only applicable when inserting the report into a Jupyter notebook.
        Can be set after report creation using report.width, report.height.

    Returns
    -------
    report_text: HTMLDocument Object
        Contains the HTML code for the GLM Report.

    """
    '''
    Bug in Pandas 0.18 : https://github.com/pandas-dev/pandas/issues/13257
    pd.set_option('display.precision', 2)
    limits number of digits shown instead of precision.
    Hence pd.option_context('display.precision', 2) has been used.
    '''
    display_mode_selector = {'slice': 'z', 'glass': 'lzry'}
    if not display_mode:
        display_mode = display_mode_selector[plot_type]

    try:
        design_matrices = model.design_matrices_
    except AttributeError:
        design_matrices = [model.design_matrix_]

    html_template_path = os.path.join(HTML_TEMPLATE_ROOT_PATH,
                                      'report_template.html')
    with open(html_template_path) as html_file_obj:
        html_template_text = html_file_obj.read()
    report_template = string.Template(html_template_text)

    contrasts = _coerce_to_dict(contrasts)
    contrast_plots = _plot_contrasts(contrasts, design_matrices)
    page_title, page_heading_1, page_heading_2 = _make_headings(
        contrasts,
        title,
        model,
    )
    with pd.option_context('display.max_colwidth', 100):
        model_attributes = _model_attributes_to_dataframe(model)
        model_attributes_html = _dataframe_to_html(model_attributes,
                                                   precision=2,
                                                   header=False,
                                                   sparsify=False,
                                                   )
    statistical_maps = make_stat_maps(model, contrasts)
    html_design_matrices = _dmtx_to_svg_url(design_matrices)
    mask_img = model.mask_img or model.masker_.mask_img_
    mask_plot_html_code = _mask_to_svg(mask_img=mask_img,
                                       bg_img=bg_img,
                                       )
    all_components = _make_stat_maps_contrast_clusters(
        stat_img=statistical_maps,
        contrasts_plots=contrast_plots,
        threshold=threshold,
        alpha=alpha,
        cluster_threshold=cluster_threshold,
        height_control=height_control,
        min_distance=min_distance,
        bg_img=bg_img,
        display_mode=display_mode,
        plot_type=plot_type,
    )
    all_components_text = '\n'.join(all_components)
    report_values = {'page_title': escape(page_title),
                     'page_heading_1': page_heading_1,
                     'page_heading_2': page_heading_2,
                     'model_attributes': model_attributes_html,
                     'all_contrasts_with_plots': ''.join(
                         contrast_plots.values()),
                     'design_matrices': html_design_matrices,
                     'mask_plot': mask_plot_html_code,
                     'component': all_components_text,
                     }
    report_text = report_template.safe_substitute(**report_values)
    report_text = HTMLDocument(report_text)
    # setting report size for better visual experience in Jupyter Notebooks.
    report_text.width, report_text.height = _check_report_dims(report_dims)
    return report_text


def _check_report_dims(report_size):
    """
    Warns user & reverts to default if report dimensions are non-numerical.

    Parameters
    ----------
    report_size: Tuple[int, int]
        Report width, height in jupyter notebook.

    Returns
    -------
    report_size: Tuple[int, int]
        Valid values for report width, height in jupyter notebook.

    """
    width, height = report_size
    try:
        width = int(width)
        height = int(height)
    except ValueError:
        warnings.warn('Report size has invalid values. '
                      'Using default 1600x800')
        width, height = (1600, 800)
    return width, height


def _coerce_to_dict(input_arg):
    """ Constructs a dict from the provided arg.

    If input_arg is:
      dict then returns it unchanged.

      string or collection of Strings or Sequence[int],
      returns a dict {str(value): value, ...}

    Parameters
    ----------
    input_arg: String or Collection[str or Int or Sequence[Int]]
     or Dict[str, str or np.array]
        Can be of the form:
         'string'
         ['string_1', 'string_2', ...]
         list/array
         [list/array_1, list/array_2, ...]
         {'string_1': list/array1, ...}

    Returns
    -------
    Dict[str, np.array or str]

    """
    if not isinstance(input_arg, dict):
        if isinstance(input_arg, Iterable):
            if not isinstance(input_arg[0], Iterable):
                input_arg = [input_arg]
        input_arg = [input_arg] if isinstance(input_arg, str) else input_arg
        input_arg = {str(contrast_): contrast_ for contrast_ in input_arg}
    return input_arg


def plot_to_svg(plot):
    """
    Creates an SVG image as a data URL
    from a Matplotlib Axes or Figure object.

    Parameters
    ----------
    plot: Matplotlib Axes or Figure object
        Contains the plot information.

    Returns
    -------
    url_plot_svg: String
        SVG Image Data URL
    """
    with io.BytesIO() as buffer:
        try:
            plot.figure.savefig(buffer, format='svg')
        except AttributeError:
            plot.savefig(buffer, format='svg')
        svg_plot = buffer.getvalue()
    try:
        url_svg_plot = quote(svg_plot.decode('utf8'))
    except KeyError:  # Fails on Python2.
        url_svg_plot = quote(svg_plot)
    return url_svg_plot


def _plot_contrasts(contrasts, design_matrices):
    """
    Accepts dict of contrasts and list of design matrices and generates
    a dict of contrast titles & HTML for SVG Image data url
    for corresponding contrast plot.

    Parameters
    ----------
    contrasts: Dict[str, np.array or str]
        Contrast information, as a dict
          {'contrast_title_1, contrast_info_1/title_1, ...}

    design_matrices: List[pd.Dataframe]
        Design matrices computed in the model.

    Returns
    -------
    contrast_plots: Dict[str, svg img]
        Dict of contrast title and svg image data url
        for corresponding contrast plot.
    """
    all_contrasts_plots = {}
    contrast_template_path = os.path.join(HTML_TEMPLATE_ROOT_PATH,
                                          'contrast_template.html'
                                          )
    with open(contrast_template_path) as html_template_obj:
        contrast_template_text = html_template_obj.read()

    for design_matrix in design_matrices:
        for contrast_name, contrast_data in contrasts.items():
            contrast_text_ = string.Template(contrast_template_text)
            contrast_plot = plot_contrast_matrix(contrast_data, design_matrix,
                                                 colorbar=True)
            contrast_plot.set_xlabel(contrast_name)
            contrast_plot.figure.set_figheight(2)
            contrast_plot.figure.set_tight_layout(True)
            url_contrast_plot_svg = plot_to_svg(contrast_plot)
            # prevents sphinx-gallery & jupyter
            # from scraping & inserting plots
            plt.close()
            contrasts_for_subsitution = {
                'contrast_plot': url_contrast_plot_svg,
                'contrast_name': contrast_name,
            }
            contrast_text_ = contrast_text_.safe_substitute(
                contrasts_for_subsitution
            )
            all_contrasts_plots[contrast_name] = contrast_text_
    return all_contrasts_plots


def _make_headings(contrasts, title, model):
    """ Creates report page title, heading & sub-heading
     using title text or contrast names.
    Accepts contrasts and user supplied title string or
    contrasts and user supplied 3 element list or tuple.

    If title is not in (None, 'auto'),
    page title == heading,
    model type == sub-heading

    Parameters
    ----------
    contrasts: Dict[str, np.array or str]
        Contrast information, as a dict in the form
            {'contrast_title_1': contrast_info_1/title_1, ...}
        Contrast titles are used in page title and secondary heading
        if `title` is not 'auto' or None.

    title: String or List/Tuple with 3 elements
        User supplied text for HTML Page title and primary heading.
        Or 3 element List/Tuple for Title Heading, sub-heading resp.
        Overrides title auto-generation.

    model: FirstLevelModel or SecondLevelModel
        The model, passed in to determine its type
        to be used in page title & headings.

    Returns
    -------
    (HTML page title, heading, sub-heading): Tuple[str, str, str]
        If title is user-supplied, then subheading is empty string.
    """
    if type(model) == stats.first_level_model.FirstLevelModel:
        model_type = 'First Level Model'
    elif type(model) == stats.second_level_model.SecondLevelModel:
        model_type = 'Second Level Model'

    if title:
        return title, title, model_type

    contrasts_names = sorted(list(contrasts.keys()))
    contrasts_text = ', '.join(contrasts_names)

    page_title = 'Report: {} for {}'.format(model_type, contrasts_text)
    page_heading_1 = 'Statistical Report for {}'.format(contrasts_text)
    page_heading_2 = model_type
    return page_title, page_heading_1, page_heading_2


def _model_attributes_to_dataframe(model):
    """ Returns an HTML table with pertinent model attributes & information.

    Parameters
    ----------
    model: FirstLevelModel or SecondLevelModel object.

    Returns
    -------
    HTML Table: String
        HTML table with the pertinent attributes of the model.
    """
    selected_attributes = [
        'subject_label',
        'drift_model',
        'hrf_model',
        'standardize',
        'noise_model',
        't_r',
        'high_pass',
        'target_shape',
        'signal_scaling',
        'drift_order',
        'scaling_axis',
        'smoothing_fwhm',
        'target_affine',
        'slice_time_ref',
        'fir_delays',
    ]
    attribute_units = {
        't_r': 's',
        'high_pass': 'Hz',
    }

    selected_attributes.sort()
    display_attributes = OrderedDict(
        (attr_name, getattr(model, attr_name))
        for attr_name in selected_attributes
        if hasattr(model, attr_name)
    )
    model_attributes = pd.DataFrame.from_dict(display_attributes,
                                              orient='index',
                                              )
    attribute_names_with_units = {
        attribute_name_: attribute_name_ + ' ({})'.format(attribute_unit_)
        for attribute_name_, attribute_unit_
        in attribute_units.items()
    }
    model_attributes.rename(index=attribute_names_with_units,
                            inplace=True)
    return model_attributes


def make_stat_maps(model, contrasts):
    """ Given a model and contrasts, return the corresponding z-maps

    Parameters
    ----------
    model: FirstLevelModel or SecondLevelModel object
        Must have a fitted design matrix(ces).

    contrasts: Dict[str, ndarray or str]
        Dict of contrasts for a first or second level model.
        Corresponds to the contrast_def for the FirstLevelModel
        (nilearn.stats.first_level_model.FirstLevelModel.compute_contrast)
        & second_level_contrast for a SecondLevelModel
        (nilearn.stats.second_level_model.SecondLevelModel.compute_contrast)

    Returns
    -------
    statistical_maps: Dict[str, niimg]
        Dict of statistical z-maps keyed to contrast names/titles.

    See Also
    --------
    nilearn.stats.first_level_model.FirstLevelModel.compute_contrast
    nilearn.stats.second_level_model.SecondLevelModel.compute_contrast
    """
    statistical_maps = {contrast_id: model.compute_contrast(contrast_val)
                        for contrast_id, contrast_val in contrasts.items()
                        }
    return statistical_maps


def _dmtx_to_svg_url(design_matrices):
    """ Accepts a FirstLevelModel or SecondLevelModel object
    with fitted design matrices & generates SVG Image URL,
    which can be inserted into an HTML template.

    Parameters
    ----------
    design_matrices: List[pd.Dataframe]
        Design matrices computed in the model.

    Returns
    -------
    svg_url_design_matrices: String
        SVG Image URL for the plotted design matrices,
    """
    html_design_matrices = []
    dmtx_template_path = os.path.join(HTML_TEMPLATE_ROOT_PATH,
                                      'design_matrix_template.html'
                                      )
    with open(dmtx_template_path) as html_template_obj:
        dmtx_template_text = html_template_obj.read()

    for dmtx_count, design_matrix in enumerate(design_matrices, start=1):
        dmtx_text_ = string.Template(dmtx_template_text)
        dmtx_plot = plot_design_matrix(design_matrix)
        dmtx_title = 'Session {}'.format(dmtx_count)
        plt.title(dmtx_title, y=0.987)
        dmtx_plot = _resize_plot_inches(dmtx_plot, height_change=.3)
        url_design_matrix_svg = plot_to_svg(dmtx_plot)
        # prevents sphinx-gallery & jupyter from scraping & inserting plots
        plt.close()
        dmtx_text_ = dmtx_text_.safe_substitute(
            {'design_matrix': url_design_matrix_svg,
             'dmtx_title': dmtx_title,
             }
        )
        html_design_matrices.append(dmtx_text_)
    svg_url_design_matrices = ''.join(html_design_matrices)
    return svg_url_design_matrices


def _resize_plot_inches(plot, width_change=0, height_change=0):
    """
    Accepts a matplotlib figure or axes object and resizes it (in inches).
    Returns the original object.

    Parameters
    ----------
    plot: matplotlib.Figure() or matplotlib.Axes()
        The matplotlib Figure/Axes object to be resized.

    width_change: `float`
        The amount of change to be added on to original width.
        Use negative values for reducing figure dimensions.

    height_change `float`
        The amount of change to be added on to original height.
        Use negative values for reducing figure dimensions.

    Returns
    -------
    plot: matplotlib.Figure() or matplotlib.Axes()
        The matplotlib Figure/Axes object after being resized.
    """
    try:
        orig_size = plot.figure.get_size_inches()
    except AttributeError:
        orig_size = plot.get_size_inches()
    new_size = (orig_size[0] + width_change,
                orig_size[1] + height_change,
                )
    try:
        plot.figure.set_size_inches(new_size, forward=True)
    except AttributeError:
        plot.set_size_inches(new_size)
    return plot


def _mask_to_svg(mask_img, bg_img):
    """
    Plot cuts of an mask image and creates SVG code of it.

    Parameters
    ----------
    mask_img : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        The mask image; it could be binary mask or an atlas or ROIs
        with integer values.

    bg_img : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        The background image that the mask will be plotted on top of.
        To turn off background image, just pass "bg_img=None".

    Returns
    -------
    mask_plot_svg: str
        SVG Image Data URL for the mask plot.
    """
    if mask_img:
        mask_plot = plot_roi(roi_img=mask_img,  # noqa: F841
                             bg_img=bg_img,
                             display_mode='z',
                             cmap='Set1',
                             )
        mask_plot_svg = plot_to_svg(plt.gcf())
        # prevents sphinx-gallery & jupyter from scraping & inserting plots
        plt.close()
    else:
        mask_plot_svg = None  # HTML image tag's alt attribute is used.
    return mask_plot_svg


def _make_stat_maps_contrast_clusters(stat_img, contrasts_plots, threshold,
                                      alpha,
                                      cluster_threshold, height_control,
                                      min_distance, bg_img,
                                      display_mode, plot_type):
    """ Populates a smaller HTML sub-template with the proper values,
     make a list containing one or more of such components
     & returns the list to be inserted into the HTML Report Template.
    Each component contains the HTML code for
    a contrast & its corresponding statistical maps & cluster table;

    Parameters
    ----------
    stat_img : Niimg-like object or None
       statistical image (presumably in z scale)
       whenever height_control is 'fpr' or None,
       stat_img=None is acceptable.
       If it is 'fdr' or 'bonferroni',
       an error is raised if stat_img is None.

    contrasts_plots: Dict[str, str]
        Contains contrast names & HTML code of the contrast's SVG plot.

    threshold: float
       desired threshold in z-scale.
       This is used only if height_control is None

    alpha: float
        number controlling the thresholding (either a p-value or q-value).
        Its actual meaning depends on the height_control parameter.
        This function translates alpha to a z-scale threshold.

    cluster_threshold : float
        cluster size threshold. In the returned thresholded map,
        sets of connected voxels (`clusters`) with size smaller
        than this number will be removed.

    height_control: string
        false positive control meaning of cluster forming
        threshold: 'fpr' or 'fdr' or 'bonferroni' or None

    min_distance: `float`
        For display purposes only.
        Minimum distance between subpeaks in mm. Default is 8 mm.

    bg_img : Niimg-like object
        Only used when plot_type is 'slice'.
        See http://nilearn.github.io/manipulating_images/input_output.html
        The background image for stat maps to be plotted on upon.
        If nothing is specified, the MNI152 template will be used.
        To turn off background image, just pass "bg_img=False".

    display_mode: string
        Choose the direction of the cuts:
        'x' - sagittal, 'y' - coronal, 'z' - axial,
        'l' - sagittal left hemisphere only,
        'r' - sagittal right hemisphere only,
        'ortho' - three cuts are performed in orthogonal directions.

        Possible values are:
        'ortho', 'x', 'y', 'z', 'xz', 'yx', 'yz',
        'l', 'r', 'lr', 'lzr', 'lyr', 'lzry', 'lyrz'.

    plot_type: string
        ['slice', 'glass']
        The type of plot to be drawn.

    Returns
    -------
    all_components: List[String]
        Each element is a set of HTML code for
        contrast name, contrast plot, statistical map, cluster table.
    """
    all_components = []
    components_template_path = os.path.join(
        HTML_TEMPLATE_ROOT_PATH,
        'stat_maps_contrast_clusters_template.html'
    )
    with open(components_template_path) as html_template_obj:
        components_template_text = html_template_obj.read()
    for contrast_name, stat_map_img in stat_img.items():
        component_text_ = string.Template(components_template_text)
        thresholded_stat_map, threshold = threshold_stats_img(
            stat_img=stat_map_img,
            threshold=threshold,
            alpha=alpha,
            cluster_threshold=cluster_threshold,
            height_control=height_control,
        )
        table_details = _clustering_params_to_dataframe(threshold,
                                                        cluster_threshold,
                                                        min_distance,
                                                        height_control,
                                                        alpha,
                                                        )
        stat_map_svg = _stat_map_to_svg(
            stat_img=thresholded_stat_map,
            bg_img=bg_img,
            display_mode=display_mode,
            plot_type=plot_type,
            table_details=table_details,
        )
        cluster_table = get_clusters_table(
            stat_map_img,
            stat_threshold=threshold,
            cluster_threshold=cluster_threshold,
            min_distance=min_distance,
        )

        cluster_table_html = _dataframe_to_html(cluster_table,
                                                precision=2,
                                                index=False,
                                                classes='cluster-table',
                                                )
        table_details_html = _dataframe_to_html(
            table_details,
            precision=2,
            header=False,
            classes='cluster-details-table',
        )
        components_values = {
            'contrast_name': escape(contrast_name),
            'contrast_plot': contrasts_plots[contrast_name],
            'stat_map_img': stat_map_svg,
            'cluster_table_details': table_details_html,
            'cluster_table': cluster_table_html,
        }
        component_text_ = component_text_.safe_substitute(**components_values)
        all_components.append(component_text_)
    return all_components


def _clustering_params_to_dataframe(threshold,
                                    cluster_threshold,
                                    min_distance,
                                    height_control,
                                    alpha,
                                    ):
    """
    Creates a Pandas DataFrame from the supplied arguments.
    For use as part of the Cluster Table.

    Parameters
    ----------
    threshold: `float`
        Cluster forming threshold in same scale as `stat_img` (either a
        p-value or z-scale value).

    cluster_threshold : `int` or `None`, optional
        Cluster size threshold, in voxels.

    min_distance: `float`
        For display purposes only.
        Minimum distance between subpeaks in mm. Default is 8 mm.

    height_control: string or None
        false positive control meaning of cluster forming
        threshold: 'fpr' (default) or 'fdr' or 'bonferroni' or None

    alpha: float
        Default is 0.001
        Number controlling the thresholding (either a p-value or q-value).
        Its actual meaning depends on the height_control parameter.
        This function translates alpha to a z-scale threshold.

    Returns
    -------
    Pandas.DataFrame
        Dataframe with clustering parameters
    """
    table_details = OrderedDict()
    threshold = np.around(threshold, 3)
    if height_control:
        table_details.update({'Height control': height_control})
        '''
        HTMLDocument.get_iframe() invoked in Python2 Jupyter Notebooks
        mishandles certain unicode characters
        & raises error due to greek alpha symbol.
        This is simpler than overloading the class using inheritance,
        especially given limited Python2 use at time of release.
        '''
        if os.sys.version_info.major == 2:
            table_details.update({'alpha': alpha})
        else:
            table_details.update({u'\u03B1': alpha})
        table_details.update({'Threshold (computed)': threshold})
    else:
        table_details.update({'Height control': 'None'})
        table_details.update({'Threshold Z': threshold})
    table_details.update(
        {'Cluster size threshold (voxels)': cluster_threshold}
    )
    table_details.update({'Minimum distance (mm)': min_distance})
    table_details = pd.DataFrame.from_dict(table_details,
                                           orient='index',
                                           )
    return table_details


def _stat_map_to_svg(stat_img,
                     bg_img,
                     display_mode,
                     plot_type,
                     table_details,
                     ):
    """ Generates SVG code for a statistical map,
    including its clustering parameters.

    Parameters
    ----------
    stat_img : Niimg-like object or None
       Statistical image (presumably in z scale),
       to be plotted as slices or glass brain.
       Does not perform any thresholding.

    bg_img : Niimg-like object
        Only used when plot_type is 'slice'.
        See http://nilearn.github.io/manipulating_images/input_output.html
        The background image for stat maps to be plotted on upon.
        If nothing is specified, the MNI152 template will be used.
        To turn off background image, just pass "bg_img=False".

    display_mode: string
        Choose the direction of the cuts:
        'x' - sagittal, 'y' - coronal, 'z' - axial,
        'l' - sagittal left hemisphere only,
        'r' - sagittal right hemisphere only,
        'ortho' - three cuts are performed in orthogonal directions.

        Possible values are:
        'ortho', 'x', 'y', 'z', 'xz', 'yx', 'yz',
        'l', 'r', 'lr', 'lzr', 'lyr', 'lzry', 'lyrz'.

    plot_type: string
        ['slice', 'glass']
        The type of plot to be drawn.

    table_details: pandas.Dataframe
        Dataframe listing the parameters used for clustering,
        to be included in the plot.

    Returns
    -------
    stat_map_svg: string
        SVG Image Data URL representing a statistical map.
    """
    if plot_type == 'slice':
        stat_map_plot = plot_stat_map(stat_img,
                                      bg_img=bg_img,
                                      display_mode=display_mode,
                                      )
    elif plot_type == 'glass':
        stat_map_plot = plot_glass_brain(stat_img,
                                         display_mode=display_mode,
                                         colorbar=True,
                                         plot_abs=False,
                                         )
    else:
        raise ValueError('Invalid plot type provided. Acceptable options are'
                         "'slice' or 'glass'.")
    with pd.option_context('display.precision', 2):
        stat_map_plot = _add_params_to_plot(table_details, stat_map_plot)
    fig = plt.gcf()
    stat_map_svg = plot_to_svg(fig)
    # prevents sphinx-gallery & jupyter from scraping & inserting plots
    plt.close()
    return stat_map_svg


def _add_params_to_plot(table_details, stat_map_plot):
    """
    Inserts thresholding parameters into the stat map plot as figure suptitle.

    Parameters
    ----------
    table_details: Dict[String, Any]
        Dict of parameters and values used in thresholding.

    stat_map_plot: matplotlib.Axes
        Axes object of the stat map plot.

    Returns
    -------
    stat_map_plot: matplotlib.Axes
        Axes object of the stat map plot, with the added suptitle .
    """
    thresholding_params = [':'.join([name, str(val)]) for name, val in
                           table_details[0].items()]
    thresholding_params = '  '.join(thresholding_params)
    suptitle_text = plt.suptitle(thresholding_params,
                                 fontsize=11,
                                 x=.45,
                                 wrap=True,
                                 )
    fig = list(stat_map_plot.axes.values())[0].ax.figure
    fig = _resize_plot_inches(plot=fig,
                              width_change=.2,
                              height_change=1,
                              )
    if stat_map_plot._black_bg:
        suptitle_text.set_color('w')
    return stat_map_plot


def _dataframe_to_html(df, precision, **kwargs):
    """ Makes HTML table from provided dataframe.
    Removes HTML5 non-compliant attributes (ex: `border`).

    Parameters
    ----------
    df: pandas.Dataframe
        Dataframe to be converted into HTML table.

    precision: int
        The display precision for float values in the table.

    **kwargs: keyworded arguments
        Supplies keyworded arguments for func: pandas.Dataframe.to_html()

    Returns
    -------
    html_table: String
        Code for HTML table.
    """
    with pd.option_context('display.precision', precision):
        html_table = df.to_html(**kwargs)
    html_table = html_table.replace('border="1" ', '')
    return html_table

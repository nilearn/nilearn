"""Utility functions that do not require matplotlib."""

import os
import warnings
from collections import OrderedDict
from collections.abc import Iterable
from decimal import Decimal

import numpy as np
import pandas as pd

from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel


def check_report_dims(report_size):
    """Warns user & reverts to default if report dimensions are non-numerical.

    Parameters
    ----------
    report_size : Tuple[int, int]
        Report width, height in jupyter notebook.

    Returns
    -------
    report_size : Tuple[int, int]
        Valid values for report width, height in jupyter notebook.

    """
    width, height = report_size
    try:
        width = int(width)
        height = int(height)
    except ValueError:
        warnings.warn(
            "Report size has invalid values. Using default 1600x800",
            stacklevel=3,
        )
        width, height = (1600, 800)
    return width, height


def clustering_params_to_dataframe(
    threshold,
    cluster_threshold,
    min_distance,
    height_control,
    alpha,
):
    """Create a Pandas DataFrame from the supplied arguments.

    For use as part of the Cluster Table.

    Parameters
    ----------
    threshold : float
        Cluster forming threshold in same scale as `stat_img` (either a
        p-value or z-scale value).

    cluster_threshold : int or None
        Cluster size threshold, in voxels.

    min_distance : float, default=8
        For display purposes only.
        Minimum distance between subpeaks in mm.

    height_control : string or None
        False positive control meaning of cluster forming
        threshold: 'fpr' (default) or 'fdr' or 'bonferroni' or None

    alpha : float
        Number controlling the thresholding (either a p-value or q-value).
        Its actual meaning depends on the height_control parameter.
        This function translates alpha to a z-scale threshold.

    Returns
    -------
    table_details : Pandas.DataFrame
        Dataframe with clustering parameters.

    """
    table_details = OrderedDict()
    threshold = np.around(threshold, 3)
    if height_control:
        table_details.update({"Height control": height_control})
        # HTMLDocument.get_iframe() invoked in Python2 Jupyter Notebooks
        # mishandles certain unicode characters
        # & raises error due to greek alpha symbol.
        # This is simpler than overloading the class using inheritance,
        # especially given limited Python2 use at time of release.
        if alpha < 0.001:
            alpha = f"{Decimal(alpha):.2E}"
        if os.sys.version_info.major == 2:
            table_details.update({"alpha": alpha})
        else:
            table_details.update({"\u03b1": alpha})
        table_details.update({"Threshold (computed)": threshold})
    else:
        table_details.update({"Height control": "None"})
        table_details.update({"Threshold Z": threshold})
    table_details.update(
        {"Cluster size threshold (voxels)": cluster_threshold}
    )
    table_details.update({"Minimum distance (mm)": min_distance})
    table_details = pd.DataFrame.from_dict(
        table_details,
        orient="index",
    )
    return table_details


def coerce_to_dict(input_arg):
    """Construct a dict from the provided arg.

    If input_arg is:
      dict or None then returns it unchanged.

      string or collection of Strings or Sequence[int],
      returns a dict {str(value): value, ...}

    Parameters
    ----------
    input_arg : String or Collection[str or Int or Sequence[Int]]
     or Dict[str, str or np.array] or None
        Can be of the form:
         'string'
         ['string_1', 'string_2', ...]
         list/array
         [list/array_1, list/array_2, ...]
         {'string_1': list/array1, ...}

    Returns
    -------
    input_args: Dict[str, np.array or str] or None

    """
    if input_arg is None:
        return None
    if not isinstance(input_arg, dict):
        if isinstance(input_arg, Iterable) and not isinstance(
            input_arg[0], Iterable
        ):
            input_arg = [input_arg]
        input_arg = [input_arg] if isinstance(input_arg, str) else input_arg
        input_arg = {str(contrast_): contrast_ for contrast_ in input_arg}
    return input_arg


def make_headings(contrasts, title, model):
    """Create report page title, heading & sub-heading \
    using title text or contrast names.

    Accepts contrasts and user supplied title string or
    contrasts and user supplied 3 element list or tuple.

    If title is not in (None, 'auto'),
    page title == heading,
    model type == sub-heading

    Parameters
    ----------
    contrasts : Dict[str, np.array or str]
        Contrast information, as a dict in the form
            {'contrast_title_1': contrast_info_1/title_1, ...}
        Contrast titles are used in page title and secondary heading
        if `title` is not 'auto' or None.

    title : String or List/Tuple with 3 elements
        User supplied text for HTML Page title and primary heading.
        Or 3 element List/Tuple for Title Heading, sub-heading resp.
        Overrides title auto-generation.

    model : FirstLevelModel or SecondLevelModel
        The model, passed in to determine its type
        to be used in page title & headings.

    Returns
    -------
    (HTML page title, heading, sub-heading) : Tuple[str, str, str]
        If title is user-supplied, then subheading is empty string.

    """
    model_type = return_model_type(model)

    if title:
        return title, title, model_type

    contrasts_names = sorted(contrasts.keys())
    contrasts_text = ", ".join(contrasts_names)

    page_title = f"Report: {model_type} for {contrasts_text}"
    page_heading_1 = f"Statistical Report for {contrasts_text}"
    page_heading_2 = model_type
    return page_title, page_heading_1, page_heading_2


def make_stat_maps(model, contrasts, output_type="z_score"):
    """Given a model and contrasts, return the corresponding z-maps.

    Parameters
    ----------
    model : FirstLevelModel or SecondLevelModel object
        Must have a fitted design matrix(ces).

    contrasts : Dict[str, ndarray or str]
        Dict of contrasts for a first or second level model.
        Corresponds to the contrast_def for the FirstLevelModel
        (nilearn.glm.first_level.FirstLevelModel.compute_contrast)
        & second_level_contrast for a SecondLevelModel
        (nilearn.glm.second_level.SecondLevelModel.compute_contrast)

    output_type : :obj:`str`, default='z_score'
        The type of statistical map to retain from the contrast.

        .. versionadded:: 0.9.2

    Returns
    -------
    statistical_maps : Dict[str, niimg]
        Dict of statistical z-maps keyed to contrast names/titles.

    See Also
    --------
    nilearn.glm.first_level.FirstLevelModel.compute_contrast
    nilearn.glm.second_level.SecondLevelModel.compute_contrast

    """
    statistical_maps = {
        contrast_id: model.compute_contrast(
            contrast_val,
            output_type=output_type,
        )
        for contrast_id, contrast_val in contrasts.items()
    }
    return statistical_maps


def model_attributes_to_dataframe(model, is_volume_glm=True):
    """Return an HTML table with pertinent model attributes & information.

    Parameters
    ----------
    model : Any masker or FirstLevelModel or SecondLevelModel object.

    is_volume_glm : bool, optional, default=True
        Whether the GLM model is for a volume image or not. Only relevant for
        FirstLevelModel and SecondLevelModel objects.

    Returns
    -------
    attributes_df: pandas.DataFrame
        DataFrame with the pertinent attributes of the model.
    """
    if model.__class__.__name__ in ["FirstLevelModel", "SecondLevelModel"]:
        return _glm_model_attributes_to_dataframe(
            model, is_volume_glm=is_volume_glm
        )
    else:
        attributes_df = OrderedDict(
            (
                attr_name,
                (
                    str(getattr(model, attr_name))
                    if isinstance(getattr(model, attr_name), dict)
                    else getattr(model, attr_name)
                ),
            )
            for attr_name in model.get_params()
        )
        attributes_df = pd.DataFrame.from_dict(attributes_df, orient="index")
        attributes_df.index.names = ["Parameter"]
        attributes_df.columns = ["Value"]
        return attributes_df


def _glm_model_attributes_to_dataframe(model, is_volume_glm=True):
    """Return a pandas dataframe with pertinent model attributes & information.

    Parameters
    ----------
    model : FirstLevelModel or SecondLevelModel object.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the pertinent attributes of the model.
    """
    selected_attributes = [
        "subject_label",
        "drift_model",
        "hrf_model",
        "standardize",
        "noise_model",
        "t_r",
        "signal_scaling",
        "scaling_axis",
        "smoothing_fwhm",
        "slice_time_ref",
    ]
    if is_volume_glm:
        selected_attributes.extend(["target_shape", "target_affine"])
    if hasattr(model, "hrf_model") and model.hrf_model == "fir":
        selected_attributes.append("fir_delays")
    if hasattr(model, "drift_model"):
        if model.drift_model == "cosine":
            selected_attributes.append("high_pass")
        elif model.drift_model == "polynomial":
            selected_attributes.append("drift_order")

    attribute_units = {
        "t_r": "seconds",
        "high_pass": "Hertz",
    }

    selected_attributes.sort()
    display_attributes = OrderedDict(
        (attr_name, getattr(model, attr_name))
        for attr_name in selected_attributes
        if hasattr(model, attr_name)
    )
    model_attributes = pd.DataFrame.from_dict(
        display_attributes,
        orient="index",
    )
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


def dataframe_to_html(df, precision, **kwargs):
    """Make HTML table from provided dataframe.

    Removes HTML5 non-compliant attributes (ex: `border`).

    Parameters
    ----------
    df : pandas.Dataframe
        Dataframe to be converted into HTML table.

    precision : int
        The display precision for float values in the table.

    **kwargs : keyworded arguments
        Supplies keyworded arguments for func: pandas.Dataframe.to_html()

    Returns
    -------
    html_table : String
        Code for HTML table.

    """
    with pd.option_context("display.precision", precision):
        html_table = df.to_html(**kwargs)
    html_table = html_table.replace('border="1" ', "")
    return html_table.replace('class="dataframe"', 'class="pure-table"')


def return_model_type(model):
    if isinstance(model, FirstLevelModel):
        return "First Level Model"
    elif isinstance(model, SecondLevelModel):
        return "Second Level Model"

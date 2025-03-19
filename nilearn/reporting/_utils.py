"""Utility functions that do not require matplotlib."""

import warnings
from collections import OrderedDict
from decimal import Decimal

import numpy as np
import pandas as pd


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
    is_volume_glm,
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

    min_distance : float
        For display purposes only.
        Minimum distance between subpeaks in mm.

    height_control : string or None
        False positive control meaning of cluster forming
        threshold: 'fpr' (default) or 'fdr' or 'bonferroni' or None

    alpha : float
        Number controlling the thresholding (either a p-value or q-value).
        Its actual meaning depends on the height_control parameter.
        This function translates alpha to a z-scale threshold.

    is_volume_glm: bool
        True if we are dealing with volume data.

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
        table_details.update({"\u03b1": alpha})
        table_details.update({"Threshold (computed)": threshold})
    else:
        table_details.update({"Height control": "None"})
        table_details.update({"Threshold Z": threshold})

    if is_volume_glm:
        table_details.update(
            {"Cluster size threshold (voxels)": cluster_threshold}
        )
        table_details.update({"Minimum distance (mm)": min_distance})

    table_details = pd.DataFrame.from_dict(
        table_details,
        orient="index",
    )

    return table_details


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


def model_attributes_to_dataframe(model):
    """Return dataframe with pertinent model attributes & information.

    Parameters
    ----------
    model : Any masker or FirstLevelModel or SecondLevelModel object.

    Returns
    -------
    attributes_df: pandas.DataFrame
        DataFrame with the pertinent attributes of the model.
    """
    if model.__class__.__name__ in ["FirstLevelModel", "SecondLevelModel"]:
        return _glm_model_attributes_to_dataframe(model)
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

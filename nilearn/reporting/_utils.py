"""Utility functions that do not require matplotlib."""

import warnings
from collections import OrderedDict

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
    model : Any masker object.

    Returns
    -------
    attributes_df: pandas.DataFrame
        DataFrame with the pertinent attributes of the model.
    """
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

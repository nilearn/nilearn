"""Utility functions that do not require matplotlib."""

from collections import OrderedDict

import pandas as pd


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

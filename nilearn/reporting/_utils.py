"""Utility functions that do not require matplotlib."""

from collections import OrderedDict

import pandas as pd

from nilearn._utils.niimg import repr_niimgs
from nilearn.typing import NiimgLike


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
    html_table = html_table.replace("\\n", "<br>")
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
    attributes = []
    for attr_name in model.get_params():
        if isinstance(getattr(model, attr_name), dict):
            attributes.append((attr_name, str(getattr(model, attr_name))))
        elif getattr(model, attr_name) is not None:
            attributes.append((attr_name, getattr(model, attr_name)))
    attributes = OrderedDict(attributes)

    for k, v in attributes.items():
        if isinstance(v, NiimgLike):
            attributes[k] = repr_niimgs(v, shorten=False)

    attributes_df = pd.DataFrame.from_dict(attributes, orient="index")
    attributes_df.index.names = ["Parameter"]
    attributes_df.columns = ["Value"]
    return attributes_df

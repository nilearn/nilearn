"""Utility functions for the reporting module."""

import base64
import io
import urllib.parse
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

TEMPLATE_ROOT_PATH = Path(__file__).parent / "data"

CSS_PATH = TEMPLATE_ROOT_PATH / "css"

HTML_TEMPLATE_PATH = TEMPLATE_ROOT_PATH / "html"

HTML_PARTIALS_PATH = HTML_TEMPLATE_PATH / "partials"


def _figure_to_bytes(fig, format):
    """Save figure as as certain format and return it as bytes."""
    with io.BytesIO() as io_buffer:
        fig.savefig(
            io_buffer, format=format, facecolor="white", edgecolor="white"
        )
        return io_buffer.getvalue()


def figure_to_svg_bytes(fig):
    """Save figure as svg and return it as bytes."""
    return _figure_to_bytes(fig, format="svg")


def figure_to_png_bytes(fig):
    """Save figure as png and return it as bytes."""
    return _figure_to_bytes(fig, format="png")


def figure_to_svg_base64(fig):
    """Save figure as svg and return it as 64 bytes."""
    return base64.b64encode(figure_to_svg_bytes(fig)).decode()


def figure_to_png_base64(fig):
    """Save figure as png and return it as 64 bytes."""
    return base64.b64encode(figure_to_png_bytes(fig)).decode()


def figure_to_svg_quoted(fig):
    """Save figure as svg and return it as quoted string."""
    return urllib.parse.quote(figure_to_svg_bytes(fig).decode("utf-8"))


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
        attributes_df = pd.DataFrame.from_dict(model.get_params()).T
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

import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nilearn._utils.helpers import stringify_path
from nilearn._utils.logger import find_stack_level


# kept here for architectural reasons
# pad_contrast_matrix (that uses expression_to_contrast_vector)
# needs to be imported in plotting and reporting
# and those cannot import from glm
def expression_to_contrast_vector(expression, design_columns):
    """Convert a string describing a :term:`contrast` \
       to a :term:`contrast` vector.

    Parameters
    ----------
    expression : :obj:`str`
        The expression to convert to a vector.

    design_columns : :obj:`list` or array of :obj:`str`
        The column names of the design matrix.

    """
    if expression in design_columns:
        contrast_vector = np.zeros(len(design_columns))
        contrast_vector[list(design_columns).index(expression)] = 1.0
        return contrast_vector

    eye_design = pd.DataFrame(
        np.eye(len(design_columns)), columns=design_columns
    )
    try:
        contrast_vector = eye_design.eval(
            expression, engine="python"
        ).to_numpy()
    except Exception:
        raise ValueError(
            f"The expression ({expression}) is not valid. "
            "This could be due to "
            "defining the contrasts using design matrix columns that are "
            "invalid python identifiers."
        )

    return contrast_vector


def pad_contrast_matrix(contrast_def, design_matrix, verbose=1):
    """Pad contrasts with zeros.

    TODO
    try to refactor with "pad_contrast"
    from nilearn/glm/_utils.py


    Parameters
    ----------
    contrast_def : :class:`numpy.ndarray`, str
        Contrast to be padded

    design_matrix : :class:`pandas.DataFrame`
        Design matrix to use.

    Returns
    -------
    axes : :class:`numpy.ndarray`
        Padded contrast

    """
    design_column_names = design_matrix.columns.tolist()
    if isinstance(contrast_def, str):
        contrast_def = expression_to_contrast_vector(
            contrast_def, design_column_names
        )
    n_columns_design_matrix = len(design_column_names)
    n_columns_contrast_def = (
        contrast_def.shape[0]
        if contrast_def.ndim == 1
        else contrast_def.shape[1]
    )
    horizontal_padding = n_columns_design_matrix - n_columns_contrast_def
    if horizontal_padding == 0:
        return contrast_def
    if verbose:
        warnings.warn(
            (
                f"Contrasts will be padded with {horizontal_padding} "
                "column(s) of zeros."
            ),
            category=UserWarning,
            stacklevel=find_stack_level(),
        )
    contrast_def = np.pad(
        contrast_def,
        ((0, 0), (0, horizontal_padding)),
        "constant",
        constant_values=(0, 0),
    )
    return contrast_def


def check_and_load_tables(tables_to_check, var_name):
    """Load tables.

       Tables will be 'loaded'
       if they are pandas.DataFrame, \
       or a CSV or TSV file that can be loaded to a pandas.DataFrame.

       Numpy arrays will also be appended as is.

    tables_to_check : str or pathlib.Path to a TSV or CSV \
              or pandas.DataFrame or numpy.ndarray or, \
              a list of str or pathlib.Path to a TSV or CSV \
              or pandas.DataFrame or numpy.ndarray
              In the case of CSV file,
              the first column is considered to be index column.
              numpy.ndarray will not be appended to the output.

    var_name : str
               name of the `tables_to_check` passed,
               to print in the error message

    Returns
    -------
    list of pandas.DataFrame or numpy.arrays

    Raises
    ------
    TypeError
    If any of the elements in `tables_to_check` does not have a correct type.
    ValueError
    If a specified path in `tables_to_check`
    cannot be loaded to a pandas.DataFrame.

    """
    if not isinstance(tables_to_check, list):
        tables_to_check = [tables_to_check]

    tables = []
    for table_idx, table in enumerate(tables_to_check):
        table = stringify_path(table)

        if not isinstance(table, (str, pd.DataFrame, np.ndarray)):
            raise TypeError(
                f"{var_name} can only be a pandas DataFrame, "
                "a Path object or a string, or a numpy array. "
                f"A {type(table)} was provided at idx {table_idx}"
            )

        if isinstance(table, str):
            loaded = _read_events_table(table)
            tables.append(loaded)
        elif isinstance(table, (pd.DataFrame, np.ndarray)):
            tables.append(table)

    return tables


def _read_events_table(table_path):
    """Load the contents of the event file specified by `table_path
    DEF_TINY = 1e-50`\
       to a pandas.DataFrame.


    Parameters
    ----------
    table_path : :obj:`str`, :obj:`pathlib.Path`
        Path to a TSV or CSV file. In the case of CSV file,
        the first column is considered to be index column.

    Returns
    -------
    pandas.Dataframe
        Pandas Dataframe with events data loaded from file.

    Raises
    ------
    ValueError
    If file loading fails.
    """
    table_path = Path(table_path)
    if table_path.suffix == ".tsv":
        loaded = pd.read_csv(table_path, sep="\t")
    elif table_path.suffix == ".csv":
        loaded = pd.read_csv(table_path)
    else:
        raise ValueError(
            f"Tables to load can only be TSV or CSV.\nGot {table_path}"
        )
    return loaded


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
        if (
            isinstance(input_arg, Iterable)
            and not isinstance(input_arg[0], Iterable)
        ) or isinstance(input_arg, str):
            input_arg = [input_arg]
        input_arg = {str(contrast_): contrast_ for contrast_ in input_arg}
    return input_arg


def sanitize_contrasts(
    contrasts: dict[str, Any] | None,
) -> None | dict[str, str | np.ndarray | list]:
    contrasts = coerce_to_dict(contrasts)
    if contrasts is not None:
        for k, v in contrasts.items():
            if not isinstance(k, str):
                raise TypeError(
                    f"contrast names must be strings, not {type(k)}"
                )

            if not isinstance(v, (str, np.ndarray, list)):
                raise TypeError(
                    "contrast definitions must be strings or array_likes, "
                    f"not {v.__class__.__name__}"
                )
    return contrasts

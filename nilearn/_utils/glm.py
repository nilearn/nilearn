from collections.abc import Iterable
from pathlib import Path
from typing import Literal, overload

import numpy as np
import pandas as pd

from nilearn._utils.helpers import stringify_path
from nilearn._utils.param_validation import check_is_of_allowed_type


@overload
def validate_design_matrix(
    design_matrix: str | Path | pd.DataFrame,
    output_as: Literal["pd"],
    name: str = ...,
) -> pd.DataFrame: ...


@overload
def validate_design_matrix(
    design_matrix: str | Path | pd.DataFrame,
    output_as: None = ...,
    name: str = ...,
) -> tuple[pd.Index, np.ndarray, list]: ...


def validate_design_matrix(
    design_matrix: str | Path | pd.DataFrame,
    output_as: Literal[None, "pd"] = None,
    name: str = "design_matrix",
) -> pd.DataFrame | tuple[pd.Index, np.ndarray, list]:
    """Check that the provided DataFrame is indeed a valid design matrix \
    descriptor.

    Parameters
    ----------
    design_matrix : :obj:`str`, :obj:`pathlib.Path` or :obj:`pandas.DataFrame`
        Describes a design matrix.
        Can be a TSV or CSV file, or a path to one.

    output_as : ``None`` or ``"pd"``, default=None
        If ``"pd"``, the loaded design matrix is returned as a
        :obj:`pandas.DataFrame`. Otherwise, a triplet of fields
        (``frame_times``, ``matrix``, ``names``) is returned.

    name : :obj:`str`, default="design_matrix"
        Name of the ``design_matrix`` argument, used in error messages.

    Returns
    -------
    loaded_design_matrix : :obj:`pandas.DataFrame`
        Returned only when ``output_as="pd"``.

    frame_times : :obj:`pandas.Index` of shape (n_frames,)
        Sampling times of the design matrix in seconds.

    matrix : :obj:`numpy.ndarray` of shape (n_frames, n_regressors)
        Numerical values for the design matrix.

    names : :obj:`list` of shape (n_regressors,)
        Names of the design matrix columns.
    """
    check_is_of_allowed_type(design_matrix, (str, Path, pd.DataFrame), name)

    loaded_design_matrix: pd.DataFrame = check_and_load_tables(
        design_matrix, name
    )[0]

    if len(loaded_design_matrix.columns) == 0:
        raise ValueError("Design matrices dataframe cannot be empty.")

    if output_as is not None:
        if output_as != "pd":
            raise ValueError(
                f"'output_as' must be None or 'pd'. Got : {output_as}"
            )

        return loaded_design_matrix

    names = list(loaded_design_matrix.keys())
    frame_times = loaded_design_matrix.index
    matrix = loaded_design_matrix.to_numpy()

    return frame_times, matrix, names


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


def _read_events_table(table_path: str | Path) -> pd.DataFrame:
    """Load the contents of the event file specified by `table_path`\
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

    if not table_path.exists():
        raise ValueError(f"The file '{table_path!s}' does not exist.")

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

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import overload

import numpy as np
import pandas as pd

from nilearn._utils.helpers import stringify_path


@overload
def check_and_load_tables(
    tables_to_check: pd.DataFrame
    | str
    | Path
    | Sequence[str | Path | pd.DataFrame],
    var_name: str,
) -> list[pd.DataFrame]: ...


@overload
def check_and_load_tables(
    tables_to_check: np.ndarray
    | Sequence[str | Path | pd.DataFrame | np.ndarray],
    var_name: str,
) -> list[pd.DataFrame | np.ndarray]: ...


def check_and_load_tables(
    tables_to_check: pd.DataFrame
    | str
    | Path
    | np.ndarray
    | Sequence[str | Path | pd.DataFrame | np.ndarray],
    var_name: str,
) -> list[pd.DataFrame] | list[pd.DataFrame | np.ndarray]:
    """Load tables.

       Tables will be 'loaded'
       if they are pandas.DataFrame, \
       or a CSV or TSV file that can be loaded to a pandas.DataFrame.

       Numpy arrays will also be appended as is.

    tables_to_check : :obj:`str` or pathlib.Path to a TSV or CSV \
              or pandas.DataFrame or numpy.ndarray or, \
              a list or tuple of str or pathlib.Path to a TSV or CSV \
              or pandas.DataFrame or numpy.ndarray
              In the case of CSV file,
              the first column is considered to be index column.
              numpy.ndarray will not be appended to the output.

    var_name : :obj:`str`
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
    to_check: Sequence[str | Path | pd.DataFrame | np.ndarray]
    if isinstance(tables_to_check, Sequence) and not isinstance(
        tables_to_check, str
    ):
        to_check = tables_to_check
    else:
        to_check = [tables_to_check]

    tables: list[pd.DataFrame | np.ndarray] = []
    for table_idx, table in enumerate(to_check):
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
    input_arg : :obj:`str` or Collection[str or Int or Sequence[Int]]
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

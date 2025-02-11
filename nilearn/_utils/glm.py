from pathlib import Path

import numpy as np
import pandas as pd

from nilearn._utils import stringify_path


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

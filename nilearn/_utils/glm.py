import numpy as np
import pandas as pd

from nilearn._utils import stringify_path


def check_and_load_tables(tables_, var_name):
    """Check each element in `tables_` either to be a pandas.DataFrame, \
       or a CSV or TSV file that can be loaded to a pandas.DataFrame.

       Load to pandas.DataFrame in the case of file path.

    tables_ : list of str or pathlib.Path to a TSV or CSV \
              or pandas.DataFrame or numpy.ndarray
              In the case of CSV file,
              the first column is considered to be index column.

    var_name : str
               name of the `tables_` passed, to print in the error message

    Returns
    -------
    list of pandas.DataFrame

    Raises
    ------
    TypeError
    If any of the elements in `tables_` does not have a correct type.
    ValueError
    If a specified path in `tables_` can not be loaded to a pandas.DataFrame.

    """
    tables = []
    for table_idx, table in enumerate(tables_):
        table = stringify_path(table)
        if isinstance(table, str):
            loaded = _read_events_table(table)
            tables.append(loaded)
        elif isinstance(table, pd.DataFrame):
            tables.append(table)
        elif isinstance(table, np.ndarray):
            pass
        else:
            raise TypeError(
                f"{var_name} can only be a pandas DataFrame, "
                "a Path object or a string, or a numpy array. "
                f"A {type(table)} was provided at idx {table_idx}"
            )
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
    try:
        # kept for historical reasons, a lot of tests use csv with index column
        loaded = pd.read_csv(table_path, index_col=0)
    except:  # noqa: E722
        raise ValueError(f"table path {table_path} could not be loaded")
    if loaded.empty:
        try:
            loaded = pd.read_csv(table_path, sep="\t")
        except:  # noqa: E722
            raise ValueError(f"table path {table_path} could not be loaded")
    return loaded

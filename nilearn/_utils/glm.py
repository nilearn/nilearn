import numpy as np
import pandas as pd

from nilearn._utils import stringify_path


def check_and_load_tables(tables_, var_name):
    """Check tables can be loaded in DataFrame to raise error if necessary."""
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
                "a Path object or a string. "
                f"A {type(table)} was provided at idx {table_idx}"
            )
    return tables


def _read_events_table(table):
    """Accept the path to en event.tsv file \
    and loads it as a Pandas Dataframe.

    Raises an error if loading fails.

    Parameters
    ----------
    table : :obj:`str`, :obj:`pathlib.Path`
        Accepts the path to an events file.

    Returns
    -------
    loaded : pandas.Dataframe object
        Pandas Dataframe with e events data.

    """
    try:
        # kept for historical reasons, a lot of tests use csv with index column
        loaded = pd.read_csv(table, index_col=0)
    except:  # noqa: E722
        raise ValueError(f"table path {table} could not be loaded")
    if loaded.empty:
        try:
            loaded = pd.read_csv(table, sep="\t")
        except:  # noqa: E722
            raise ValueError(f"table path {table} could not be loaded")
    return loaded

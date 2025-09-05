import warnings
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from nilearn._utils.helpers import stringify_path
from nilearn._utils.logger import find_stack_level


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


def create_cosine_drift(high_pass, frame_times):
    """Create a cosine drift matrix with frequencies or equal to high_pass.

    Parameters
    ----------
    high_pass : :obj:`float`
        Cut frequency of the high-pass filter in Hz

    frame_times : array of shape (n_scans,)
        The sampling times in seconds

    Returns
    -------
    cosine_drift : array of shape(n_scans, n_drifts)
        Cosine drifts plus a constant regressor at cosine_drift[:, -1]

    References
    ----------
    http://en.wikipedia.org/wiki/Discrete_cosine_transform DCT-II

    """
    n_frames = len(frame_times)
    n_times = np.arange(n_frames)
    dt = (frame_times[-1] - frame_times[0]) / (n_frames - 1)
    if high_pass * dt >= 0.5:
        warnings.warn(
            "High-pass filter will span all accessible frequencies "
            "and saturate the design matrix. "
            "You may want to reduce the high_pass value."
            f"The provided value is {high_pass} Hz",
            stacklevel=find_stack_level(),
        )
    order = np.minimum(
        n_frames - 1, int(np.floor(2 * n_frames * high_pass * dt))
    )
    cosine_drift = np.zeros((n_frames, order + 1))
    normalizer = np.sqrt(2.0 / n_frames)

    for k in range(1, order + 1):
        cosine_drift[:, k - 1] = normalizer * np.cos(
            (np.pi / n_frames) * (n_times + 0.5) * k
        )

    cosine_drift[:, -1] = 1.0
    return cosine_drift

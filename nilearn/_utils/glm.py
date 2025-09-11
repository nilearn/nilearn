from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from nilearn._utils.helpers import stringify_path


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


def make_stat_maps(
    model, contrasts, output_type="z_score", first_level_contrast=None
):
    """Given a model and contrasts, return the corresponding z-maps.

    Parameters
    ----------
    model : FirstLevelModel or SecondLevelModel object
        Must have a fitted design matrix(ces).

    contrasts : Dict[str, ndarray or str]
        Dict of contrasts for a first or second level model.
        Corresponds to the contrast_def for the FirstLevelModel
        (nilearn.glm.first_level.FirstLevelModel.compute_contrast)
        & second_level_contrast for a SecondLevelModel
        (nilearn.glm.second_level.SecondLevelModel.compute_contrast)

    output_type : :obj:`str`, default='z_score'
        The type of statistical map to retain from the contrast.

        .. versionadded:: 0.9.2

    %(first_level_contrast)s

        .. versionadded:: 0.12.0

    Returns
    -------
    statistical_maps : Dict[str, niimg] or Dict[str, Dict[str, niimg]]
        Dict of statistical z-maps keyed to contrast names/titles.

    See Also
    --------
    nilearn.glm.first_level.FirstLevelModel.compute_contrast
    nilearn.glm.second_level.SecondLevelModel.compute_contrast

    """
    from nilearn.glm.second_level import SecondLevelModel

    if isinstance(model, SecondLevelModel):
        return {
            contrast_name: model.compute_contrast(
                contrast_data,
                output_type=output_type,
                first_level_contrast=first_level_contrast,
            )
            for contrast_name, contrast_data in contrasts.items()
        }

    return {
        contrast_name: model.compute_contrast(
            contrast_data,
            output_type=output_type,
        )
        for contrast_name, contrast_data in contrasts.items()
    }

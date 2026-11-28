"""Utilities to handle dataframes from different libraries.

Nilearn works with :class:`pandas.DataFrame` internally.

These utilities rely on `narwhals <https://narwhals-dev.github.io/narwhals/>`_
to recognize dataframes from other libraries (for example polars)
and convert them to pandas
where they enter the public API.
"""

from typing import Any

import narwhals.stable.v2 as nw
import pandas as pd


def is_dataframe(obj: Any) -> bool:
    """Check if an object is an eager dataframe.

    Parameters
    ----------
    obj : Any
        Object to check.

    Returns
    -------
    :obj:`bool`
        ``True`` if ``obj`` is an eager dataframe
        from any library supported by narwhals
        (for example pandas or polars),
        ``False`` otherwise.
        Lazy frames (for example ``polars.LazyFrame``) are not eager
        and return ``False``.
    """
    return nw.dependencies.is_into_dataframe(obj)


def is_series(obj: Any) -> bool:
    """Check if an object is a series.

    Parameters
    ----------
    obj : Any
        Object to check.

    Returns
    -------
    :obj:`bool`
        ``True`` if ``obj`` is a series
        from any library supported by narwhals
        (for example pandas or polars),
        ``False`` otherwise.
    """
    return nw.dependencies.is_into_series(obj)


def to_pandas(obj: Any) -> Any:
    """Convert a dataframe or a series to pandas.

    pandas objects are returned unchanged, without a copy,
    and so is any object that is not a dataframe or a series
    (for example a numpy array, a string or a path).
    This means this function can be called on any input
    of a parameter that accepts a dataframe among other types.

    The conversion goes through numpy one column at a time,
    so it does not require pyarrow
    (unlike ``polars.DataFrame.to_pandas``).
    The resulting dataframe has a default
    :class:`~pandas.RangeIndex`.

    Missing values are converted as numpy represents them:
    ``NaN`` for numeric columns (integer columns become float columns)
    and ``None`` for other columns.

    Parameters
    ----------
    obj : Any
        Object to convert.

    Returns
    -------
    :class:`pandas.DataFrame`, :class:`pandas.Series` or Any
        ``obj`` converted to pandas if it is a dataframe or a series,
        ``obj`` unchanged otherwise.

    Raises
    ------
    TypeError
        If ``obj`` is a lazy frame (for example ``polars.LazyFrame``).
    """
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj

    native = nw.from_native(obj, pass_through=True, allow_series=True)

    if isinstance(native, nw.LazyFrame):
        raise TypeError(
            "Lazy dataframes are not supported. "
            "Materialize them before passing them to nilearn "
            f"(for example with '.collect()'). Got: {type(obj)}"
        )

    if isinstance(native, nw.DataFrame):
        return pd.DataFrame(
            {
                name: native.get_column(name).to_numpy()
                for name in native.columns
            }
        )

    if isinstance(native, nw.Series):
        return pd.Series(native.to_numpy(), name=native.name)

    return obj

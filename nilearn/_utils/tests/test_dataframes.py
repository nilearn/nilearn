import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nilearn._utils.dataframes import is_dataframe, is_series, to_pandas

NOT_DATAFRAMES = [
    np.zeros((3, 2)),
    "confounds.tsv",
    Path("confounds.tsv"),
    [[1.0, 2.0], [3.0, 4.0]],
    None,
]


@pytest.fixture
def df() -> pd.DataFrame:
    """Return a pandas dataframe with columns of different types."""
    return pd.DataFrame(
        {
            "onset": [0.0, 1.5, 3.0],
            "n": [1, 2, 3],
            "trial_type": ["a", "b", "a"],
            "flag": [True, False, True],
        }
    )


@pytest.mark.ai_generated
def test_is_dataframe(df, to_dataframe_backend):
    """Check dataframes of each library are detected."""
    assert is_dataframe(to_dataframe_backend(df))


@pytest.mark.ai_generated
@pytest.mark.parametrize("obj", [*NOT_DATAFRAMES, pd.Series([1.0, 2.0])])
def test_is_dataframe_false(obj):
    """Check objects that are not dataframes are not detected."""
    assert not is_dataframe(obj)


@pytest.mark.ai_generated
def test_is_dataframe_lazy_frame():
    """Check lazy frames are not considered as dataframes."""
    pl = pytest.importorskip("polars")

    assert not is_dataframe(pl.DataFrame({"a": [1.0]}).lazy())


@pytest.mark.ai_generated
def test_is_series(df, to_dataframe_backend):
    """Check series of each library are detected, but not dataframes."""
    converted = to_dataframe_backend(df)

    assert is_series(converted["onset"])
    assert not is_series(converted)


@pytest.mark.ai_generated
@pytest.mark.parametrize("obj", NOT_DATAFRAMES)
def test_is_series_false(obj):
    """Check objects that are not series are not detected."""
    assert not is_series(obj)


@pytest.mark.ai_generated
def test_to_pandas_returns_pandas_unchanged(df):
    """Check pandas objects are returned as is: no copy, index kept."""
    df.index = [10, 20, 30]

    assert to_pandas(df) is df
    series = df["onset"]
    assert to_pandas(series) is series


@pytest.mark.ai_generated
@pytest.mark.parametrize("obj", NOT_DATAFRAMES)
def test_to_pandas_non_dataframe_unchanged(obj):
    """Check objects that are not dataframes or series are returned as is."""
    assert to_pandas(obj) is obj


@pytest.mark.ai_generated
def test_to_pandas(df, to_dataframe_backend):
    """Check dataframes of each library are converted to the same \
    pandas dataframe.
    """
    result = to_pandas(to_dataframe_backend(df))

    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, df)


@pytest.mark.ai_generated
def test_to_pandas_series(df, to_dataframe_backend):
    """Check series of each library are converted to pandas."""
    result = to_pandas(to_dataframe_backend(df)["onset"])

    assert isinstance(result, pd.Series)
    pd.testing.assert_series_equal(result, df["onset"])


@pytest.mark.ai_generated
def test_to_pandas_missing_values():
    """Check how missing values of a polars dataframe are converted."""
    pl = pytest.importorskip("polars")

    result = to_pandas(
        pl.DataFrame(
            {
                "float": [1.0, None],
                "int": [1, None],
                "str": ["a", None],
            }
        )
    )

    assert np.isnan(result["float"][1])
    # numpy has no missing integer: the column becomes float
    assert result["int"].dtype == np.float64
    assert np.isnan(result["int"][1])
    assert result["str"].isna().tolist() == [False, True]


@pytest.mark.ai_generated
def test_to_pandas_without_pyarrow(df, monkeypatch):
    """Check converting a polars dataframe does not need pyarrow.

    pyarrow is not a dependency of nilearn,
    but polars needs it for ``polars.DataFrame.to_pandas``.
    """
    pl = pytest.importorskip("polars")
    polars_df = pl.from_pandas(df)

    # make any import of pyarrow fail
    monkeypatch.setitem(sys.modules, "pyarrow", None)

    pd.testing.assert_frame_equal(to_pandas(polars_df), df)


@pytest.mark.ai_generated
def test_to_pandas_lazy_frame_error():
    """Check lazy frames are rejected with an informative error."""
    pl = pytest.importorskip("polars")

    with pytest.raises(TypeError, match="Lazy dataframes are not supported"):
        to_pandas(pl.DataFrame({"a": [1.0]}).lazy())

import pandas as pd
import pytest

from nilearn._utils.glm import check_and_load_tables


def test_img_table_checks():
    # check tables type and that can be loaded
    with pytest.raises(
        ValueError, match="Tables to load can only be TSV or CSV."
    ):
        check_and_load_tables([".csv", ".csv"], "")
    with pytest.raises(
        TypeError,
        match="can only be a pandas DataFrame, a Path object or a string",
    ):
        check_and_load_tables([[], pd.DataFrame()], "")
    with pytest.raises(
        ValueError, match="Tables to load can only be TSV or CSV."
    ):
        check_and_load_tables([".csv", pd.DataFrame()], "")

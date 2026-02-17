# ----------------------------------------------------------------------
# pytest-csv - https://github.com/nicoulaj/pytest-csv
# copyright (c) 2018-2021 pytest-csv contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------


def pytest_csv_register_columns(columns):
    """
    Called on plugin initialization.

    Use it to add your own column types (or override the builtin ones).

    For instance, in conftest.py:
    >>> def pytest_csv_register_columns(columns):
    >>>
    >>>     # constant column
    >>>     columns['my_constant_column'] = 'foobar'
    >>>
    >>>     # simple column
    >>>     columns['my_simple_column'] = lambda item, report: {'my column': report.nodeid}
    >>>
    >>>     # a more complex column type that creates several columns in the CSV
    >>>     def my_multiple_columns(item, report):
    >>>         yield 'my column 1', report.nodeid
    >>>         yield 'my column 2', 42
    >>>     columns['my_multiple_columns'] = my_multiple_columns

    Then run pytest with your new column id:

        $ py.test --csv tests.csv --csv-columns id,status,my_constant_column,my_simple_column,my_multiple_columns

    In certain situations, a (failure) line might need to be emitted for cases
    where the original report was lost. For example, when an xdist test runner
    crashes while running the test. In such cases, the column function will be
    called with item=None and an artificially-generated test failure report;
    if they throw an exception, the their column will be empty.

    :param columns: dictionary of (column id, CSVColumn object)
    """


def pytest_csv_written(csv_path):
    """
    Called whenever a CSV file has been written.

    :param csv_path: CSV file path
    """

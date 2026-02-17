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

from collections import OrderedDict

from . import _hooks
from ._reporter import CSVReporter
from .column import *


def pytest_addoption(parser):
    group = parser.getgroup('terminal reporting')
    group.addoption(
        '--csv',
        dest='csv_path',
        action='store',
        metavar='path',
        default=None,
        help='create CSV report file at given path'
    )
    group.addoption(
        '--csv-columns',
        dest='csv_columns',
        action='store',
        type=str,
        nargs='+',
        default=[ID, MODULE, NAME, FILE, DOC, MARKERS, STATUS, MESSAGE, DURATION],
        help='define columns in output CSV'
    )
    group.addoption(
        '--csv-add-columns',
        dest='csv_add_columns',
        action='store',
        type=str,
        nargs='+',
        default=[],
        help='add columns to the default set of columns in output CSV'
    )
    group.addoption(
        '--csv-delimiter',
        dest='csv_delimiter',
        action='store',
        metavar='delimiter character',
        default=',',
        help='delimiter character to use in CSV files'
    )
    group.addoption(
        '--csv-quote-char',
        dest='csv_quote_char',
        action='store',
        metavar='quoting character',
        default='"',
        help='quoting character to use in CSV files'
    )


def pytest_addhooks(pluginmanager):
    pluginmanager.add_hookspecs(_hooks)


def pytest_configure(config):
    csv_path = config.option.csv_path
    if csv_path:
        columns_registry = dict(BUILTIN_COLUMNS_REGISTRY)
        config.hook.pytest_csv_register_columns(columns=columns_registry)

        # TODO improve error handling, if user puts a wrong column name it will raise a KeyError
        columns = OrderedDict((column_id, columns_registry[column_id.strip()])
                              for column_ids in config.option.csv_columns + config.option.csv_add_columns
                              for column_id in column_ids.split(','))

        config._csv_reporter = CSVReporter(csv_path=csv_path,
                                           columns=columns,
                                           delimiter=config.option.csv_delimiter,
                                           quote_char=config.option.csv_quote_char)
        config.pluginmanager.register(config._csv_reporter)


def pytest_unconfigure(config):
    csv_reporter = getattr(config, '_csv_reporter', None)
    if csv_reporter:
        del config._csv_reporter
        config.pluginmanager.unregister(csv_reporter)

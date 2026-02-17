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

import datetime
import os

import six

from ._ids import *
from ._status import *
from ._utils import parse_node_id, get_test_doc, get_test_args, get_test_markers, format_mark_info, \
    format_mark_info_args


def column_id(item, report):
    yield ID, item.nodeid


def column_module(item, report):
    mod, cls, func, params = parse_node_id(item.nodeid)
    yield MODULE, mod or ''


def column_class(item, report):
    mod, cls, func, params = parse_node_id(item.nodeid)
    yield CLASS, cls or ''


def column_function(item, report):
    mod, cls, func, params = parse_node_id(item.nodeid)
    yield FUNCTION, func or ''


def column_name(item, report):
    mod, cls, func, params = parse_node_id(item.nodeid)
    yield NAME, '%s[%s]' % (func, params) if params else func


def column_file(item, report):
    yield FILE, item.location[0]


def column_doc(item, report):
    yield DOC, get_test_doc(item).strip()


def column_status(item, report):
    if report.passed:
        yield STATUS, XPASSED if hasattr(report, 'wasxfail') else PASSED
    elif report.failed:
        yield STATUS, XFAILED if hasattr(report, 'wasxfail') else ERROR if report.when != 'call' else FAILED
    elif report.skipped:
        yield STATUS, XFAILED if hasattr(report, 'wasxfail') else SKIPPED


def column_success(item, report):
    yield SUCCESS, str(not bool(report.failed))


def column_message(item, report):
    if report.passed:
        if hasattr(report, 'wasxfail'):
            yield MESSAGE, report.wasxfail
        else:
            yield MESSAGE, ''

    elif report.failed:
        if hasattr(report, 'wasxfail'):
            yield MESSAGE, report.wasxfail
        else:
            if hasattr(report.longrepr, 'reprcrash') and hasattr(report.longrepr.reprcrash, 'message'):
                yield MESSAGE, report.longrepr.reprcrash.message
            elif isinstance(report.longrepr, six.string_types):
                yield MESSAGE, report.longrepr
            else:
                yield MESSAGE, str(report.longrepr)

    elif report.skipped:
        if hasattr(report, 'wasxfail'):
            yield MESSAGE, report.wasxfail
        else:
            _, _, message = report.longrepr
            yield MESSAGE, message


def column_duration(item, report):
    yield DURATION, str(getattr(report, 'duration', ''))


def column_duration_formatted(item, report):
    yield DURATION_FORMATTED, str(datetime.timedelta(seconds=getattr(report, 'duration', 0)))


def column_markers(item, report):
    yield MARKERS, ','.join(format_mark_info(mark, False)
                            for mark in sorted(get_test_markers(item), key=lambda mark: mark.name))


def column_markers_with_args(item, report):
    yield MARKERS, ','.join(format_mark_info(mark, True)
                            for mark in sorted(get_test_markers(item), key=lambda mark: mark.name))


def column_markers_as_columns(item, report):
    for mark in get_test_markers(item):
        yield mark.name, format_mark_info_args(mark) or str(True)


def column_markers_args_as_columns(item, report):
    for mark in get_test_markers(item):
        if not mark.args and not mark.kwargs:
            yield mark.name, str(True)
        else:
            for i, arg in enumerate(mark.args):
                yield '%s.%i' % (mark.name, i), str(arg)
            for name, value in six.iteritems(mark.kwargs):
                yield '%s.%s' % (mark.name, name), str(value)


def column_parameters(item, report):
    yield PARAMETERS, ','.join('%s=%s' % (k, v) for k, v in sorted(six.iteritems(get_test_args(item))))


def column_parameters_as_columns(item, report):
    for name, value in six.iteritems(get_test_args(item)):
        yield name, str(value)


def column_properties(item, report):
    yield PROPERTIES, ','.join('%s=%s' % (k, v) for k, v in sorted(getattr(report, 'user_properties', [])))


def column_properties_as_columns(item, report):
    for name, value in sorted(getattr(report, 'user_properties', [])):
        yield name, str(value)


def column_working_directory(item, report):
    yield WORKING_DIRECTORY, os.getcwd()

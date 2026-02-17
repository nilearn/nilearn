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

import getpass
import re
from itertools import chain

import os
import six

__NODE_ID__ = re.compile(
    r'(?P<module>.+)\.py(?:::(?P<class>[^:]+)(?:::.+)?)?::(?P<function>[^\[]+)(?:\[(?P<params>.*)\])?')


def parse_node_id(node_id):
    match = re.search(__NODE_ID__, node_id)
    if match:
        return match.group('module').replace('/', '.'), \
               match.group('class') or '', \
               match.group('function'), \
               match.group('params') or ''
    raise Exception('Failed parsing pytest node id: "%s"' % node_id)


def get_user():
    try:
        return getpass.getuser()
    except:
        # workaround for https://bugs.python.org/issue32731
        return os.path.basename(os.path.expanduser("~"))


def get_test_doc(item):
    try:
        return item.obj.__doc__ or ''
    except AttributeError:
        return ''


def get_test_args(item):
    return item.callspec.params if hasattr(item, 'callspec') else {}


def get_test_markers(item):
    if hasattr(item, 'iter_markers'):
        return list(item.iter_markers())

    # pytest <3.8 backward compatibility
    from _pytest.mark import MarkInfo
    return [v for v in six.itervalues(item.keywords) if isinstance(v, MarkInfo)]


def format_mark_info(mark, with_args=True):
    if not with_args or (not mark.args and not mark.kwargs):
        return mark.name
    return '%s(%s)' % (mark.name, format_mark_info_args(mark))


def format_mark_info_args(mark):
    return ','.join(chain(
        (str(arg) for arg in mark.args),
        ('%s=%s' % (k, v) for k, v in sorted(six.iteritems(mark.kwargs)))
    ))

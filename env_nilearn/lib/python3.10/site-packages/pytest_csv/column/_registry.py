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

import platform

from ._builtin import *
from ._ids import *
from ._utils import get_user

BUILTIN_COLUMNS_REGISTRY = {
    ID: column_id,
    MODULE: column_module,
    CLASS: column_class,
    FUNCTION: column_function,
    NAME: column_name,
    FILE: column_file,
    DOC: column_doc,
    STATUS: column_status,
    SUCCESS: column_success,
    DURATION: column_duration,
    DURATION_FORMATTED: column_duration_formatted,
    MESSAGE: column_message,
    MARKERS: column_markers,
    MARKERS_WITH_ARGS: column_markers_with_args,
    MARKERS_AS_COLUMNS: column_markers_as_columns,
    MARKERS_ARGS_AS_COLUMNS: column_markers_args_as_columns,
    PARAMETERS: column_parameters,
    PARAMETERS_AS_COLUMNS: column_parameters_as_columns,
    PROPERTIES: column_properties,
    PROPERTIES_AS_COLUMNS: column_properties_as_columns,
    HOST: platform.node(),
    USER: get_user(),
    SYSTEM: platform.system(),
    SYSTEM_RELEASE: platform.release(),
    SYSTEM_VERSION: platform.version(),
    PYTHON_IMPLEMENTATION: platform.python_implementation(),
    PYTHON_VERSION: platform.python_version(),
    WORKING_DIRECTORY: column_working_directory
}

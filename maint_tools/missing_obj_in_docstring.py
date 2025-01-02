"""Utility to find missing :obj:`` in doc string type.

The script support either a folder or file path as argument
and write the results
in the file `missing_objefault.md`.
"""

import ast

from numpydoc.docscrape import NumpyDocString
from rich import print
from utils import list_functions, list_modules, root_dir

VALUES = [
    "integers",
    "integer",
    "Integer",
    "strings",
    "string",
    "String",
    "boolean",
    "Boolean",
    # "list",
    "List",
    "tuple",
    "Tuple",
    # "dict",
    "Dict",
    # "int", # 43
    "Int",
    "float",
    "Float",
    # "str",
    "Bool",
    # "bool"
]


def get_missing(docstring, values=None):
    """Return missing obj in doc string.

    Returns
    -------
    missing: list[Tuple[str, str, str]]
        Parameters missing :obj:`` from the docstring..
    """
    doc = NumpyDocString(docstring)
    params = {param.name: param.type for param in doc["Parameters"]}

    if values is None:
        values = VALUES

    missing = []
    for v in values:
        for arg_name, arg_desc in params.items():
            if v in arg_desc and f":obj:`{v}`" not in arg_desc:
                missing.append((arg_name, arg_desc, v))

    return missing


if __name__ == "__main__":
    input_path = root_dir()

    filenames = list_modules()

    n_issues = 0

    for filename in filenames:
        create_module_header = True

        relative_filename = filename.relative_to(input_path.parent)

        for func in list_functions(filename):
            docstring = ast.get_docstring(func, clean=False)
            if not docstring:
                print(f"{filename}:{func.lineno} - No docstring detected")
                continue

            missing = get_missing(docstring)

            n_issues += len(missing)

            # Log arguments with missing default values in documentation.
            if missing:
                print(f"{filename}:{func.lineno}")
                for param, desc, value in missing:
                    print(f" '{param}: {desc}' missing :obj:`{value}`")

    print(f"{n_issues} detected")

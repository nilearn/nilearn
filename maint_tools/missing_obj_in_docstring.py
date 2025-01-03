"""Utility to find missing :obj:`` in doc string type.

The script support either a folder or file path as argument
and write the results
in the file `missing_objefault.md`.
"""

import ast

from numpydoc.docscrape import NumpyDocString
from rich import print
from utils import list_classes, list_functions, list_modules

VALUES = [
    "integers",
    "integer",
    "Integer",
    "strings",
    "string",
    "String",
    "boolean",
    "Boolean",
    "list",
    "List",
    "tuple",
    "Tuple",
    "dict",
    "Dict",
    # "int",
    "Int",
    # "float",
    "Float",
    "str",
    "Bool",
    "bool",
]


def main():
    """Find missing :obj:`` in doc string type."""
    filenames = list_modules()

    n_issues = 0

    for filename in filenames:
        for func_def in list_functions(filename):
            check_fill_doc_decorator(func_def, filename)

            docstring = ast.get_docstring(func_def, clean=False)

            if not docstring:
                print(
                    f"{filename}:{func_def.lineno} "
                    f"- [red] {func_def.name} No docstring detected"
                )
                continue

            missing = get_missing(docstring)

            n_issues += len(missing)

            # Log arguments with missing default values in documentation.
            if missing:
                print(f"{filename}:{func_def.lineno} - {func_def.name}")
                for param, desc, value in missing:
                    print(f" '{param}: {desc}' - [red]missing :obj:`{value}`")

        for class_def in list_classes(filename):
            check_fill_doc_decorator(class_def, filename)

            docstring = ast.get_docstring(class_def, clean=False)
            if not docstring:
                print(
                    f"{filename}:{class_def.lineno} "
                    f"- {class_def.name} - [red] No docstring detected"
                )
            else:
                try:
                    missing = get_missing(docstring)
                except Exception:
                    continue

                n_issues += len(missing)

                if missing:
                    print(f"{filename}:{class_def.lineno} - {class_def.name}")
                    for param, desc, value in missing:
                        print(
                            f" '{param}: {desc}' "
                            f"- [red] missing :obj:`{value}`"
                        )

            for meth_def in list_functions(class_def):
                docstring = ast.get_docstring(meth_def, clean=False)
                if not docstring:
                    print(
                        f"{filename}:{meth_def.lineno} "
                        f"- {meth_def.name} - [red] No docstring detected"
                    )
                    continue

                missing = get_missing(docstring)

                n_issues += len(missing)

                if missing:
                    print(f"{filename}:{meth_def.lineno} - {meth_def.name}")
                    for param, desc, value in missing:
                        print(
                            f" '{param}: {desc}' "
                            f"- [red] missing :obj:`{value}`"
                        )

    print(f"{n_issues} detected")


def check_fill_doc_decorator(ast_node, filename):
    """Check that fill_doc decorator is present when needed.

    Checks if '%(' is present in the doc string
    and warns if the function or class
    does not have the @fill_doc dcorator.

    Also warns if the decorator is used for no reason.
    """
    expand_docstring = False
    if ast.get_docstring(ast_node, clean=False):
        expand_docstring = "%(" in ast.get_docstring(ast_node, clean=False)

    if isinstance(ast_node, ast.ClassDef):
        methods_docstrings = [
            ast.get_docstring(meth_def, clean=False)
            for meth_def in list_functions(ast_node)
        ]
        expand_docstring_any_method = any(
            "%(" in x for x in methods_docstrings
        )
        expand_docstring = expand_docstring or expand_docstring_any_method

    if expand_docstring:
        if len(ast_node.decorator_list) == 0:
            print(
                f"{filename}:{ast_node.lineno} "
                "- [red]missing @fill_doc decorator."
            )
    elif any(
        getattr(x, "name", "") == "fill_doc" for x in ast_node.decorator_list
    ):
        print(
            f"{filename}:{ast_node.lineno} "
            "- [red]@fill_doc decorator not needed."
        )


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
    main()

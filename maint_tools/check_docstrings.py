"""Utility to check dostrings.

- checks docstrings of functions, classes and methods
- checks for:
    - find missing :obj:`` in doc string type
    - if a function of class definition uses the fill_doc decorator properly
"""

import ast
import re
from pathlib import Path

from numpydoc.docscrape import NumpyDocString
from rich import print
from utils import list_classes, list_functions, list_modules

# List of values to check for missing :obj:`` link
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
    "int",
    "Int",
    "float",
    "Float",
    "str",
    "Bool",
    "bool",
]


def main() -> None:
    """Find missing :obj:`` in doc string type."""
    print("\n[blue]Finding missing :obj:`` in doc string type.\n")

    filenames = list_modules()

    n_issues = 0

    for filename in filenames:
        for func_def in list_functions(filename):
            check_fill_doc_decorator(func_def, filename)
            n_issues = check_docstring(func_def, filename, n_issues)

        for class_def in list_classes(filename):
            check_fill_doc_decorator(class_def, filename)
            n_issues = check_docstring(class_def, filename, n_issues)

            for meth_def in list_functions(class_def):
                n_issues = check_docstring(meth_def, filename, n_issues)

    print(f"{n_issues} detected\n\n")


def check_fill_doc_decorator(
    ast_node: ast.ClassDef | ast.FunctionDef, filename: str | Path
) -> None:
    """Check that fill_doc decorator is present when needed.

    Checks if '%(' is present in the doc string
    and warns if the function or class
    does not have the @fill_doc decorator.

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
        (
            getattr(x, "name", "") == "fill_doc"
            or getattr(x, "id", "") == "fill_doc"
            or getattr(x, "attr", "") == "fill_doc"
        )
        for x in ast_node.decorator_list
    ):
        print(
            f"{filename}:{ast_node.lineno} "
            "- [red]@fill_doc decorator not needed."
        )


def check_docstring(ast_node, filename: str, n_issues: int) -> int:
    """Check that defaults in an AST node are present in docstring type."""
    docstring = ast.get_docstring(ast_node, clean=False)
    if not docstring:
        print(
            f"{filename}:{ast_node.lineno} "
            f"- {ast_node.name} - [red] No docstring detected"
        )
    else:
        try:
            missing = get_missing(docstring)
        except Exception:
            return n_issues

        n_issues += len(missing)

        if missing:
            print(f"{filename}:{ast_node.lineno} - {ast_node.name}")
            for param, desc, value in missing:
                print(f" '{param}: {desc}' - [red] missing :obj:`{value}`")

    return n_issues


def get_missing(docstring: str, values=None) -> list[tuple[str, str, str]]:
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
            regex = f"{v}" + "[, ]"
            if re.search(regex, arg_desc) and f":obj:`{v}`" not in arg_desc:
                missing.append((arg_name, arg_desc, v))

    return missing


if __name__ == "__main__":
    main()

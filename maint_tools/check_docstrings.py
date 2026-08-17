# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "nilearn[plotting,plotly]>=0.12",
#    "numpydoc",
#    "rich",
# ]
# ///
"""Utility to check dostrings.

- checks docstrings of functions, classes and methods
- checks for:
    - find missing :obj:`` in doc string type

The fill_doc checks live in check_filldoc.py.
"""

import ast
import re
from contextlib import suppress
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

    filenames = list_modules(
        skip_private=False, folders_to_skip=["data", "tests"]
    )

    for filename in filenames:
        for func_def in list_functions(filename, include="all"):
            check_missing_return_annotation(func_def, filename)

            docstring = _get_docstring(func_def, filename)
            if docstring is None:
                continue

            check_docstring(func_def, filename)
            check_returns_yields_and_annotation(func_def, filename)

        for class_def in list_classes(filename, include="all"):
            if _get_docstring(class_def, filename) is not None:
                check_docstring(class_def, filename)

            for meth_def in list_functions(class_def, include="all"):
                if meth_def.name == "__init__":
                    continue

                check_missing_return_annotation(meth_def, filename)

                docstring = _get_docstring(meth_def, filename)
                if docstring is None:
                    continue

                check_docstring(meth_def, filename)
                check_returns_yields_and_annotation(meth_def, filename)


def _get_docstring(ast_node, filename):
    docstring = ast.get_docstring(ast_node, clean=False)
    if not bool(docstring):
        print(
            f"{filename}:{ast_node.lineno} "
            f"- {ast_node.name} - [red] No docstring detected"
        )
        return None
    else:
        return docstring


def check_docstring(ast_node, filename: str | Path) -> None:
    """Check that defaults in an AST node are present in docstring type."""
    docstring = ast.get_docstring(ast_node, clean=False)

    missing = None
    with suppress(Exception):
        missing = get_missing(docstring)

    if missing:
        print(f"{filename}:{ast_node.lineno} - {ast_node.name}")
        for param, desc, value in missing:
            print(f" '{param}: {desc}' - [red] missing :obj:`{value}`")


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


def function_has_return_value(node: ast.AST) -> bool:
    """Return True if function contains a return statement with a value."""
    return any(
        isinstance(subnode, ast.Return) and subnode.value is not None
        for subnode in ast.walk(node)
    )


def function_has_yield(node: ast.AST) -> bool:
    """Return True if function contains a yield statement."""
    return any(
        isinstance(subnode, (ast.Yield, ast.YieldFrom))
        for subnode in ast.walk(node)
    )


def has_none_return_annotation(node: ast.FunctionDef) -> bool:
    """Return True if function has explicit -> None return annotation."""
    if node.returns is None:
        return False

    # Python 3.8+: ast.Constant(value=None)
    if isinstance(node.returns, ast.Constant):
        return node.returns.value is None

    # -> None
    if isinstance(node.returns, ast.Name):
        return node.returns.id == "None"

    return False


def check_missing_return_annotation(ast_node, filename: str | Path) -> None:
    """Warn if a function or method has no return type annotation."""
    if not isinstance(ast_node, ast.FunctionDef):
        return

    if ast_node.returns is None:
        print(
            f"{filename}:{ast_node.lineno} "
            f"- {ast_node.name} - [red]missing return type annotation"
        )


def check_returns_yields_and_annotation(
    ast_node, filename: str | Path
) -> None:
    """Check consistency between return/yield behavior, \
        docstring, and annotations.
    """
    if not isinstance(ast_node, ast.FunctionDef):
        return

    has_return_value = function_has_return_value(ast_node)
    has_yield = function_has_yield(ast_node)

    docstring = ast.get_docstring(ast_node, clean=False)

    np_docstring = NumpyDocString(docstring)

    # if the function returns (or yields) a value
    # then its docstring must have Returns (or Yields) section
    if has_return_value or has_yield:
        if has_yield and bool(np_docstring["Yields"]):
            print(
                f"{filename}:{ast_node.lineno} "
                f"- {ast_node.name} "
                "- [red]missing Yields section in docstring"
            )
        elif has_return_value and not bool(np_docstring["Returns"]):
            print(
                f"{filename}:{ast_node.lineno} "
                f"- {ast_node.name} "
                "- [red]missing Return section in docstring"
            )

    # if the function does not return or yield anything
    # then its return type annotation must be None
    elif not has_none_return_annotation(ast_node):
        print(
            f"{filename}:{ast_node.lineno} "
            f"- {ast_node.name} - [red]missing return annotation '-> None'"
        )


if __name__ == "__main__":
    main()

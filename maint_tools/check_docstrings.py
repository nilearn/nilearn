# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "nilearn[plotting]>=0.12",
#    "numpydoc",
#    "rich",
# ]
# ///
"""Utility to check dostrings.

- checks docstrings of functions, classes and methods
- checks for:
    - find missing :obj:`` in doc string type
    - if a function of class definition uses the fill_doc decorator properly
"""

import ast
import contextlib
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

    filenames = list_modules(
        skip_private=False, folders_to_skip=["data", "input_data", "tests"]
    )

    for filename in filenames:
        for func_def in list_functions(filename, include="all"):
            check_missing_return_annotation(func_def, filename)

            docstring = _get_doctring(func_def, filename)
            if docstring is None:
                continue

            check_fill_doc_decorator(func_def, filename)
            check_docstring(func_def, filename)
            check_returns_yields_and_annotation(func_def, filename)

        for class_def in list_classes(filename, include="all"):
            if _get_doctring(func_def, filename) is not None:
                check_fill_doc_decorator(class_def, filename)
                check_docstring(class_def, filename)

            for meth_def in list_functions(class_def, include="all"):
                if meth_def.name == "__init__":
                    continue

                check_missing_return_annotation(meth_def, filename)

                docstring = _get_doctring(meth_def, filename)
                if docstring is None:
                    continue

                check_fill_doc_decorator(meth_def, filename)
                check_docstring(meth_def, filename)
                check_returns_yields_and_annotation(meth_def, filename)


def _get_doctring(ast_node, filename):
    docstring = ast.get_docstring(ast_node, clean=False)
    if not bool(docstring):
        print(
            f"{filename}:{ast_node.lineno} "
            f"- {ast_node.name} - [red] No docstring detected"
        )
        return None
    else:
        return docstring


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
            "%(" in x for x in methods_docstrings if x is not None
        )
        expand_docstring = expand_docstring or expand_docstring_any_method

    has_fill_doc_decorator = False
    if len(ast_node.decorator_list) == 0:
        has_fill_doc_decorator = False
    elif any(
        (
            getattr(x, "name", "") == "fill_doc"
            or getattr(x, "id", "") == "fill_doc"
            or getattr(x, "attr", "") == "fill_doc"
        )
        for x in ast_node.decorator_list
    ):
        has_fill_doc_decorator = True

    if expand_docstring:
        if not has_fill_doc_decorator:
            print(
                f"{filename}:{ast_node.lineno} "
                "- [red]missing @fill_doc decorator."
            )
    elif has_fill_doc_decorator:
        print(
            f"{filename}:{ast_node.lineno} "
            "- [red]@fill_doc decorator not needed."
        )

    if expand_docstring and not contains_check_params_call(ast_node):
        print(
            f"{filename}:{ast_node.lineno} "
            "- [red]expandable docstring used "
            "but no call to check_params found."
        )


def contains_check_params_call(node: ast.AST) -> bool:
    """Return True if the AST node contains a call to `check_params`."""
    for subnode in ast.walk(node):
        if isinstance(subnode, ast.Call):
            func = subnode.func

            # check_params(...)
            if isinstance(func, ast.Name) and func.id == "check_params":
                return True

            # something.check_params(...)
            if isinstance(func, ast.Attribute) and func.attr == "check_params":
                return True

    return False


def check_docstring(ast_node, filename: str | Path) -> None:
    """Check that defaults in an AST node are present in docstring type."""
    docstring = ast.get_docstring(ast_node, clean=False)
    missing = None
    with contextlib.suppress(Exception):
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
    for subnode in ast.walk(node):
        if isinstance(subnode, ast.Return) and subnode.value is not None:
            return True
    return False


def function_has_yield(node: ast.AST) -> bool:
    """Return True if function contains a yield statement."""
    for subnode in ast.walk(node):
        if isinstance(subnode, (ast.Yield, ast.YieldFrom)):
            return True
    return False


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
    bool(np_docstring["Returns"])

    # function returns / yields a value → must have Returns / Yields section
    if has_return_value or has_yield:
        if not docstring:
            return

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

    # no return & no yield → must be annotated as -> None
    elif not has_none_return_annotation(ast_node):
        print(
            f"{filename}:{ast_node.lineno} "
            f"- {ast_node.name} - [red]missing return annotation '-> None'"
        )


if __name__ == "__main__":
    main()

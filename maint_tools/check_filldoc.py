# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "nilearn[plotting,plotly]>=0.12",
#    "rich",
# ]
# ///
"""Check that the fill_doc decorator is used where it is needed.

Docstrings in nilearn can contain placeholders like ``%(smoothing_fwhm)s``
that ``@fill_doc`` expands at import time. When the decorator is missing the
placeholder survives into the docstring, so ``help()`` and editor tooltips
show the raw ``%(name)s`` to the reader.

This script checks three things and exits with a non zero status if any of
them fails, so that CI catches regressions:

- a docstring contains ``%(`` but the decorator is absent
- the decorator is present but the docstring has nothing to expand
- a docstring uses a parameter from ``TYPE_MAPS`` but ``check_params`` is
  never called

The third one is about a promise rather than about rendering. A parameter
from ``TYPE_MAPS`` is one nilearn validates the type of at run time, and
``check_params`` is what performs that validation. A docstring that documents
such a parameter while the function never calls ``check_params`` promises a
check that nobody makes.

A placeholder wrapped in double backticks is quoted rather than used, so it is
stripped before any of this. ``check_params`` itself is the reason: its
docstring explains what a template looks like by writing ``%(data_dir)s`` in
prose, and there is nothing there for ``@fill_doc`` to expand.

Private modules are included on purpose: their docstrings are what a
developer reads through ``help()``.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

from rich import print
from utils import list_classes, list_functions, list_modules

from nilearn._utils.param_validation import TYPE_MAPS

INLINE_LITERAL = re.compile(r"``[^`\n]*``")

MISSING_DECORATOR = "missing @fill_doc decorator"
UNNEEDED_DECORATOR = "@fill_doc decorator not needed"
MISSING_CHECK_PARAMS = (
    "expandable docstring used but no call to check_params found"
)


def main() -> int:
    """Report every misuse of fill_doc and return an exit status."""
    print("\n[blue]Checking the fill_doc decorator.\n")

    errors: list[tuple[Path, int, str, str]] = []

    filenames = list_modules(
        skip_private=False, folders_to_skip=["data", "tests"]
    )

    for filename in filenames:
        for func_def in list_functions(filename, include="all"):
            if ast.get_docstring(func_def, clean=False):
                errors.extend(check_fill_doc_decorator(func_def, filename))

        for class_def in list_classes(filename, include="all"):
            if ast.get_docstring(class_def, clean=False):
                errors.extend(check_fill_doc_decorator(class_def, filename))

            for meth_def in list_functions(class_def, include="all"):
                if meth_def.name == "__init__":
                    continue
                if ast.get_docstring(meth_def, clean=False):
                    errors.extend(check_fill_doc_decorator(meth_def, filename))

    if not errors:
        print("[green]No fill_doc problem found.")
        return 0

    print(f"[red]Found {len(errors)} problems with fill_doc:\n")
    for filename, lineno, name, problem in errors:
        print(f"{filename}:{lineno} - {name} - [red]{problem}.")

    counts = {
        problem: sum(1 for e in errors if e[3] == problem)
        for problem in (
            MISSING_DECORATOR,
            UNNEEDED_DECORATOR,
            MISSING_CHECK_PARAMS,
        )
    }
    print("")
    for problem, count in counts.items():
        if count:
            print(f"[red]{count} x {problem}")

    return 1


def check_fill_doc_decorator(
    ast_node: ast.ClassDef | ast.FunctionDef, filename: str | Path
) -> list[tuple[Path, int, str, str]]:
    """Return the fill_doc problems of a single function or class.

    Checks whether ``%(`` is present in the docstring and whether the node
    carries the ``@fill_doc`` decorator, in both directions.
    """
    tmp = "|".join(list(TYPE_MAPS.keys()))
    check_params_docstring_regex = rf"\%\([{tmp}]\)s"

    docstring = ast.get_docstring(ast_node, clean=False)

    expand_docstring = False
    check_params_needed = False
    if docstring:
        expand_docstring = bool(re.search(r"\%\(", expandable(docstring)))
        check_params_needed = bool(
            re.search(check_params_docstring_regex, expandable(docstring))
        )

    if isinstance(ast_node, ast.ClassDef):
        methods_docstrings = [
            ast.get_docstring(meth_def, clean=False)
            for meth_def in list_functions(ast_node)
        ]
        expand_docstring = expand_docstring or any(
            re.search(r"\%\(", expandable(x))
            for x in methods_docstrings
            if x is not None
        )
        check_params_needed = check_params_needed or any(
            re.search(check_params_docstring_regex, expandable(x))
            for x in methods_docstrings
            if x is not None
        )

    has_fill_doc_decorator = any(
        (
            getattr(x, "name", "") == "fill_doc"
            or getattr(x, "id", "") == "fill_doc"
            or getattr(x, "attr", "") == "fill_doc"
        )
        for x in ast_node.decorator_list
    )

    errors = []
    if expand_docstring and not has_fill_doc_decorator:
        errors.append(
            (filename, ast_node.lineno, ast_node.name, MISSING_DECORATOR)
        )
    elif has_fill_doc_decorator and not expand_docstring:
        errors.append(
            (filename, ast_node.lineno, ast_node.name, UNNEEDED_DECORATOR)
        )

    if check_params_needed and not contains_check_params_call(ast_node):
        errors.append(
            (filename, ast_node.lineno, ast_node.name, MISSING_CHECK_PARAMS)
        )

    return errors


def expandable(docstring: str) -> str:
    """Return the docstring without the placeholders that are only quoted.

    ``%(data_dir)s`` inside double backticks is a mention of a template, not a
    template, and ``@fill_doc`` leaves it alone.
    """
    return INLINE_LITERAL.sub("", docstring)


def contains_check_params_call(node: ast.AST) -> bool:
    """Return True if the node calls check_params anywhere in its body."""
    for sub_node in ast.walk(node):
        if (
            isinstance(sub_node, ast.Call)
            and getattr(sub_node.func, "id", "") == "check_params"
        ):
            return True
        if isinstance(sub_node, ast.ClassDef):
            for meth_def in list_functions(sub_node):
                if contains_check_params_call(meth_def):
                    return True
    return False


if __name__ == "__main__":
    sys.exit(main())

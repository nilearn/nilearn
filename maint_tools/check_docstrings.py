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
    - if a function of class definition uses the fill_doc decorator properly
"""

import ast
import re
import warnings
from contextlib import suppress
from pathlib import Path

from numpydoc.docscrape import NumpyDocString
from rich import print
from utils import list_classes, list_functions, list_modules

from nilearn._utils.docs import _indentcount_lines, docdict, docdict_indented
from nilearn._utils.param_validation import TYPE_MAPS

# Docstring parameters that do not match the function, class or method
# signature, grouped by the kind of problem found.
UNDOCUMENTED_PARAMS: list[str] = []
EXTRA_PARAMS: list[str] = []
DUPLICATE_PARAMS: list[str] = []
# "name:type" instead of "name : type": numpydoc then treats the whole
# line as the name, an easy typo to fix.
MISSING_SPACE_BEFORE_COLON: list[str] = []

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

            check_fill_doc_decorator(func_def, filename)
            check_docstring(func_def, filename)
            check_returns_yields_and_annotation(func_def, filename)

        for class_def in list_classes(filename, include="all"):
            if _get_docstring(class_def, filename) is not None:
                check_fill_doc_decorator(class_def, filename)
                check_docstring(class_def, filename)

            for meth_def in list_functions(class_def, include="all"):
                if meth_def.name == "__init__":
                    continue

                check_missing_return_annotation(meth_def, filename)

                docstring = _get_docstring(meth_def, filename)
                if docstring is None:
                    continue

                check_fill_doc_decorator(meth_def, filename)
                check_docstring(meth_def, filename)
                check_returns_yields_and_annotation(meth_def, filename)

    print()

    if UNDOCUMENTED_PARAMS:
        message = ["Docstring parameters do not match signature for:"]
        message.append("Undocumented parameter(s):")
        message.extend(f"  - {x}" for x in UNDOCUMENTED_PARAMS)
        warnings.warn("\n".join(message), stacklevel=2)

    if EXTRA_PARAMS or DUPLICATE_PARAMS or MISSING_SPACE_BEFORE_COLON:
        message = ["Docstring parameters do not match signature for:"]
        if EXTRA_PARAMS:
            message.append("Extra documented parameter(s) not in signature:")
            message.extend(f"  - {x}" for x in EXTRA_PARAMS)
        if DUPLICATE_PARAMS:
            message.append("Duplicate documented parameter(s):")
            message.extend(f"  - {x}" for x in DUPLICATE_PARAMS)
        if MISSING_SPACE_BEFORE_COLON:
            message.append(
                "Missing space before colon (e.g. 'foo:' instead of 'foo :'):"
            )
            message.extend(f"  - {x}" for x in MISSING_SPACE_BEFORE_COLON)
        raise ValueError("\n".join(message))


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


def check_fill_doc_decorator(
    ast_node: ast.ClassDef | ast.FunctionDef, filename: str | Path
) -> None:
    """Check that fill_doc decorator is present when needed.

    Checks if '%(' is present in the doc string
    and warns if the function or class
    does not have the @fill_doc decorator.

    Also warns if the decorator is used for no reason.
    """
    tmp = "|".join(list(TYPE_MAPS.keys()))
    check_params_docstring_regex = rf"\%\([{tmp}]\)s"

    expand_docstring = False
    check_params_needed = False
    if ast.get_docstring(ast_node, clean=False):
        expand_docstring = re.search(
            r"\%\(", ast.get_docstring(ast_node, clean=False)
        )
        check_params_needed = re.search(
            check_params_docstring_regex,
            ast.get_docstring(ast_node, clean=False),
        )

    if isinstance(ast_node, ast.ClassDef):
        methods_docstrings = [
            ast.get_docstring(meth_def, clean=False)
            for meth_def in list_functions(ast_node)
        ]
        expand_docstring_any_method = any(
            re.search(r"\%\(", x) for x in methods_docstrings if x is not None
        )
        check_params_needed_any_method = any(
            re.search(check_params_docstring_regex, x)
            for x in methods_docstrings
            if x is not None
        )
        expand_docstring = expand_docstring or expand_docstring_any_method
        check_params_needed = (
            check_params_needed or check_params_needed_any_method
        )

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

    if check_params_needed and not contains_check_params_call(ast_node):
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
    with suppress(Exception):
        missing = get_missing(docstring)

    if missing:
        print(f"{filename}:{ast_node.lineno} - {ast_node.name}")
        for param, desc, value in missing:
            print(f" '{param}: {desc}' - [red] missing :obj:`{value}`")

    check_parameters_docstring(ast_node, filename)


def get_parameters(
    ast_node: ast.ClassDef | ast.FunctionDef,
) -> list[str]:
    """Return the parameter names of a function, method or class.

    For a class, the parameters of its ``__init__`` method are returned,
    as ``__init__`` parameters are expected to be documented
    in the class docstring.
    """
    if isinstance(ast_node, ast.ClassDef):
        init = next(
            (
                node
                for node in ast_node.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "__init__"
            ),
            None,
        )
        if init is None:
            return []
        ast_node = init

    args = ast_node.args
    all_args = args.posonlyargs + args.args + args.kwonlyargs
    if args.vararg is not None:
        all_args.append(args.vararg)
    if args.kwarg is not None:
        all_args.append(args.kwarg)

    return [arg.arg for arg in all_args if arg.arg not in ("self", "cls")]


def expand_doc_templates(docstring: str) -> str:
    """Expand ``%(...)s`` placeholders using nilearn's docdict.

    Mirrors what the ``@fill_doc`` decorator does at runtime, so that
    parameters filled in from shared docdict entries can be checked
    against the actual function or class signature.
    """
    if "%(" not in docstring:
        return docstring

    lines = docstring.splitlines()
    icount = 0 if len(lines) < 2 else _indentcount_lines(lines[1:])
    try:
        indented = docdict_indented[icount]
    except KeyError:
        indent = " " * icount
        docdict_indented[icount] = indented = {}
        for name, dstr in docdict.items():
            dlines = dstr.splitlines()
            try:
                newlines = [dlines[0]] + [indent + ln for ln in dlines[1:]]
                indented[name] = "\n".join(newlines)
            except IndexError:
                indented[name] = dstr

    with suppress(TypeError, ValueError, KeyError):
        docstring = docstring % indented
    return docstring


def check_parameters_docstring(
    ast_node: ast.ClassDef | ast.FunctionDef, filename: str | Path
) -> None:
    """Check docstring parameters against the function or class signature.

    Performs the same checks as ``check_parameters_doctring``
    in ``nilearn/conftest.py``:

    - no undocumented parameters
    - no extra (documented but nonexistent) parameters
    - no duplicate parameter entries
    - each documented parameter has a type

    Functions, classes and methods that fail one of the first three checks
    are recorded respectively in ``UNDOCUMENTED_PARAMS``, ``EXTRA_PARAMS``
    or ``DUPLICATE_PARAMS``.
    """
    docstring = ast.get_docstring(ast_node, clean=False)
    if docstring is None:
        return
    docstring = expand_doc_templates(docstring)

    parameters = get_parameters(ast_node)
    if not parameters:
        return

    doc = NumpyDocString(docstring)
    if not doc["Parameters"]:
        return

    identifier = f"{filename}:{ast_node.lineno} - {ast_node.name}"

    documented = []
    for param in doc["Parameters"]:
        if param.name.startswith("_"):
            continue
        if not param.type:
            if ":" in param.name:
                # e.g. "foo:type" instead of "foo : type": numpydoc then
                # treats the whole line as the name.
                MISSING_SPACE_BEFORE_COLON.append(
                    identifier
                    + f" - missing space before colon: '{param.name}'"
                )
                continue
            print(
                f"{filename}:{ast_node.lineno} - {ast_node.name} "
                f"- [red]missing type for parameter '{param.name}'"
            )
        documented.extend(name.strip() for name in param.name.split(","))

    undocumented = [
        p for p in parameters if p not in documented and "kwarg" not in p
    ]
    extras = [
        p for p in documented if p not in parameters and "kwarg" not in p
    ]
    duplicates = {
        p for p in documented if documented.count(p) > 1 and "kwarg" not in p
    }

    if undocumented:
        UNDOCUMENTED_PARAMS.append(
            identifier + f" - undocumented parameter(s): {undocumented}"
        )
    if extras:
        EXTRA_PARAMS.append(
            identifier
            + f" - extra documented parameter(s) not in signature: {extras}"
        )
    if duplicates:
        DUPLICATE_PARAMS.append(
            identifier + f" - duplicate documented parameter(s): {duplicates}"
        )


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

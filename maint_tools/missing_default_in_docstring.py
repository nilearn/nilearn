"""Utility to find non-documented default value in docstrings."""

import ast
import re

from numpydoc.docscrape import NumpyDocString
from rich import print
from utils import list_classes, list_functions, list_modules


def main():
    """Flag functions or methods with missing defaults."""
    filenames = list_modules()

    n_issues = 0

    for filename in filenames:
        for func_def in list_functions(filename):
            n_issues = check_def(func_def, n_issues, filename)

        for _ in list_classes(filename):
            for meth_def in list_functions(filename):
                n_issues = check_def(meth_def, n_issues, filename)

    print(f"{n_issues} issues detected")


def check_def(ast_def, n_issues, filename):
    """Check AST definitions for missing default values in doc strings."""
    docstring = ast.get_docstring(ast_def, clean=False)

    if not docstring:
        print(f"{filename}:{ast_def.lineno} - No docstring detected")
        return

    default_args = list_parameters_with_defaults(ast_def)

    missing = get_missing(docstring, default_args)

    n_issues += len(missing)

    # Log arguments with missing default values in documentation.
    if missing:
        print(f"{filename}:{ast_def.lineno} - {ast_def.name}")
        for k, d, v in missing:
            print(f" `{k} : {d[0:20]}...`[red] - missing `Default={v}`")

    return n_issues


def list_parameters_with_defaults(ast_def):
    """List parameters in function that have a default value."""
    default_args = {
        k.arg: ast.unparse(v)
        for k, v in zip(ast_def.args.args[::-1], ast_def.args.defaults[::-1])
    }
    # kwargs with default value
    default_args |= {
        k.arg: ast.unparse(v)
        for k, v in zip(
            ast_def.args.kwonlyargs[::-1], ast_def.args.kw_defaults[::-1]
        )
    }
    return default_args


def get_missing(docstring, default_args):
    """Return missing default values documentation.

    Returns
    -------
    missing: list[Tuple[str, str]]
        Parameters missing from the docstring. `(arg name, arg value)`.
    """
    doc = NumpyDocString(docstring)
    params = {param.name: param for param in doc["Parameters"]}

    missing = []
    for argname, argvalue in default_args.items():
        if f"%({argname})s" in params:
            # Skip the generation for templated arguments.
            continue
        if argname not in params:
            missing.append((argname, "", argvalue))
        else:
            desc = "".join(params[argname].desc)
            # Match any of the following patterns:
            # arg : type, default value
            # arg : type, default=value
            # arg : type, default: value
            # arg : type, Default value
            # arg : type, Default=value
            # arg : type, Default: value
            m = re.search(
                r"(default|Default)(\s|:\s|=)(\'|\")?"
                + re.escape(str(argvalue))
                + r"(\'|\")?",
                desc,
            )
            if not m:
                missing.append((argname, desc, argvalue))

    return missing


if __name__ == "__main__":
    main()

"""Utility to find non-documented default value in docstrings.

Also flags if default definition is in the description of the parameter
instead of the type section.
"""

import ast
import importlib
import re

from numpydoc.docscrape import NumpyDocString
from rich import print
from utils import list_classes, list_functions, list_modules

import nilearn

PUBLIC_API_ONLY = True

PUBLIC_API = []
for x in nilearn.__all__:
    if x.startswith("_"):
        continue
    mod = importlib.import_module(f"nilearn.{x}")
    PUBLIC_API.extend(mod.__all__)


def main():
    """Flag functions or methods with missing defaults."""
    filenames = list_modules()

    n_issues = 0

    for filename in filenames:
        for func_def in list_functions(filename):
            if PUBLIC_API_ONLY and func_def.name not in PUBLIC_API:
                continue
            n_issues = check_def(func_def, n_issues, filename)

        for class_def in list_classes(filename):
            if PUBLIC_API_ONLY and class_def.name not in PUBLIC_API:
                continue
            for meth_def in list_functions(class_def):
                n_issues = check_def(meth_def, n_issues, filename)

    print(f"{n_issues} issues detected")


def check_def(ast_def, n_issues, filename):
    """Check AST definitions for missing default values in doc strings."""
    docstring = ast.get_docstring(ast_def, clean=False)

    if not docstring:
        print(f"{filename}:{ast_def.lineno} - No docstring detected")
        return

    default_args = list_parameters_with_defaults(ast_def)

    missing, in_desc = get_missing(docstring, default_args)

    n_issues += len(missing) + len(in_desc)

    if missing:
        print(f"{filename}:{ast_def.lineno} - {ast_def.name}")
        for k, d, v in missing:
            print(f" `{k} : {d[0:120]}...`[red] - missing `Default={v}`")

    if in_desc:
        print(f"{filename}:{ast_def.lineno} - {ast_def.name}")
        for k, d, _ in in_desc:
            print(
                f" `{k} : {d[0:20]}...`[red] - Default found in description."
            )

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
    in_desc = []
    for argname, argvalue in default_args.items():
        if f"%({argname})s" in params:
            # Skip the generation for templated arguments.
            continue

        if argname not in params:
            # missing.append((argname, "", argvalue))
            continue

        if argname == "y":
            continue

        # Match any of the following patterns:
        # arg : type, default.*value
        str_arg = str(argvalue)
        if str_arg != "None":
            continue
        if "%" in argvalue:
            str_arg = str_arg.replace("%", "%%")
        regex = (
            (r"(default|Default).*" + re.escape(str_arg))
            .replace("'", "")
            .replace('"', "")
        )

        type = "".join(params[argname].type)
        if not re.search(regex, type):
            missing.append((argname, type, argvalue))

        # desc = "".join(params[argname].desc)
        # if re.search(regex, desc):
        #     in_desc.append((argname, desc, argvalue))

    return missing, in_desc


if __name__ == "__main__":
    main()

"""Utility to find non-documented default value in docstrings.

Also flags if:

- default definition is in the description of the parameter
instead of the type section.

- "optional" is found in type section of a parameter
"""

import ast
import re

from numpydoc.docscrape import NumpyDocString
from rich import print
from utils import list_classes, list_functions, list_modules, public_api

# Set to true if you want to restrict the checks
# to the user facing part of the API.
PUBLIC_API_ONLY = True


def main():
    """Flag functions or methods with missing defaults."""
    print("\n[blue]Flag functions or methods with missing defaults.\n")

    filenames = list_modules()

    n_issues = 0

    for filename in filenames:
        for func_def in list_functions(filename):
            if PUBLIC_API_ONLY and func_def.name not in public_api:
                continue
            n_issues = check_def(func_def, n_issues, filename)

        for class_def in list_classes(filename):
            if PUBLIC_API_ONLY and class_def.name not in public_api:
                continue
            for meth_def in list_functions(class_def):
                n_issues = check_def(meth_def, n_issues, filename)

    print(f"{n_issues} issues detected\n\n")


def check_def(ast_def, n_issues, filename) -> int:
    """Check AST definitions for missing default values in doc strings."""
    docstring = ast.get_docstring(ast_def, clean=False)

    if not docstring:
        print(f"{filename}:{ast_def.lineno} - No docstring detected")
        return 0

    default_args = list_parameters_with_defaults(ast_def)

    missing, in_desc, optional_found = get_missing(docstring, default_args)

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

    if optional_found:
        print(f"{filename}:{ast_def.lineno} - {ast_def.name}")
        for k, d, _ in optional_found:
            print(
                f" `{k} : {d[-40:]}...`[red] - 'optional' found in type desc."
            )

    return n_issues


def list_parameters_with_defaults(ast_def):
    """List parameters in function that have a default value."""
    default_args = {
        k.arg: ast.unparse(v)
        for k, v in zip(
            ast_def.args.args[::-1], ast_def.args.defaults[::-1], strict=False
        )
    }
    # kwargs with default value
    default_args |= {
        k.arg: ast.unparse(v)
        for k, v in zip(
            ast_def.args.kwonlyargs[::-1],
            ast_def.args.kw_defaults[::-1],
            strict=False,
        )
    }
    return default_args


def get_missing(
    docstring, default_args
) -> tuple[list[str], list[str], list[str]]:
    """Return missing default values documentation.

    Returns
    -------
    missing: list[Tuple[str, str]]
        Parameters missing from the docstring. `(arg name, arg value)`.
    """
    doc = NumpyDocString(docstring)
    params = {param.name: param for param in doc["Parameters"]}

    missing = []
    optional_found = []
    in_desc = []
    for argname, argvalue in default_args.items():
        if f"%({argname})s" in params or f"%({argname}0)s" in params:
            # Skip the generation for templated arguments.
            continue

        if argname not in params:
            missing.append((argname, "", argvalue))
            continue

        # Match any of the following patterns:
        # arg : type, default.*value
        str_arg = str(argvalue)
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

        if "optional" in type:
            optional_found.append((argname, type, argvalue))

        desc = "".join(params[argname].desc)
        if re.search(regex, desc):
            in_desc.append((argname, desc, argvalue))

    return missing, in_desc, optional_found


if __name__ == "__main__":
    main()

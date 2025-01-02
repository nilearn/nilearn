"""Utility to find non-documented default value in docstrings."""

import ast
import re

from numpydoc.docscrape import NumpyDocString
from rich import print
from utils import list_functions, list_modules


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
        elif argname not in params:
            missing.append((argname, argvalue))
        else:
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
                "".join(params[argname].desc),
            )
            if not m:
                missing.append((argname, argvalue))

    return missing


if __name__ == "__main__":
    filenames = list_modules()

    n_issues = 0

    for filename in filenames:
        create_module_header = True

        for func in list_functions(filename):
            docstring = ast.get_docstring(func, clean=False)

            if not docstring:
                print(f"{filename}:{func.lineno} - No docstring detected")
                continue

            # args with default value
            default_args = {
                k.arg: ast.unparse(v)
                for k, v in zip(func.args.args[::-1], func.args.defaults[::-1])
            }
            # kwargs with default value
            default_args |= {
                k.arg: ast.unparse(v)
                for k, v in zip(
                    func.args.kwonlyargs[::-1], func.args.kw_defaults[::-1]
                )
            }

            missing = get_missing(docstring, default_args)

            n_issues += len(missing)

            # Log arguments with missing default values in documentation.
            if missing:
                print(f"{filename}:{func.lineno}")
                for k, v in missing:
                    print(f" `{k}` `Default={v}`")

    print(f"{n_issues} issues detected")

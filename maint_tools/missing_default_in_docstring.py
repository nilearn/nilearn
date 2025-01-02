"""Utility to find non-documented default value in docstrings.

Also flags if default definition is in the description of the parameter
instead of the type section.
"""

import ast
import re

import libcst as cst
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

        update_docstrings(filename)

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
            print(f" `{k} : {d[0:20]}...`[red] - missing `Default={v}`")

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
            missing.append((argname, "", argvalue))
            continue

        # Match any of the following patterns:
        # arg : type, default value
        # arg : type, default=value
        # arg : type, default: value
        # arg : type, Default value
        # arg : type, Default=value
        # arg : type, Default: value
        regex = (
            r"(default|Default)(\s|:\s|=)(\'|\")?"
            + re.escape(str(argvalue))
            + r"(\'|\")?"
        )

        type = "".join(params[argname].type)
        if not re.search(regex, type):
            missing.append((argname, type, argvalue))

        desc = "".join(params[argname].desc)
        if re.search(regex, desc):
            in_desc.append((argname, desc, argvalue))

    return missing, in_desc


class DocstringUpdater(cst.CSTTransformer):
    """Update doc string in CST node."""

    def leave_FunctionDef(self, original_node, updated_node):  # noqa: N802
        """Update node with new doc string."""
        # Get existing docstring
        docstring = None
        if original_node.body.body and isinstance(
            original_node.body.body[0], cst.SimpleStatementLine
        ):
            first_stmt = original_node.body.body[0].body[0]
            if isinstance(first_stmt, cst.Expr) and isinstance(
                first_stmt.value, cst.SimpleString
            ):
                docstring = first_stmt.value.value.strip("\"'")

        # Generate new docstring with defaults from the signature
        new_docstring = self._generate_updated_docstring(
            original_node, docstring
        )
        if new_docstring:
            new_docstring_node = cst.Expr(
                value=cst.SimpleString(f'"""{new_docstring}"""')
            )
            body = updated_node.body.with_changes(
                body=(new_docstring_node,) + updated_node.body.body[1:]
            )
            return updated_node.with_changes(body=body)
        return updated_node

    def _generate_updated_docstring(self, node, docstring):
        print(node)
        return docstring


def update_docstrings(file_path):
    """Update default in doc strings."""
    with file_path.open() as file:
        code = file.read()

    tree = cst.parse_module(code)
    updated_tree = tree.visit(DocstringUpdater())

    with file_path.open("w") as file:
        file.write(updated_tree.code)

    print(f"Updated docstrings in {file_path}")


if __name__ == "__main__":
    main()

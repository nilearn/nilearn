"""Update default values in docstrings."""

from __future__ import annotations

import ast
import itertools
from pathlib import Path

from docstring_parser import parse
from docstring_parser.common import DocstringStyle
from rich import print

from .utils import root_dir

TARGET_STRINGS = [
    "Default=",
    "Default =",
    "Default :",
    "Default:",
    "Default = ",
    "default=",
    "Default is ",
    "Defaults to ",
]
TARGET_STRINGS += [target.lower() for target in TARGET_STRINGS]


def check_docstring(docstring: str, file: Path, lineno: int) -> str:
    """Check docstring content."""
    docstring = parse(docstring, style=DocstringStyle.NUMPYDOC)

    for param in docstring.params:
        if param.description is not None and (
            "default " in param.description or "Default " in param.description
        ):
            print(
                f" default found '{param.arg_name}' "
                f"in {file.resolve()}:{lineno}"
            )

    for target_str, param in itertools.product(
        TARGET_STRINGS, docstring.params
    ):
        update_docstring(param, target_str, file, lineno)


def update_docstring(param, target_str, file, lineno):
    """Update parameters default in docstring."""
    if param.arg_name.startswith("%("):
        return

    if (
        param.description is not None
        and target_str in param.description
        and param.default is None
    ):
        # extract default value from description
        default = param.description.split(target_str)[1].split(".")
        default = ".".join(default[:-1])

        type_name = f"{param.type_name}, default={default}"

        print(
            f"updating '{param.arg_name}' in {file.resolve()}:{lineno}",
            f"with '{default}'",
        )

        with file.open() as f:
            content = f.readlines()

        with file.open("w") as f:
            update_def = False
            update_desc = False
            # skip the line from beginning of file to lineno
            for i, line in enumerate(content):
                if i < lineno:
                    update_def = False
                    update_desc = False
                elif i == lineno:
                    update_def = True
                    update_desc = False

                if update_def and line.startswith(f"    {param.arg_name} :"):
                    f.write(f"    {param.arg_name} : {type_name}\n")
                    print(" updating type name")
                    update_def = False
                    update_desc = True

                elif update_def and line.startswith(
                    f"        {param.arg_name} :"
                ):
                    f.write(f"        {param.arg_name} : {type_name}\n")
                    print(" updating type name")
                    update_def = False
                    update_desc = True

                elif update_desc:
                    if line == f"        {target_str}{default}.\n":
                        f.write("")
                        print(" updating description")
                        update_desc = False
                    elif line.endswith(f" {target_str}{default}.\n"):
                        f.write(line.replace(f" {target_str}{default}.", ""))
                        update_desc = False
                        print(" updating description")
                    else:
                        f.write(line)

                else:
                    f.write(line)


def check_functions(body, file):
    """Check functions of a module or methods of a class."""
    for node in body:
        if isinstance(node, ast.FunctionDef):
            print(f"function: '{node.name}' in {file.resolve()}:{node.lineno}")
            docstring = ast.get_docstring(node)
            docstring = check_docstring(docstring, file, node.lineno)


def main():
    """Update defaults."""
    modules = (root_dir() / "nilearn").glob("**/*.py")

    for file in modules:
        with file.open() as f:
            module = ast.parse(f.read())

        check_functions(module.body, file)

        class_definitions = [
            node for node in module.body if isinstance(node, ast.ClassDef)
        ]

        for class_def in class_definitions:
            print(
                f"class: '{class_def.name}' "
                f"in {file.resolve()}:{class_def.lineno}"
            )

            docstring = ast.get_docstring(class_def)
            check_docstring(docstring, file, class_def.lineno)

            check_functions(class_def.body, file)


if __name__ == "__main__":
    main()

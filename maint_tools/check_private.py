"""Check that private functions are not used in public modules."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
from rich import print


def root_dir() -> Path:
    """Return path to root directory."""
    return Path(__file__).parent.parent


def list_private_functions(body) -> list[str]:
    """Check functions of a module or methods of a class."""
    function_definitions = [
        node for node in body if isinstance(node, ast.FunctionDef)
    ]
    return [f.name for f in function_definitions if f.name.startswith("_")]


def main():
    """List private function that are only mnetioned once in their module."""
    print("\nCheck .py files in nilearn\n")

    modules = (root_dir() / "nilearn").glob("**/*.py")

    files_to_skip = ["test_", "conftest.py"]

    private_functions = {
        "module": [],
        "name": [],
        "count_in": [],
        "count_out": [],
    }

    for file in modules:
        if any(file.name.startswith(s) for s in files_to_skip):
            continue

        with file.open() as f:
            module = ast.parse(f.read())

        functions = list_private_functions(module.body)
        if private_functions:
            for f in functions:
                private_functions["module"].append(file)
                private_functions["name"].append(f)
                private_functions["count_in"].append(0)
                private_functions["count_out"].append(0)

    # check if functions have the same name
    tmp = pd.Series(private_functions["name"]).value_counts()
    if not all(tmp == 1):
        print("Some private functions have the same name.")
        print(tmp[tmp > 1])

    # count how many times each private function is mentioned

    # in its own module
    for i, func in enumerate(private_functions["name"]):
        with private_functions["module"][i].open() as file:
            lines = file.readlines()
        private_functions["count_in"][i] = sum(
            f"{func}" in line for line in lines
        )
        assert private_functions["count_in"][i] >= 1

    # out of its own module
    modules = (root_dir() / "nilearn").glob("**/*.py")
    for file in modules:
        if any(file.name.startswith(s) for s in files_to_skip):
            continue
        with file.open() as f:
            lines = f.readlines()
        for line in lines:
            for i, func in enumerate(private_functions["name"]):
                if str(private_functions["module"][i]) == str(file):
                    continue
                if f" {func}(" in line:
                    private_functions["count_out"][i] += 1

    # report
    for i, func in enumerate(private_functions["name"]):
        if private_functions["count_in"][i] < 2:
            print(
                f"'{func}' mentioned only {private_functions['count_in'][i]} "
                f"time in {private_functions['module'][i]}."
            )
    print()
    for i, func in enumerate(private_functions["name"]):
        if private_functions["count_out"][i] > 0:
            print(
                f"'{func}' mentioned "
                f"{private_functions['count_out'][i]} times "
                f"outside of {private_functions['module'][i]}."
            )


if __name__ == "__main__":
    main()

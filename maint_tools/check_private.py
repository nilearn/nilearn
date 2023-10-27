"""Check that private functions are not used in public modules."""

from __future__ import annotations

import ast
from pathlib import Path

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
    }

    for file in modules:
        if any(file.name.startswith(s) for s in files_to_skip):
            continue

        with open(file) as f:
            module = ast.parse(f.read())

        functions = list_private_functions(module.body)
        if private_functions:
            for f in functions:
                private_functions["module"].append(file)
                private_functions["name"].append(f)
                private_functions["count_in"].append(0)

    for i, func in enumerate(private_functions["name"]):
        # check that each private function is mentioned twice in its own module
        with open(private_functions["module"][i]) as file:
            lines = file.readlines()

        private_functions["count_in"][i] = sum(func in line for line in lines)
        if private_functions["count_in"][i] < 2:
            print(
                f"'{func}' mentioned only {private_functions['count_in'][i]} "
                f"time in {private_functions['module'][i]}"
            )


if __name__ == "__main__":
    main()

"""Utilities for maintenance."""

import ast
from pathlib import Path
from typing import Literal

FOLDERS_TO_SKIP = ["externals", "data", "input_data", "tests", "_utils"]

FILES_TO_SKIP = ["test_", "conftest"]


def root_dir() -> Path:
    """Return path to root directory."""
    return Path(__file__).parent.parent


def list_modules(
    folders_to_skip=None,
    files_to_skip: list[str] | None = None,
    skip_private: bool = True,
) -> list[Path]:
    """List modules in nilearn folder."""
    if folders_to_skip is None:
        folders_to_skip = FOLDERS_TO_SKIP

    if files_to_skip is None:
        files_to_skip = FILES_TO_SKIP

    if skip_private:
        files_to_skip += "_"

    modules = []
    for mod in (root_dir() / "nilearn").glob("**/*.py"):
        if any(x.stem in folders_to_skip for x in mod.parents):
            continue
        if any(mod.name.startswith(s) for s in files_to_skip):
            continue
        modules.append(mod)

    return modules


def list_functions(
    file: Path | ast.Module | ast.ClassDef,
    include: Literal["private", "public", "all"] = "public",
) -> list[ast.FunctionDef]:
    """Return AST of the functions in a module."""
    if isinstance(file, Path):
        with file.open() as f:
            module = ast.parse(f.read())
    else:
        module = file
    function_definitions = [
        node for node in module.body if isinstance(node, ast.FunctionDef)
    ]
    if include == "all":
        return function_definitions
    elif include == "private":
        return [f for f in function_definitions if f.name.startswith("_")]
    return [f for f in function_definitions if not f.name.startswith("_")]


def list_classes(
    file: Path, include: Literal["private", "public", "all"] = "public"
) -> list[ast.ClassDef]:
    """Return AST of the Classes in a module."""
    with file.open() as f:
        module = ast.parse(f.read())
    class_definitions = [
        node for node in module.body if isinstance(node, ast.ClassDef)
    ]
    if include == "all":
        return class_definitions
    elif include == "private":
        return [c for c in class_definitions if c.name.startswith("_")]
    return [c for c in class_definitions if not c.name.startswith("_")]

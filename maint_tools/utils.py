"""Utilities for maintenance."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from typing import Literal

import nilearn

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
    """Return AST of the functions in a module.

    Can also be used to return the AST of methods in a class.
    """
    return list_nodes(file, ast.FunctionDef, include)


def list_classes(
    file: Path, include: Literal["private", "public", "all"] = "public"
) -> list[ast.ClassDef]:
    """Return AST of the Classes in a module."""
    return list_nodes(file, ast.ClassDef, include)


def list_nodes(
    file: Path | ast.Module | ast.ClassDef,
    node_type,
    include: Literal["private", "public", "all"] = "public",
) -> list[ast.ClassDef] | list[ast.FunctionDef]:
    """Return AST of the nodes in a module."""
    if isinstance(file, Path):
        with file.open() as f:
            module = ast.parse(f.read())
    else:
        module = file
    node_definitions = [
        node for node in module.body if isinstance(node, node_type)
    ]

    if include == "all":
        return node_definitions
    elif include == "private":
        return [c for c in node_definitions if c.name.startswith("_")]
    return [c for c in node_definitions if not c.name.startswith("_")]


def update_api(api, mod):
    """Add function and class names of a module to user facing API listing."""
    for x in mod.__all__:
        if x.startswith("_"):
            continue
        if inspect.isfunction(mod.__dict__[x]) or inspect.isclass(
            mod.__dict__[x]
        ):
            api.append(x)
    return api


public_api = []
for subpackage in nilearn.__all__:
    if subpackage.startswith("_"):
        continue
    mod = importlib.import_module(f"nilearn.{subpackage}")
    public_api = update_api(public_api, mod)
    for x in mod.__all__:
        if x.startswith("_"):
            continue
        if inspect.ismodule(mod.__dict__[x]):
            submod = importlib.import_module(f"nilearn.{subpackage}.{x}")
            public_api = update_api(public_api, submod)

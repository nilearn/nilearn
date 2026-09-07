"""Flag test calls passing keyword arguments equal to their default value.

Scans the nilearn test suite for calls to nilearn functions, methods, or
class constructors where a keyword argument is explicitly set to the same
value as its default.

Example: given

.. code-block:: python

    def high_variance_confounds(
        imgs, n_confounds=5, percentile=2.0, detrend=True, mask_img=None
    ): ...

a test calling ``high_variance_confounds(imgs, mask_img=None)`` gets
flagged, because ``mask_img=None`` is already the default and passing it
explicitly does not exercise anything beyond the implicit default
behavior.

This is a heuristic: method calls (``obj.method(...)``) are matched by
method name only, since the type of ``obj`` is not resolved. A keyword
argument is flagged if it matches the default of *any* nilearn callable
sharing that name, so every flag should be checked by hand before the
"extra-default" keyword argument is removed from the test.

Pass ``--fix`` to automatically remove the flagged keyword arguments.
Only "safe" matches are removed automatically: cases where every nilearn
callable sharing that name agrees on the default value for that
parameter. Ambiguous matches (a method name shared by classes that
disagree on the default) are only ever reported, never auto-removed.
"""

import argparse
import ast
import importlib
import inspect
import pkgutil
import re
from collections import defaultdict
from pathlib import Path

from rich import print
from utils import root_dir

import nilearn

MODULE_PARTS_TO_SKIP = {"tests", "data", "_assets"}


def build_registry() -> dict[str, dict[str, dict[str, object]]]:
    """Map a callable name to the defaults of every nilearn callable.

    Returns
    -------
    registry : dict[str, dict[str, dict[str, object]]]
        ``{name: {owner: {param_name: default_value}}}`` where ``name`` is
        the function, method or class name as it would appear at a call
        site, and ``owner`` is the dotted qualified name of the object the
        defaults were extracted from.
    """
    registry: dict[str, dict[str, dict[str, object]]] = {}

    def _register(name, owner, defaults):
        if not defaults:
            return
        registry.setdefault(name, {})[owner] = defaults

    def _defaults(sig: inspect.Signature) -> dict[str, object]:
        skip_kinds = (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
        return {
            name: param.default
            for name, param in sig.parameters.items()
            if param.default is not inspect.Parameter.empty
            and param.kind not in skip_kinds
        }

    for module_info in pkgutil.walk_packages(
        nilearn.__path__, prefix="nilearn."
    ):
        name_parts = module_info.name.split(".")
        if any(part in MODULE_PARTS_TO_SKIP for part in name_parts):
            continue
        if any(part.startswith("test_") for part in name_parts):
            continue
        try:
            module = importlib.import_module(module_info.name)
        except Exception:
            continue

        for member_name, member in inspect.getmembers(module):
            member_module = getattr(member, "__module__", None)
            if not member_module or not member_module.startswith("nilearn"):
                continue

            if inspect.isfunction(member):
                try:
                    sig = inspect.signature(member)
                except (TypeError, ValueError):
                    continue
                owner = f"{member_module}.{member_name}"
                _register(member_name, owner, _defaults(sig))

            elif inspect.isclass(member):
                try:
                    sig = inspect.signature(member.__init__)
                except (TypeError, ValueError):
                    sig = None
                if sig is not None:
                    owner = f"{member_module}.{member_name}"
                    _register(member_name, owner, _defaults(sig))

                for meth_name, meth in inspect.getmembers(
                    member, predicate=inspect.isfunction
                ):
                    if meth_name.startswith("__"):
                        continue
                    meth_module = getattr(meth, "__module__", "")
                    if not meth_module.startswith("nilearn"):
                        continue
                    try:
                        meth_sig = inspect.signature(meth)
                    except (TypeError, ValueError):
                        continue
                    owner = f"{member_module}.{member_name}.{meth_name}"
                    _register(meth_name, owner, _defaults(meth_sig))

    return registry


def list_test_files(path: Path | str | None = None) -> list:
    """List test files to scan, defaulting to the whole nilearn package."""
    base = Path(path) if path else (root_dir() / "nilearn")
    return sorted(base.glob("**/tests/test_*.py"))


class Flag:
    """A keyword argument that matches its callable's default value."""

    def __init__(
        self,
        call: ast.Call,
        keyword: ast.keyword,
        message: str,
        safe_to_fix: bool,
    ):
        self.call = call
        self.keyword = keyword
        self.message = message
        self.safe_to_fix = safe_to_fix


def _match_owners(
    arg: str, value: object, candidates: dict[str, dict[str, object]]
) -> tuple[list[str], bool]:
    """Return owners whose default for ``arg`` equals ``value``.

    Also returns whether the match is unambiguous, i.e. no candidate
    owner defines a *different* default for that same parameter name.
    """
    matching_owners = []
    conflicting = False
    for owner, defaults in candidates.items():
        if arg not in defaults:
            continue
        try:
            is_default = bool(value == defaults[arg])
        except Exception:
            is_default = False
        if is_default:
            matching_owners.append(owner)
        else:
            conflicting = True
    return matching_owners, not conflicting


def check_call(
    call: ast.Call, registry: dict[str, dict[str, dict[str, object]]]
) -> list[Flag]:
    """Return flags for keyword args in ``call`` matching a default."""
    if isinstance(call.func, ast.Name):
        name = call.func.id
    elif isinstance(call.func, ast.Attribute):
        name = call.func.attr
    else:
        return []

    candidates = registry.get(name)
    if not candidates:
        return []

    flags = []
    for keyword in call.keywords:
        if keyword.arg is None:
            # **kwargs unpacking: nothing to compare.
            continue

        try:
            value = ast.literal_eval(keyword.value)
        except (ValueError, TypeError, SyntaxError):
            # Not a literal (e.g. a variable or another call): skip,
            # as we cannot safely tell if it matches the default.
            continue

        matching_owners, safe_to_fix = _match_owners(
            keyword.arg, value, candidates
        )
        if not matching_owners:
            continue

        owners = ", ".join(matching_owners)
        note = "" if safe_to_fix else " [ambiguous: not auto-fixed]"
        message = (
            f"`{name}(..., {keyword.arg}="
            f"{ast.unparse(keyword.value)})` "
            f"matches default in: {owners}{note}"
        )
        flags.append(Flag(call, keyword, message, safe_to_fix))

    return flags


def _line_starts(source: str) -> list[int]:
    """Return the character offset of the start of each line."""
    starts = [0]
    for line in source.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    return starts


def _offset(
    line_starts: list[int], lineno: int | None, col: int | None
) -> int:
    """Convert a 1-indexed (lineno, col) AST position to a char offset."""
    assert lineno is not None
    assert col is not None
    return line_starts[lineno - 1] + col


def _merge_runs(indices: list[int]) -> list[tuple[int, int]]:
    """Merge sorted indices into contiguous (first, last) index runs."""
    runs = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = prev = idx
    runs.append((start, prev))
    return runs


def compute_removal_edits(
    call: ast.Call,
    keywords_to_remove: set,
    source: str,
    line_starts: list[int],
) -> list[tuple[int, int]]:
    """Compute (start, end) character offsets to delete from ``source``.

    Removes each flagged keyword argument along with its separating
    comma, preserving the formatting of the arguments that remain.
    """
    combined = list(call.args) + list(call.keywords)
    removed_indices = [
        idx
        for idx, item in enumerate(combined)
        if id(item) in keywords_to_remove
    ]
    if not removed_indices:
        return []

    n = len(combined)
    edits = []
    for i, j in _merge_runs(removed_indices):
        if j == n - 1 and i > 0:
            start = _offset(
                line_starts,
                combined[i - 1].end_lineno,
                combined[i - 1].end_col_offset,
            )
        else:
            start = _offset(
                line_starts, combined[i].lineno, combined[i].col_offset
            )

        if j == n - 1:
            end = _offset(
                line_starts,
                combined[j].end_lineno,
                combined[j].end_col_offset,
            )
            if i == 0:
                # The whole argument list was removed: also swallow a
                # dangling trailing comma left before the closing paren.
                match = re.match(r"\s*,", source[end:])
                if match:
                    end += match.end()
        else:
            end = _offset(
                line_starts,
                combined[j + 1].lineno,
                combined[j + 1].col_offset,
            )
        edits.append((start, end))
    return edits


def _iter_test_functions(tree: ast.Module):
    """Yield ``test_*`` function definitions anywhere in ``tree``."""
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and node.name.startswith("test_"):
            yield node


def _has_ai_generated_marker(func) -> bool:
    """Return True if ``func`` already carries the ai_generated marker."""
    for decorator in func.decorator_list:
        try:
            if ast.unparse(decorator) == "pytest.mark.ai_generated":
                return True
        except Exception:
            continue
    return False


def marker_edits(
    tree: ast.Module, fixed_linenos: set, line_starts: list[int]
) -> list[tuple[int, int, str]]:
    """Insert ``@pytest.mark.ai_generated`` above every modified test.

    Per this project's conventions, any test function that got a
    keyword argument removed by ``--fix`` must carry this marker.
    """
    edits = []
    for func in _iter_test_functions(tree):
        end = func.end_lineno or func.lineno
        if not any(func.lineno <= ln <= end for ln in fixed_linenos):
            continue
        if _has_ai_generated_marker(func):
            continue
        target_lineno = (
            func.decorator_list[0].lineno
            if func.decorator_list
            else func.lineno
        )
        indent = " " * func.col_offset
        offset = line_starts[target_lineno - 1]
        edits.append((offset, offset, f"{indent}@pytest.mark.ai_generated\n"))
    return edits


def import_pytest_edit(
    tree: ast.Module, source: str, line_starts: list[int]
) -> tuple[int, int, str] | None:
    """Insert ``import pytest`` if the file does not already have it."""
    if re.search(r"(?m)^import pytest$", source):
        return None
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            offset = line_starts[node.lineno - 1]
            return (offset, offset, "import pytest\n")
    return (0, 0, "import pytest\n")


def fix_file(source: str, tree: ast.Module, flags: list[Flag]) -> str:
    """Return ``source`` with every safe-to-fix flag's keyword removed."""
    flags_by_call = defaultdict(list)
    for flag in flags:
        if flag.safe_to_fix:
            flags_by_call[id(flag.call)].append(flag)

    if not flags_by_call:
        return source

    line_starts = _line_starts(source)
    edits: list[tuple[int, int, str]] = []
    fixed_linenos = set()
    for call_flags in flags_by_call.values():
        call = call_flags[0].call
        fixed_linenos.add(call.lineno)
        keywords_to_remove = {id(flag.keyword) for flag in call_flags}
        for start, end in compute_removal_edits(
            call, keywords_to_remove, source, line_starts
        ):
            edits.append((start, end, ""))

    mark_edits = marker_edits(tree, fixed_linenos, line_starts)
    edits.extend(mark_edits)

    if mark_edits:
        pytest_edit = import_pytest_edit(tree, source, line_starts)
        if pytest_edit is not None:
            edits.append(pytest_edit)

    for start, end, text in sorted(edits, key=lambda e: e[0], reverse=True):
        source = source[:start] + text + source[end:]
    return source


def main():
    """Flag test calls passing keyword arguments equal to their default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Restrict the scan to test files under this path.",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help=(
            "Remove flagged keyword arguments, but only when every "
            "nilearn callable sharing that name agrees on the default "
            "(ambiguous matches are reported but left untouched)."
        ),
    )
    args = parser.parse_args()

    print("\n[blue]Indexing nilearn callables and their defaults...\n")
    registry = build_registry()

    print(
        "[blue]Scanning tests for keyword arguments set to "
        "their default value...\n"
    )

    n_issues = 0
    n_fixed = 0
    for test_file in list_test_files(args.path):
        try:
            relative = test_file.relative_to(root_dir())
        except ValueError:
            relative = test_file

        source = test_file.read_text()
        tree = ast.parse(source)

        flags = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            flags.extend(check_call(node, registry))

        for flag in flags:
            print(f"{relative}:{flag.call.lineno} - {flag.message}")
            n_issues += 1

        if not args.fix or not flags:
            continue

        fixed_source = fix_file(source, tree, flags)
        if fixed_source != source:
            n_fixed += sum(1 for flag in flags if flag.safe_to_fix)
            test_file.write_text(fixed_source)

    print(f"\n{n_issues} redundant default keyword arguments detected")
    if args.fix:
        print(f"{n_fixed} removed automatically\n")


if __name__ == "__main__":
    main()

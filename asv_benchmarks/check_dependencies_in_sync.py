"""Check that asv_benchmarks/asv.conf.json stays in sync with pyproject.toml.

``asv.conf.json``'s ``pythons`` and ``matrix`` pins are meant to track
nilearn's actual minimum supported versions, i.e. the ``min_plotting``
dependency group in ``pyproject.toml`` (which itself includes ``min``)
and ``project.requires-python``.

A stale pin here can silently drift below what nilearn now declares as
its floor, or below what has pre-built wheels for the configured
``pythons``, forcing asv to build a dependency from source -- which
tends to fail on the CI runners (see gh-6592 / gh-6593).
"""

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

ASV_CONF = Path(__file__).parent / "asv.conf.json"
PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"


def _strip_json_comments(text: str) -> str:
    """Strip asv.conf.json's ``//`` line comments so it becomes valid JSON.

    Tracks whether we are inside a string literal so a ``//`` that is
    part of a value (for example in the ``repo`` URL) is left alone.
    """
    out = []
    in_string = False
    escape = False
    i, n = 0, len(text)
    while i < n:
        char = text[i]
        if in_string:
            out.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            i += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            i += 1
            continue
        if char == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        out.append(char)
        i += 1
    return "".join(out)


def load_asv_conf(path: Path = ASV_CONF) -> dict:
    """Load asv_benchmarks/asv.conf.json, tolerating its ``//`` comments."""
    return json.loads(_strip_json_comments(path.read_text()))


def _resolve_group(
    groups: dict, name: str, _seen: set | None = None
) -> list[str]:
    """Flatten a PEP 735 ``dependency-groups`` entry into requirement strings.

    Recurses into ``{"include-group": ...}`` entries such as the one
    ``min_plotting`` uses to pull in ``min``.
    """
    _seen = _seen or set()
    if name in _seen:
        raise ValueError(f"Circular dependency-group include: {name!r}")
    _seen.add(name)

    resolved = []
    for entry in groups[name]:
        if isinstance(entry, dict):
            resolved.extend(
                _resolve_group(groups, entry["include-group"], _seen)
            )
        else:
            resolved.append(entry)
    return resolved


def min_versions_from_pyproject(
    path: Path = PYPROJECT, group: str = "min_plotting"
) -> dict[str, str]:
    """Return ``{canonical package name: pinned "==" version}`` for a group.

    Requirements with no ``==`` clause (for example a bare ``tox>=4``)
    are skipped: there is nothing to compare an asv matrix pin against.
    """
    with path.open("rb") as f:
        pyproject = tomllib.load(f)

    versions = {}
    for entry in _resolve_group(pyproject["dependency-groups"], group):
        requirement = Requirement(entry)
        pinned = [s for s in requirement.specifier if s.operator == "=="]
        if not pinned:
            continue
        versions[canonicalize_name(requirement.name)] = pinned[0].version
    return versions


def min_python_from_pyproject(path: Path = PYPROJECT) -> str:
    """Return the ``"major.minor"`` floor of ``project.requires-python``."""
    with path.open("rb") as f:
        pyproject = tomllib.load(f)

    requires_python = pyproject["project"]["requires-python"]
    floors = [
        s.version
        for s in SpecifierSet(requires_python)
        if s.operator in (">=", "==")
    ]
    if not floors:
        raise ValueError(
            f"Could not find a lower bound in requires-python = "
            f"{requires_python!r}"
        )
    lowest = min(floors, key=lambda v: tuple(map(int, v.split("."))))
    return ".".join(lowest.split(".")[:2])


def check_in_sync(
    asv_conf: dict,
    pyproject_versions: dict[str, str],
    pyproject_min_python: str,
) -> list[str]:
    """Return mismatch descriptions between ``asv_conf`` and pyproject.toml.

    An empty list means everything is in sync.
    """
    problems = []

    asv_pythons = asv_conf["pythons"]
    if asv_pythons != [pyproject_min_python]:
        problems.append(
            f"asv.conf.json 'pythons' is {asv_pythons!r} but "
            "pyproject.toml's 'requires-python' floor is "
            f"{pyproject_min_python!r}."
        )

    for package, pinned in asv_conf["matrix"].items():
        if not pinned:
            # an empty/null pin intentionally means "latest": nothing
            # to compare against a minimum version.
            continue

        canonical = canonicalize_name(package)
        if canonical not in pyproject_versions:
            problems.append(
                f"asv.conf.json pins '{package}' but pyproject.toml's "
                "'min_plotting' group has no '==' pinned version for it."
            )
            continue

        expected = pyproject_versions[canonical]
        if pinned != [expected]:
            problems.append(
                f"asv.conf.json pins '{package}' to {pinned!r} but "
                "pyproject.toml's 'min_plotting' group pins it to "
                f"'{expected}'."
            )

    return problems


def _fix_pythons_line(text: str, expected: str) -> str:
    """Rewrite asv.conf.json's ``"pythons"`` line to pin ``expected``."""
    pattern = re.compile(r'("pythons":\s*)\[[^\]]*\]')
    return pattern.sub(rf'\1["{expected}"]', text, count=1)


def _fix_matrix_versions(text: str, fixes: dict[str, str]) -> str:
    """Rewrite matching ``"package": [...]`` entries inside ``"matrix"``.

    Only the ``"matrix": { ... }`` block is touched, so a package name
    that happens to also appear elsewhere in the file (a comment, the
    ``repo`` URL) is left alone.
    """
    if not fixes:
        return text

    matrix = re.search(r'("matrix":\s*\{)(.*?)(\n\s*\})', text, re.DOTALL)
    if matrix is None:
        raise ValueError("Could not locate a 'matrix' object to fix.")

    block = matrix.group(2)
    for package, version in fixes.items():
        block = re.sub(
            rf'("{re.escape(package)}":\s*)\[[^\]]*\]',
            rf'\1["{version}"]',
            block,
        )
    return text[: matrix.start(2)] + block + text[matrix.end(2) :]


def apply_fixes(
    text: str,
    asv_conf: dict,
    pyproject_versions: dict[str, str],
    pyproject_min_python: str,
) -> str:
    """Return asv.conf.json's ``text`` with every resolvable pin fixed.

    A ``matrix`` package with no ``"=="`` pinned version in
    pyproject.toml cannot be resolved automatically and is left as-is.
    """
    if asv_conf["pythons"] != [pyproject_min_python]:
        text = _fix_pythons_line(text, pyproject_min_python)

    fixes = {
        package: pyproject_versions[canonicalize_name(package)]
        for package, pinned in asv_conf["matrix"].items()
        if pinned
        and canonicalize_name(package) in pyproject_versions
        and pinned != [pyproject_versions[canonicalize_name(package)]]
    }
    return _fix_matrix_versions(text, fixes)


def main(argv: list[str] | None = None) -> int:
    """Print any mismatches and return the process exit code.

    With ``--fix``, rewrite asv_benchmarks/asv.conf.json in place to
    resolve whatever mismatch it can before reporting.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fix",
        action="store_true",
        help=(
            "Rewrite asv_benchmarks/asv.conf.json in place to resolve "
            "any mismatch this script knows how to fix."
        ),
    )
    args = parser.parse_args(argv)

    asv_conf = load_asv_conf(ASV_CONF)
    pyproject_versions = min_versions_from_pyproject(PYPROJECT)
    pyproject_min_python = min_python_from_pyproject(PYPROJECT)

    problems = check_in_sync(
        asv_conf, pyproject_versions, pyproject_min_python
    )

    if args.fix and problems:
        new_text = apply_fixes(
            ASV_CONF.read_text(),
            asv_conf,
            pyproject_versions,
            pyproject_min_python,
        )
        ASV_CONF.write_text(new_text)
        print("Applied fixes to asv_benchmarks/asv.conf.json.\n")

        asv_conf = load_asv_conf(ASV_CONF)
        problems = check_in_sync(
            asv_conf, pyproject_versions, pyproject_min_python
        )

    if problems:
        print(
            "asv_benchmarks/asv.conf.json is out of sync with "
            "pyproject.toml:\n"
        )
        for problem in problems:
            print(f"- {problem}")
        if not args.fix:
            print(
                "\nUpdate asv_benchmarks/asv.conf.json's 'pythons' / "
                "'matrix' to match, in a dedicated commit (or re-run "
                "this script with --fix)."
            )
        return 1

    print("asv_benchmarks/asv.conf.json is in sync with pyproject.toml.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


try:
    import pytest

    # ---------------- TESTS ----------------

    @pytest.fixture
    def pyproject(tmp_path):
        """Minimal pyproject.toml exercising the "include-group" case."""
        content = """
        [project]
        requires-python = ">=3.11"

        [dependency-groups]
        min = ["numpy==1.26.0", "scikit-learn==1.6.0,!=1.7.0"]
        min_plotting = [
            {include-group = "min"},
            "matplotlib==3.8.0",
        ]
        """
        path = tmp_path / "pyproject.toml"
        path.write_text(content)
        return path

    @pytest.mark.ai_generated
    def test_min_versions_from_pyproject(pyproject):
        """Resolve include-group and keep only "==" pinned versions."""
        assert min_versions_from_pyproject(pyproject) == {
            "numpy": "1.26.0",
            "scikit-learn": "1.6.0",
            "matplotlib": "3.8.0",
        }

    @pytest.mark.ai_generated
    def test_min_python_from_pyproject(pyproject):
        """Extract the "major.minor" floor from requires-python."""
        assert min_python_from_pyproject(pyproject) == "3.11"

    @pytest.mark.ai_generated
    def test_strip_json_comments_keeps_urls_intact():
        """A "//" inside a string value must survive comment-stripping."""
        text = (
            "{\n"
            "    // a comment\n"
            '    "repo": "https://github.com/nilearn/nilearn.git"\n'
            "}\n"
        )
        assert json.loads(_strip_json_comments(text)) == {
            "repo": "https://github.com/nilearn/nilearn.git"
        }

    @pytest.mark.ai_generated
    @pytest.mark.parametrize(
        "asv_conf, pyproject_versions, pyproject_min_python, expected",
        [
            (
                {"pythons": ["3.11"], "matrix": {"numpy": ["1.26.0"]}},
                {"numpy": "1.26.0"},
                "3.11",
                [],
            ),
            (
                {"pythons": ["3.10"], "matrix": {}},
                {},
                "3.11",
                [
                    "asv.conf.json 'pythons' is ['3.10'] but "
                    "pyproject.toml's 'requires-python' floor is '3.11'."
                ],
            ),
            (
                {"pythons": ["3.11"], "matrix": {"numpy": ["1.22.4"]}},
                {"numpy": "1.26.0"},
                "3.11",
                [
                    "asv.conf.json pins 'numpy' to ['1.22.4'] but "
                    "pyproject.toml's 'min_plotting' group pins it to "
                    "'1.26.0'."
                ],
            ),
            (
                {"pythons": ["3.11"], "matrix": {"unknown-pkg": ["1.0"]}},
                {},
                "3.11",
                [
                    "asv.conf.json pins 'unknown-pkg' but pyproject.toml's "
                    "'min_plotting' group has no '==' pinned version "
                    "for it."
                ],
            ),
            (
                # an empty pin ("latest") is never a mismatch.
                {"pythons": ["3.11"], "matrix": {"plotly": []}},
                {},
                "3.11",
                [],
            ),
        ],
    )
    def test_check_in_sync(
        asv_conf, pyproject_versions, pyproject_min_python, expected
    ):
        """Cover the python-version mismatch, pin mismatch, unknown
        package and "latest"-pin cases.
        """
        assert (
            check_in_sync(asv_conf, pyproject_versions, pyproject_min_python)
            == expected
        )

    @pytest.fixture
    def stale_asv_conf_text():
        """Build a stale asv.conf.json, including a "//"-in-URL case."""
        return (
            "{\n"
            '    // "repo" contains a "//" that must survive rewrites\n'
            '    "repo": "https://github.com/nilearn/nilearn.git",\n'
            '    "pythons": ["3.10"],\n'
            '    "matrix": {\n'
            '        "numpy": ["1.22.4"],\n'
            '        "matplotlib": ["3.8.0"],\n'
            '        "unknown-pkg": ["1.0"]\n'
            "    }\n"
            "}\n"
        )

    @pytest.mark.ai_generated
    def test_apply_fixes_rewrites_only_resolvable_mismatches(
        stale_asv_conf_text,
    ):
        """`pythons` and known packages are fixed; unknown ones are not."""
        parsed = json.loads(_strip_json_comments(stale_asv_conf_text))

        fixed_text = apply_fixes(
            stale_asv_conf_text,
            parsed,
            pyproject_versions={"numpy": "1.26.0", "matplotlib": "3.8.0"},
            pyproject_min_python="3.11",
        )
        fixed = json.loads(_strip_json_comments(fixed_text))

        assert fixed["pythons"] == ["3.11"]
        assert fixed["matrix"] == {
            "numpy": ["1.26.0"],
            "matplotlib": ["3.8.0"],  # already in sync: left untouched
            "unknown-pkg": ["1.0"],  # not resolvable: left untouched
        }
        # the "//" inside the "repo" URL must survive the rewrite
        assert "https://github.com/nilearn/nilearn.git" in fixed_text

    @pytest.mark.ai_generated
    def test_main_fix_end_to_end(tmp_path, monkeypatch, pyproject):
        """`main(["--fix"])` rewrites the file on disk and exits 0 once
        every mismatch it can resolve has been fixed.
        """
        asv_conf_path = tmp_path / "asv.conf.json"
        asv_conf_path.write_text(
            "{\n"
            '    "pythons": ["3.10"],\n'
            '    "matrix": {\n'
            '        "numpy": ["1.22.4"],\n'
            '        "matplotlib": ["3.8.0"]\n'
            "    }\n"
            "}\n"
        )
        monkeypatch.setattr(f"{__name__}.ASV_CONF", asv_conf_path)
        monkeypatch.setattr(f"{__name__}.PYPROJECT", pyproject)

        exit_code = main(["--fix"])

        assert exit_code == 0
        fixed = load_asv_conf(asv_conf_path)
        assert fixed["pythons"] == ["3.11"]
        assert fixed["matrix"] == {
            "numpy": ["1.26.0"],
            "matplotlib": ["3.8.0"],
        }

    @pytest.mark.ai_generated
    def test_main_fix_still_fails_on_unresolvable_mismatch(
        tmp_path, monkeypatch, pyproject
    ):
        """A package pyproject.toml does not pin keeps `--fix` exiting 1,
        even though the resolvable "pythons" mismatch was fixed.
        """
        asv_conf_path = tmp_path / "asv.conf.json"
        asv_conf_path.write_text(
            "{\n"
            '    "pythons": ["3.10"],\n'
            '    "matrix": {\n'
            '        "unknown-pkg": ["1.0"]\n'
            "    }\n"
            "}\n"
        )
        monkeypatch.setattr(f"{__name__}.ASV_CONF", asv_conf_path)
        monkeypatch.setattr(f"{__name__}.PYPROJECT", pyproject)

        exit_code = main(["--fix"])

        assert exit_code == 1
        fixed = load_asv_conf(asv_conf_path)
        # the fixable part was still applied
        assert fixed["pythons"] == ["3.11"]
        assert fixed["matrix"] == {"unknown-pkg": ["1.0"]}

except ImportError:
    ...

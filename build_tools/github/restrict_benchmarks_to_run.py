"""
Identify the files changed in a pull request
and select the subset of asv benchmarks to run.

Reuses restrict_tests_to_run.restrict_tests, which already encodes the
layered import architecture enforced by import-linter, to compute
which nilearn subpackages are affected by a change, then maps that
down to the asv_benchmarks/benchmarks modules that actually cover
them.
"""

import contextlib
import subprocess
import sys
from pathlib import Path

from restrict_tests_to_run import restrict_tests

with contextlib.suppress(Exception):
    from rich import print

# nilearn subpackages that currently have asv benchmark coverage,
# mapped to the "-b" regex used to select their benchmarks.
# Since asv_benchmarks/benchmarks mirrors nilearn's own package
# structure (see CONTRIBUTING.rst), the subpackage name is also the
# benchmark module prefix, except for "nilearn/utils": its benchmarks
# live in discovery.py rather than a nested utils/ directory, because
# "utils" was already taken by asv_benchmarks/benchmarks/utils.py, the
# suite's shared fixtures.
BENCHMARKED_SUBPACKAGES = {
    "nilearn/glm": "glm",
    "nilearn/image": "image",
    "nilearn/maskers": "maskers",
    "nilearn/mass_univariate": "mass_univariate",
    "nilearn/plotting": "plotting",
    "nilearn/utils": "discovery",
}

# Changes to any of these affect the benchmark suite itself, or things
# used everywhere: always run the full suite when they are touched,
# rather than trying to guess which benchmarks could be affected.
RUN_EVERYTHING_ON_CHANGES_TO = [
    "nilearn/_utils",
    "nilearn/conftest.py",
    "nilearn/exceptions.py",
    "nilearn/signal.py",
    "nilearn/nilearn_typing.py",
    "nilearn/_assets",
    "asv_benchmarks/benchmarks/utils.py",
    "asv_benchmarks/asv.conf.json",
    ".github/workflows/benchmark.yml",
]

# A "-b" regex that never matches any benchmark name, used when some
# subpackages changed but none of them (nor their higher layers) have
# any benchmark coverage: asv will then run ~0 benchmarks almost
# instantly, which is a much simpler way to express "nothing relevant
# to benchmark" than skipping workflow steps conditionally.
NOTHING_TO_BENCHMARK = "(?!)"


def root_folder() -> Path:
    """Return local nilearn folder."""
    return Path(__file__).parents[2]


def list_changed_files(base_ref: str) -> list[str]:
    """List files changed between HEAD and base_ref."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", base_ref],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().split("\n")


def restrict_benchmarks(changed_files: list[str]) -> list[str] | None:
    """Return the "-b" regexes to run for the given changed files.

    Returns ``None`` if the full benchmark suite should be run
    (a foundational file changed), or a (possibly empty) list of
    regexes otherwise.
    An empty list means no benchmarked subpackage was affected.
    """
    if any(
        f.startswith(trigger)
        for f in changed_files
        for trigger in RUN_EVERYTHING_ON_CHANGES_TO
    ):
        return None

    subpackages_to_test = restrict_tests(changed_files)

    return sorted(
        {
            benchmark_prefix
            for subpackage, benchmark_prefix in BENCHMARKED_SUBPACKAGES.items()
            if subpackage in subpackages_to_test
        }
    )


def print_to_file(
    benchmarks_to_run: list[str] | None, output_path=None
) -> None:
    """Dump the "-b" filter to run to a file.

    An empty file means: run the full suite. Otherwise the file
    contains a single "-b"-ready regex.
    """
    if benchmarks_to_run is None:
        content = ""
    elif len(benchmarks_to_run) == 0:
        content = NOTHING_TO_BENCHMARK
    else:
        content = "|".join(benchmarks_to_run)

    print(f"Will run benchmarks matching: {content or '<all>'}")

    if output_path is None:
        output_path = root_folder()
    output_file = output_path / "benchmarks_to_run.txt"
    with output_file.open("w") as f:
        f.write(content)


def main() -> None:
    """Save to disk the "-b" filter for the benchmarks to run."""
    print("Identifying benchmarks to run.")

    base_ref = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    print(f"Comparing against: '{base_ref}'")

    changed_files = list_changed_files(base_ref)
    print(f"{changed_files=}")

    benchmarks_to_run = restrict_benchmarks(changed_files)

    print_to_file(benchmarks_to_run)


if __name__ == "__main__":
    main()

try:
    import pytest

    # ---------------- TESTS ----------------

    @pytest.mark.parametrize(
        "benchmarks_to_run, expected_content",
        [
            (None, ""),
            ([], NOTHING_TO_BENCHMARK),
            (["glm", "image"], "glm|image"),
        ],
    )
    def test_print_to_file(tmp_path, benchmarks_to_run, expected_content):
        """Check content printed to disk."""
        print_to_file(benchmarks_to_run, tmp_path)
        output_file = tmp_path / "benchmarks_to_run.txt"
        assert output_file.is_file()
        with output_file.open("r") as f:
            content = f.read()
        assert content == expected_content

    @pytest.mark.parametrize(
        "changed_files, expected_benchmarks_to_run",
        [
            ([], []),
            (
                ["nilearn/glm/first_level/first_level.py"],
                ["discovery", "glm"],
            ),
            (
                ["nilearn/maskers/nifti_masker.py"],
                ["discovery", "glm", "image", "maskers", "plotting"],
            ),
            (
                ["nilearn/mass_univariate/permuted_least_squares.py"],
                ["discovery", "glm", "mass_univariate"],
            ),
            # decoding itself is not benchmarked, but it is a top layer,
            # whose changes always also affect the highest layer (utils)
            (["nilearn/decoding/decoder.py"], ["discovery"]),
            (["nilearn/_utils/data_gen.py"], None),
            (["asv_benchmarks/benchmarks/utils.py"], None),
        ],
    )
    def test_restrict_benchmarks(changed_files, expected_benchmarks_to_run):
        """Check subset of benchmarks to run."""
        assert restrict_benchmarks(changed_files) == expected_benchmarks_to_run

except Exception:
    ...

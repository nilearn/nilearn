#!/usr/bin/env python

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "rich"
# ]
# ///

"""
Checks what doc build type to run.

Generate:
- 'pattern.txt' that lists the examples to build.
  Its content is passed to sphinx_gallery_conf.filename_pattern
  via a PATTERN env variable.
- 'build.txt' that contains the build type to run.

On the main branch, on tagged released or if requested in the commit message,
this exits early and sets things for a full doc build.
Otherwise it checks if any examples must be built
(either because it was changed in the PR or requested in the commit message).

Must be run after running build_tools/github/merge_upstream.sh
"""

import builtins
import contextlib
import os
import subprocess
from pathlib import Path

with contextlib.suppress(builtins.BaseException):
    from rich import print


def run(cmd, capture_output=True, check=True, text=True):
    """Run a shell command and return its stdout (if captured)."""
    result = subprocess.run(
        cmd, shell=True, capture_output=capture_output, text=text, check=check
    )
    return result.stdout.strip() if capture_output else None


def main():
    """Check what doc build type to run."""
    if not Path("gitlog.txt").exists():
        raise RuntimeError(
            "'gitlog.txt' not found. "
            "It should have been generated "
            "by 'build_tools/github/merge_upstream.sh'"
        )
    if not Path("merge.txt").exists():
        raise RuntimeError(
            "'merge.txt' not found. "
            "It should have been generated "
            "by 'build_tools/github/merge_upstream.sh'"
        )

    # Set missing variables
    CI = os.getenv("CI")
    if CI is None:
        print("Running locally")
        COMMIT_SHA = run("git log --format=format:%H -n 1")
        GITHUB_REF_TYPE = "branch"
    else:
        print("Running in CI")
        COMMIT_SHA = os.getenv("COMMIT_SHA") or run(
            "git log --format=format:%H -n 1"
        )
        GITHUB_REF_TYPE = os.getenv("GITHUB_REF_TYPE", "")

    GITHUB_REF_NAME = Path("merge.txt").read_text().strip()
    COMMIT_MSG = Path("gitlog.txt").read_text()

    print(f"{COMMIT_MSG=}")

    # Ensure pattern.txt exists even if empty
    Path("pattern.txt").touch()

    # Check for full build
    if (
        GITHUB_REF_NAME == "main"
        or GITHUB_REF_TYPE == "tag"
        or "[full doc]" in COMMIT_MSG
    ):
        print("Doing a full build")
        Path("build.txt").write_text("html-strict\n")
        return

    GENERATE_REPORT = "[reports]" in COMMIT_MSG

    # Check for [example] in commit message
    example = []
    if "[example]" in COMMIT_MSG:
        print(f"Building selected example:\n{COMMIT_MSG}")
        # Extract everything after the first "] "
        try:
            examples_in_message = COMMIT_MSG.split("] ", 1)[1].strip()
            for ex in examples_in_message.split(" "):
                example.extend([f"examples/*/{ex.strip()}"])
        except IndexError:
            ...

    # Generate examples.txt
    merge_base = run(f"git merge-base {COMMIT_SHA} upstream/main")
    changed_files = run(
        f"git diff --name-only {merge_base} {COMMIT_SHA}"
    ).splitlines()

    # Append examples to the list
    changed_files.extend(example)

    Path("examples.txt").write_text("\n".join(changed_files) + "\n")

    # ----- Filter examples and build pattern
    pattern_parts = []
    for filename in changed_files:
        if (
            "examples/" in filename
            and "plot_" in filename
            and filename.endswith(".py")
        ):
            print(f"Checking example {filename} ...")
            pattern_parts.append(Path(filename).name)

    build_type = "ci-html-noplot"
    pattern = ""
    if pattern_parts:
        pattern = r"\(" + "\\|".join(pattern_parts) + r"\)"
        build_type = "html-modified-examples-only"

    if GENERATE_REPORT:
        build_type += "-reports"

    Path("build.txt").write_text(build_type + "\n")

    Path("pattern.txt").write_text(pattern + "\n")

    print(f"{pattern=}")


if __name__ == "__main__":
    main()


try:
    import pytest

    @pytest.fixture
    def clean_up():
        """Remove files created."""
        yield
        Path("build.txt").unlink()
        Path("pattern.txt").unlink()
        Path("examples.txt").unlink(missing_ok=True)

    @pytest.fixture
    def gitlog(commit_msg):
        """Create dumy content of a commit message."""
        Path("gitlog.txt").write_text(commit_msg)
        print(commit_msg)
        yield
        Path("gitlog.txt").unlink()

    @pytest.fixture
    def merge():
        """Create dummy file containing files changed in a PR."""
        Path("merge.txt").write_text("")
        yield
        Path("merge.txt").unlink()

    @pytest.mark.parametrize(
        "commit_msg, expected_in_pattern",
        [
            ("", [""]),
            ("[full doc]", [""]),
            ("[example] ", [""]),
            (
                "[example] plot_3d_and_4d_niimg.py",
                [r"\(plot_3d_and_4d_niimg.py\)"],
            ),
            (
                "[example] plot_3d_and_4d_niimg.py plot_oasis.py",
                ["plot_3d_and_4d_niimg.py", "plot_oasis.py"],
            ),
            # TODO: check and implement globbing
            # (
            #     "[example] plot_second_level*.py",
            #     ["plot_3d_and_4d_niimg.py", "plot_oasis.py"],
            # ),
        ],
    )
    def test_examples_in_commit_msg(
        commit_msg,
        gitlog,  # noqa: ARG001
        merge,  # noqa: ARG001
        expected_in_pattern,
        clean_up,  # noqa: ARG001
    ):
        """Test proper examples will be added when passed in commit msg."""
        print(f"{commit_msg=}")
        main()
        assert Path("build.txt").exists()

        if "[full doc]" not in commit_msg:
            assert Path("examples.txt").exists()
            assert Path("pattern.txt").exists()

        with Path("pattern.txt").open("r") as f:
            content = f.read()
            for ex in expected_in_pattern:
                assert ex in content

    @pytest.mark.parametrize(
        "commit_msg, expected_in_pattern",
        [
            ("", "ci-html-noplot\n"),
            ("[full doc]", "html-strict\n"),
            ("[report]", "ci-html-noplot-reports\n"),
            (
                "[example] plot_3d_and_4d_niimg.py",
                "html-modified-examples-only\n",
            ),
            (
                "[report][example] plot_3d_and_4d_niimg.py",
                "html-modified-examples-only-reports\n",
            ),
        ],
    )
    def test_build(
        commit_msg,
        gitlog,  # noqa: ARG001
        merge,  # noqa: ARG001
        expected_in_pattern,
        clean_up,  # noqa: ARG001
    ):
        """Test build type depending on commit message."""
        print(f"{commit_msg=}")
        main()
        assert Path("build.txt").exists()
        with Path("build.txt").open("r") as f:
            assert f.read() == expected_in_pattern


except ImportError:
    ...

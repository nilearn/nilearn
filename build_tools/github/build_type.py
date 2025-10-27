#!/usr/bin/env python3
"""_summary_
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
    GITLOG = Path("gitlog.txt").read_text()

    # Ensure pattern.txt exists even if empty
    Path("pattern.txt").touch()

    # Check for full build
    if (
        GITHUB_REF_NAME == "main"
        or GITHUB_REF_TYPE == "tag"
        or "[full doc]" in GITLOG
    ):
        print("Doing a full build")
        Path("build.txt").write_text("html-strict\n")
        return

    # Check for [example] in commit message
    example = []
    if "[example]" in GITLOG:
        print(f"Building selected example: {GITLOG}")
        # Extract everything after the first "] "
        try:
            examples_in_message = GITLOG.split("] ", 1)[1].strip()
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

    if pattern_parts:
        PATTERN = "(" + "\\|".join(pattern_parts) + ")"
        Path("build.txt").write_text("html-modified-examples-only\n")
    else:
        PATTERN = ""
        Path("build.txt").write_text("ci-html-noplot\n")

    Path("pattern.txt").write_text(PATTERN + "\n")

    print(f"{PATTERN=}")


if __name__ == "__main__":
    main()


try:
    import pytest

    @pytest.fixture
    def clean_up():
        """Remove files created."""
        yield
        Path("build.txt").unlink()
        Path("examples.txt").unlink()
        Path("pattern.txt").unlink()

    @pytest.fixture
    def gitlog(log):
        """Create dumy content of a commit message."""
        Path("gitlog.txt").write_text(log)
        yield
        Path("gitlog.txt").unlink()

    @pytest.fixture
    def merge():
        """Create dummy file containing files changed in a PR."""
        Path("merge.txt").write_text("")
        yield
        Path("merge.txt").unlink()

    @pytest.mark.parametrize(
        "log, expected_in_pattern",
        [
            ("", [""]),
            ("[example] ", [""]),
            ("[example] plot_3d_and_4d_niimg.py", ["plot_3d_and_4d_niimg.py"]),
            (
                "[example] plot_3d_and_4d_niimg.py plot_oasis.py",
                ["plot_3d_and_4d_niimg.py", "plot_oasis.py"],
            ),
            (
                "[example] plot_second_level*",
                ["plot_3d_and_4d_niimg.py", "plot_oasis.py"],
            ),
        ],
    )
    def test_main_commit_msg(
        log,  # noqa: ARG001
        gitlog,  # noqa: ARG001
        merge,  # noqa: ARG001
        expected_in_pattern,
        clean_up,  # noqa: ARG001
    ):
        """Test proper examples will be added when passed in commit msg."""
        main()
        assert Path("build.txt").exists()
        assert Path("examples.txt").exists()
        assert Path("pattern.txt").exists()

        with Path("pattern.txt").open("r") as f:
            content = f.read()
            for ex in expected_in_pattern:
                assert ex in content

except ImportError:
    ...

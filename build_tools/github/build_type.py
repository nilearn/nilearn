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

import os
import subprocess
import sys
from pathlib import Path


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
            "'gitlog.txt' not found."
            "should have been generated "
            "by build_tools/github/merge_upstream.sh"
        )
    if not Path("merge.txt").exists():
        raise RuntimeError(
            "'merge.txt' not found."
            "should have been generated "
            "by build_tools/github/merge_upstream.sh"
        )

    # ----- Set missing variables

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

    # ----- Check for full build
    if (
        GITHUB_REF_NAME == "main"
        or GITHUB_REF_TYPE == "tag"
        or "[full doc]" in GITLOG
    ):
        print("Doing a full build")
        Path("build.txt").write_text("html-strict\n")
        sys.exit(0)

    # ----- Check for [example] in commit message
    EXAMPLE = ""
    if "[example]" in GITLOG:
        print("Building selected example")
        # Extract everything after the first "] "
        try:
            COMMIT_MESSAGE = GITLOG.split("] ", 1)[1].strip()
            EXAMPLE = f"examples/*/{COMMIT_MESSAGE}"
        except IndexError:
            COMMIT_MESSAGE = ""
            EXAMPLE = ""

    # ----- Generate examples.txt
    merge_base = run(f"git merge-base {COMMIT_SHA} upstream/main")
    changed_files = run(
        f"git diff --name-only {merge_base} {COMMIT_SHA}"
    ).splitlines()

    # Append EXAMPLE to the list
    if EXAMPLE:
        changed_files.append(EXAMPLE)

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

    print(f"PATTERN={PATTERN}")


if __name__ == "__main__":
    main()

"""
Identify the files changed in a pull request
and select the subset of tests to to run.

- Files affected are got from git
- Determines the list of subpackages whose tests need to be run
- Dump that list to a txt file so it can be used by other processes downstream
"""

import contextlib
import subprocess
from pathlib import Path

with contextlib.suppress(Exception):
    from rich import print

BASE_TESTS = [
    "nilearn/tests/test_exceptions.py",
    "nilearn/tests/test_init.py",
    "nilearn/tests/test_package_import.py",
]

HIGHEST_LAYER = ["nilearn/utils"]
TOP_LAYER = ["nilearn/glm", "nilearn/decoding", "nilearn/decomposition"]
MID_LAYER = [
    "nilearn/datasets",
    "nilearn/image",
    "nilearn/interfaces",
    "nilearn/maskers",
    "nilearn/plotting",
    "nilearn/reporting",
    "nilearn/regions",
    "nilearn/surface",
]
IGNORE = ["input_data", "__pycache__", "tests"]


def root_folder():
    """Return local nilearn folder."""
    return Path(__file__).parents[2]


# run small sanity check that all folders are accounted for
all_folders = sorted(
    [
        f"nilearn/{x.name!s}"
        for x in (root_folder() / "nilearn").iterdir()
        if (x.is_dir() and x.name not in IGNORE)
    ]
)
known_dirs = sorted(
    [
        *HIGHEST_LAYER,
        *TOP_LAYER,
        *MID_LAYER,
        "nilearn/_utils",
        "nilearn/connectome",
        "nilearn/mass_univariate",
    ]
)
assert known_dirs == all_folders, f"\n{known_dirs=}\n{all_folders=}"


def main() -> None:
    """Save to disk a list of subpackages that need to be tested."""
    print("Identifying tests to run.")

    branch = get_current_branch()
    print(f"Running on branch: '{branch}'")

    if branch == "main":
        tests_to_run = ["nilearn"]
    else:
        changed_files = list_changed_files()
        print(f"{changed_files=}")
        tests_to_run = restrict_tests(changed_files)

    print_to_file(tests_to_run)


def get_current_branch() -> str:
    """Identify the git branch we are on."""
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def list_changed_files() -> list[str]:
    """List files changed in the pull request."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "upstream/main"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().split("\n")


def restrict_tests(changed_files: list[str]) -> list[str]:
    """Restrict subpackages to test given nilearn architecture.

    - by default we run some minimal tests
    - only keep subpackage name
    - include higher levels subpackages if they are included
    """
    subpackages_changed = []
    for x in changed_files:
        if "nilearn" not in x:
            continue
        x = x.split("/")
        if x[1] != "tests":
            subpackages_changed.append("/".join(x[0:2]))
        else:
            subpackages_changed.append("/".join(x))
    subpackages_changed = sorted(set(subpackages_changed))
    print(f"{subpackages_changed=}")

    tests_to_run = []
    for x in subpackages_changed:
        # higher layers
        tests_to_run.extend(
            [x for higher_level in HIGHEST_LAYER if x == higher_level]
        )

        # higher layers
        for top_level in [*TOP_LAYER, "nilearn/connectome"]:
            if x == top_level:
                tests_to_run.extend([x, *HIGHEST_LAYER])

        if "nilearn/mass_univariate" in x:
            tests_to_run.extend(
                ["nilearn/glm", "nilearn/mass_univariate", *HIGHEST_LAYER]
            )

        # middle layers: all imported by top layers
        if any(x == mid_level for mid_level in MID_LAYER):
            tests_to_run.extend(
                [
                    "nilearn/tests/test_masking.py",
                    *MID_LAYER,
                    *TOP_LAYER,
                    *HIGHEST_LAYER,
                ]
            )

        # lower levels
        if x == "nilearn/masking.py":
            tests_to_run.extend(
                [
                    "nilearn/tests/test_masking.py",
                    *MID_LAYER,
                    *TOP_LAYER,
                    *HIGHEST_LAYER,
                ]
            )

        # if lowest layers or _utils or test config is touched:
        # everything may be affected
        if any(
            x == lowest_layer
            for lowest_layer in [
                "nilearn/_utils",
                "nilearn/conftest.py",
                "nilearn/exceptions.py",
                "nilearn/signal.py",
                "nilearn/typing.py",
            ]
        ):
            tests_to_run.extend(
                [
                    "nilearn/tests/test_masking.py",
                    "nilearn/tests/test_signal.py",
                    "nilearn/connectome",
                    *MID_LAYER,
                    *TOP_LAYER,
                    *HIGHEST_LAYER,
                ]
            )

        # edge case where some tests files where changed
        tests_to_run.extend(
            [
                x
                for test_file in [
                    "nilearn/tests/test_masking.py",
                    "nilearn/tests/test_signal.py",
                ]
                if x == test_file
            ]
        )

    # we always run some base tests
    tests_to_run.extend(BASE_TESTS)

    tests_to_run = sorted(set(tests_to_run))

    return tests_to_run


def print_to_file(tests_to_run: list[str], output_path=None) -> None:
    """Dump list of subpackages whose tests to run to a file.

    Fall back to running all tests if an empty list is passed.
    """
    if len(tests_to_run) == 0:
        tests_to_run = ["nilearn"]

    print(f"Will run tests on {tests_to_run}")

    if output_path is None:
        output_path = root_folder()
    output_file = output_path / "tests_to_run.txt"
    with output_file.open("w") as f:
        f.write(" ".join(tests_to_run))


if __name__ == "__main__":
    main()

try:
    import pytest

    # ---------------- TESTS ----------------

    @pytest.mark.parametrize(
        "tests_to_run, expected_content",
        [([], "nilearn"), (["foo", "bar"], "foo bar")],
    )
    def test_print_to_file(tmp_path, tests_to_run, expected_content):
        """Check content printed to disk."""
        print_to_file(tests_to_run, tmp_path)
        assert (tmp_path / "tests_to_run.txt").is_file()
        with (tmp_path / "tests_to_run.txt").open("r") as f:
            content = f.read()
        assert content == expected_content

    @pytest.mark.parametrize(
        "changed_files, expected_subpackages_to_test",
        [
            ([], []),
            (
                ["nilearn/glm/first_level/first_level.py"],
                ["nilearn/glm", *HIGHEST_LAYER],
            ),
            (
                ["nilearn/decoding/decoder.py"],
                ["nilearn/decoding", *HIGHEST_LAYER],
            ),
            (
                ["nilearn/connectome/group_sparse_cov.py"],
                ["nilearn/connectome", *HIGHEST_LAYER],
            ),
            (
                ["nilearn/mass_univariate/permuted_least_squares.py"],
                ["nilearn/glm", "nilearn/mass_univariate", *HIGHEST_LAYER],
            ),
            (
                ["nilearn/plotting/cm.py"],
                [
                    *HIGHEST_LAYER,
                    *TOP_LAYER,
                    *MID_LAYER,
                    "nilearn/tests/test_masking.py",
                ],
            ),
            (
                ["nilearn/maskers/nifti_masker.py"],
                [
                    *HIGHEST_LAYER,
                    *TOP_LAYER,
                    *MID_LAYER,
                    "nilearn/tests/test_masking.py",
                ],
            ),
            (
                ["nilearn/signal.py"],
                [
                    *HIGHEST_LAYER,
                    *TOP_LAYER,
                    *MID_LAYER,
                    "nilearn/tests/test_masking.py",
                    "nilearn/tests/test_signal.py",
                    "nilearn/connectome",
                ],
            ),
            (
                ["nilearn/conftest.py"],
                [
                    *HIGHEST_LAYER,
                    *TOP_LAYER,
                    *MID_LAYER,
                    "nilearn/tests/test_masking.py",
                    "nilearn/tests/test_signal.py",
                    "nilearn/connectome",
                ],
            ),
            (
                ["nilearn/tests/test_signal.py"],
                ["nilearn/tests/test_signal.py"],
            ),
        ],
    )
    def test_restrict_tests(changed_files, expected_subpackages_to_test):
        """Check subset of tests to run."""
        subpackages_to_test = restrict_tests(changed_files)
        assert subpackages_to_test == sorted(
            {*expected_subpackages_to_test, *BASE_TESTS}
        )

except Exception:
    ...

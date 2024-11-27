"""Collects and plot run time of jobs in a GHA workflow.

Requires:
- requests
- rich
- pandas
- plotly
- kaleido

For a given github action workflow:
- ping the github API to collect the start and end time
  of all the jobs of the different runs of that workflow,
- saves to TSV
- plots the duration of each job against time.

This script should in principle run for any repo and any workflow.

This may require a github token to work
(if you need to make a lot of call to the GitHub API).

You can get a github token at:
https://github.com/settings/tokens

You can either:

- save the github token in a file and modify the script
  (see the variable USERNAME and TOKEN_FILE below).
  This is can be useful if you want to run the script locally.

- pass the token directly to the script as an argument.
  This is the way to do it when using this script in continuous integration.

USAGE
-----

.. code-block:: bash

    python maint_tools/check_gha_workflow.py $GITHUB_TOKEN
"""

import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import requests
from rich import print

USER = "nilearn"
REPO = "nilearn"

# If you do not pass a github token directly to the script,
# you must provide your github user name and the path to a file
# containing your github action token.
# See the doc string the top of the script for more details.

# your github username
USERNAME = "Remi-Gau"
# file containing your github token
TOKEN_FILE = Path("/home/remi/Documents/tokens/gh_read_repo_for_orga.txt")

BRANCH = "main"

# Set to True if yu want to also include the runs of a CI workflow
# that did not complete successfully.
INCLUDE_FAILED_RUNS = False

# Pages of runs to collect
# 100 per page
PAGES_TO_COLLECT = range(1, 30)

# If False, just plots the content of the TSV
UPDATE_TSV = True

# used by set_python_version to filter jobs by their python version
EXPECTED_PYTHON_VERSIONS = ["3.8", "3.9", "3.10", "3.11", "3.12", "3.13"]


def main(args=sys.argv) -> None:
    """Collect duration of each job and plots them."""
    # %%
    # can be found out at
    # "https://api.github.com/repos/{USER}/{REPO}/actions/workflows"
    TEST_WORKFLOW_ID = "71549417"
    output_file = Path(__file__).parent / "test_runs_timing.tsv"

    _update_tsv(
        args,
        update=UPDATE_TSV,
        output_file=output_file,
        workflow_id=TEST_WORKFLOW_ID,
    )

    test_runs_timing = pd.read_csv(
        output_file,
        sep="\t",
        parse_dates=["started_at", "completed_at"],
    )

    test_runs_timing["duration"] = (
        test_runs_timing["completed_at"] - test_runs_timing["started_at"]
    ) / pd.Timedelta(minutes=1)
    test_runs_timing["python"] = test_runs_timing["name"].apply(
        _set_python_version
    )
    test_runs_timing["OS"] = test_runs_timing["name"].apply(_set_os)
    test_runs_timing["dependencies"] = test_runs_timing["name"].apply(
        _set_dependencies
    )

    print(test_runs_timing)

    _plot_test_job_durations(test_runs_timing, output_file)

    # %%
    DOC_WORKFLOW_ID = "37349438"
    output_file = Path(__file__).parent / "doc_runs_timing.tsv"

    _update_tsv(
        args,
        update=UPDATE_TSV,
        output_file=output_file,
        workflow_id=DOC_WORKFLOW_ID,
        event_type="push",
    )

    doc_runs_timing = pd.read_csv(
        output_file,
        sep="\t",
        parse_dates=["started_at", "completed_at"],
    )

    doc_runs_timing["duration"] = (
        doc_runs_timing["completed_at"] - doc_runs_timing["started_at"]
    ) / pd.Timedelta(minutes=1)

    doc_runs_timing = doc_runs_timing[doc_runs_timing["name"] == "build_docs"]
    doc_runs_timing = doc_runs_timing[doc_runs_timing["duration"] < 360]

    print(doc_runs_timing)

    _plot_doc_job_durations(doc_runs_timing, output_file)


def _update_tsv(
    args,
    update: bool,
    output_file: Path,
    workflow_id: str,
    event_type: None | str = None,
) -> None:
    """Update TSV containing run time of every workflow."""
    update_tsv = update if output_file.exists() else True

    if not update_tsv:
        return

    if len(args) > 1:
        TOKEN = args[1]
        auth = {"Authorization": "token " + TOKEN}
    else:
        auth = _get_auth(USERNAME, TOKEN_FILE)

    jobs_data = {"name": [], "started_at": [], "completed_at": []}

    for page in PAGES_TO_COLLECT:
        runs = _get_runs(
            workflow_id,
            auth,
            page=page,
            include_failed_runs=INCLUDE_FAILED_RUNS,
            event_type=event_type,
        )
        if len(runs) > 0:
            print(f" found {len(runs)} runs")
            jobs_data = _update_jobs_data(jobs_data, runs, auth)
        else:
            break

    pd.DataFrame(jobs_data).to_csv(output_file, sep="\t", index=False)


def _plot_test_job_durations(df: pd.DataFrame, output_file: Path) -> None:
    """Plot and save."""
    fig = px.line(
        df,
        x="started_at",
        y="duration",
        color="python",
        symbol="dependencies",
        line_dash="OS",
        labels={
            "duration": "Run duration (minutes)",
            "started_at": "Run started on",
            "OS": "OS",
            "python": "python version",
        },
        title="Duration of nilearn test runs",
    )

    fig.update_xaxes(dtick="M1", tickformat="%b\n%Y")
    fig.update_layout(autosize=True, width=1000, height=700)

    fig.write_image(output_file.with_suffix(".png"), engine="kaleido")
    fig.write_html(output_file.with_suffix(".html"))


def _plot_doc_job_durations(df: pd.DataFrame, output_file: Path) -> None:
    """Plot and save."""
    fig = px.line(
        df,
        x="started_at",
        y="duration",
        title="Duration of nilearn doc build",
    )

    fig.update_xaxes(dtick="M1", tickformat="%b\n%Y")
    fig.update_layout(autosize=True, width=1000, height=700)

    fig.write_image(output_file.with_suffix(".png"), engine="kaleido")
    fig.write_html(output_file.with_suffix(".html"))


def _set_os(x: str) -> str:
    """Detect which OS the job was run on.

    .. note::
        that this depend on the way each github acton job is named.
        So this may not work as well with other repositories than Nilearn
        that uses the following naming pattern for the test job:

    'Test with ${{ matrix.py }} on ${{ matrix.os }}: ${{ matrix.description }}'
    """
    if "ubuntu" in x:
        return "ubuntu"
    elif "windows" in x:
        return "windows"
    elif "macos" in x:
        return "macos"
    else:
        return "n/a"


def _set_python_version(x: str) -> str:
    """Detect which python version the job was run on.

    .. note::
        that this depend on the way each github acton job is named.
        So this may not work as well with other repositories than Nilearn
        that uses the following naming pattern for the test job:

    'Test with ${{ matrix.py }} on ${{ matrix.os }}: ${{ matrix.description }}'
    """
    return next(
        (version for version in EXPECTED_PYTHON_VERSIONS if version in x),
        "n/a",
    )


def _set_dependencies(x: str) -> str:
    """Detect which set of dependencies was used for the run.

    .. note::
        that this depend on the way each github acton job is named.
        So this may not work as well with other repositories than Nilearn
        that uses the following naming pattern for the test job:

    'Test with ${{ matrix.py }} on ${{ matrix.os }}: ${{ matrix.description }}'
    """
    return next(
        (
            dependencies
            for dependencies in ["pre-release", "no plotting", "no plotly"]
            if dependencies in x
        ),
        "latest" if "latest dependencies" in x else "n/a",
    )


def _get_auth(username: str, token_file: Path) -> None | tuple[str, str]:
    """Get authentication with token."""
    token = None

    if token_file.exists():
        with token_file.open() as f:
            token = f.read().strip()
    else:
        warnings.warn(f"Token file not found.\n{token_file!s}")

    return None if username is None or token is None else (username, token)


def _get_runs(
    workflow_id: str,
    auth: None | tuple[str, str] = None,
    page: int = 1,
    include_failed_runs: bool = True,
    event_type: None | str = None,
) -> list[dict[str, Any]]:
    """Get list of runs for a workflow.

    Restricted to:
    - completed runs
    """
    status = "completed"

    source = f"https://api.github.com/repos/{USER}/{REPO}/actions/workflows"
    query = f"?per_page=100&status={status}&branch={BRANCH}&page={page}"
    url = f"{source}/{workflow_id}/runs{query}"

    print(f"pinging: {url}")

    content = _handle_request(url, auth)

    if not content.get("workflow_runs"):
        return []

    runs = list(content["workflow_runs"])

    if event_type:
        runs = [run for run in runs if run["event"] in event_type]

    conclusion = ["success"]
    if include_failed_runs:
        conclusion = ["success", "failure"]

    return [run for run in runs if run["conclusion"] in conclusion]


def _handle_request(url: str, auth: None | tuple[str, str]):
    """Wrap request."""
    if isinstance(auth, tuple):
        response = requests.get(url, auth=auth)
    elif isinstance(auth, dict):
        response = requests.get(url, headers=auth)

    if response.status_code != 200:
        raise RuntimeError(
            f"\n Got code: {response.status_code}.\n"
            f" With response: {response.json()}"
        )

    return response.json()


def _update_jobs_data(
    jobs_data: dict[str, list[str]],
    runs: list[dict[str, Any]],
    auth: None | tuple[str, str] = None,
) -> dict[str, list[str]]:
    """Collect info for each job in a run."""
    for run in runs:
        print(f'{run["id"]}: {run["display_title"]}')

        content = _handle_request(run["jobs_url"], auth)

        for job in content.get("jobs", {}):
            for key, value in jobs_data.items():
                value.append(job[key])

    return jobs_data


if __name__ == "__main__":
    main()

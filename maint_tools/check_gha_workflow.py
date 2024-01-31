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
- saves to TSV and plots the duration of each job against time.

This script should in principle run for any repo and any workflow.

This may require a github token to function
that you can pass to the script
or save in a file the script will read.

USAGE
-----

.. code-block:: bash

    python maint_tools/check_gha_workflow.py

    # or by passing github token directly
    python maint_tools/check_gha_workflow.py GITHUB_TOKEN


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

# Your github username
USERNAME = "Remi-Gau"
# file containing your github token
# get one at:
# https://github.com/settings/tokens
TOKEN_FILE = Path("/home/remi/Documents/tokens/gh_read_repo_for_orga.txt")

# can be found out at
# "https://api.github.com/repos/{USER}/{REPO}/actions/workflows"
WORKFLOW_ID = "71549417"

INCLUDE_FAILED_RUNS = True

# Pages of runs to collect
# 100 per page
PAGES_TO_COLLECT = range(1, 20)

# If False, just plots the content of the TSV
UPDATE_TSV = True

OUTPUT_FILE = Path(__file__).parent / "test_runs_timing.tsv"


def main(args=sys.argv) -> None:
    """Collect duration of each job and plots them."""
    update_tsv = UPDATE_TSV if OUTPUT_FILE.exists() else True

    if update_tsv:

        if len(args) > 1:
            TOKEN = args[1]
            auth = {"Authorization": "token " + TOKEN}
        else:
            auth = get_auth(USERNAME, TOKEN_FILE)

        jobs_data = {"name": [], "started_at": [], "completed_at": []}

        for page in PAGES_TO_COLLECT:
            runs = get_runs(
                WORKFLOW_ID,
                auth,
                page=page,
                include_failed_runs=INCLUDE_FAILED_RUNS,
            )
            if len(runs) > 0:
                print(f" found {len(runs)} runs")
                jobs_data = udpate_jobs_data(jobs_data, runs, auth)
            else:
                break

        df = pd.DataFrame(jobs_data)
        df.to_csv(OUTPUT_FILE, sep="\t", index=False)

    df = pd.read_csv(
        OUTPUT_FILE,
        sep="\t",
        parse_dates=["started_at", "completed_at"],
    )

    df["duration"] = (df["completed_at"] - df["started_at"]) / pd.Timedelta(
        minutes=1
    )
    df["python"] = df["name"].apply(set_python_version)
    df["OS"] = df["name"].apply(set_os)
    df["dependencies"] = df["name"].apply(set_dependencies)

    print(df)

    plot_job_durations(df)


def plot_job_durations(df: pd.DataFrame) -> None:
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

    fig.write_image(OUTPUT_FILE.with_suffix(".png"))
    fig.write_html(OUTPUT_FILE.with_suffix(".html"))


def set_os(x: str) -> str:
    """Detect which OS the job was run on."""
    if "ubuntu" in x:
        return "ubuntu"
    elif "windows" in x:
        return "windows"
    elif "macos" in x:
        return "macos"
    else:
        return "n/a"


def set_python_version(x: str) -> str:
    """Detect which python version the job was run on."""
    return next(
        (
            version
            for version in ["3.8", "3.9", "3.10", "3.11", "3.12"]
            if version in x
        ),
        "n/a",
    )


def set_dependencies(x: str) -> str:
    """Detect which set of dependencies was used for the run."""
    return next(
        (
            dependencies
            for dependencies in ["pre-release", "no plotting", "no plotly"]
            if dependencies in x
        ),
        "latest" if "latest dependencies" in x else "n/a",
    )


def get_auth(username: str, token_file: Path) -> None | tuple[str, str]:
    """Get authentication with token."""
    token = None

    if token_file.exists():
        with open(token_file) as f:
            token = f.read().strip()
    else:
        warnings.warn(f"Token file not found.\n{str(token_file)}")

    return None if username is None or token is None else (username, token)


def get_runs(
    workflow_id: str,
    auth: None | tuple[str, str] = None,
    page: int = 1,
    include_failed_runs: bool = True,
) -> list[dict[str, Any]]:
    """Get list of runs for a workflow.

    Restricted to:
    - main branch
    - completed runs
    """
    status = "completed"
    branch = "main"

    source = f"https://api.github.com/repos/{USER}/{REPO}/actions/workflows"
    query = f"?per_page=100&status={status}&branch={branch}&page={page}"
    url = f"{source}/{workflow_id}/runs{query}"

    print(f"pinging: {url}")

    content = handle_request(url, auth)

    if not content.get("workflow_runs"):
        return []
    if include_failed_runs:
        return [
            i
            for i in content["workflow_runs"]
            if i["conclusion"] in ["success", "failure"]
        ]
    return [
        i for i in content["workflow_runs"] if i["conclusion"] == "success"
    ]


def handle_request(url: str, auth: None | tuple[str, str]):
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


def udpate_jobs_data(
    jobs_data: dict[str, list[str]],
    runs: list[dict[str, Any]],
    auth: None | tuple[str, str] = None,
) -> dict[str, list[str]]:
    """Collect info for each job in a run."""
    for run in runs:
        print(f'{run["id"]}: {run["display_title"]}')

        content = handle_request(run["jobs_url"], auth)

        for job in content.get("jobs", {}):
            for key in jobs_data:
                jobs_data[key].append(job[key])

    return jobs_data


if __name__ == "__main__":
    main()

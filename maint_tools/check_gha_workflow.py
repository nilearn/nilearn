"""Collects and plot run time of jobs in a GHA workflow.

Requires:
- request
- rich
- pandas
- plotly

For a given github action workflow:
- ping the github API to collect the start and end time
  of all the jobs of the different runs of that workflow,
- saves to TSV and plots the duration of each job against time.

This script should in principle run for any repo and any workflow.

"""
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
# file containing the github token
# get one at:
# https://github.com/settings/tokens
TOKEN_FILE = Path("/home/remi/Documents/tokens/gh_read_repo_for_orga.txt")

# can be found out at
# "https://api.github.com/repos/{USER}/{REPO}/actions/workflows"
WORKFLOW_ID = "71549417"

INCLUDE_FAILED_RUNS = True

# Pages of runs to collect
# 100 per page
PAGES_TO_COLLECT = [1]

OUTPUT_FILE = Path(__file__).parent / "test_runs_timing.tsv"


def main() -> None:
    """Collect duration of each job and plots them."""
    auth = get_auth(USERNAME, TOKEN_FILE)

    jobs_data = {"name": [], "started_at": [], "completed_at": []}

    for page in PAGES_TO_COLLECT:
        runs = get_runs(
            WORKFLOW_ID,
            auth,
            page=page,
            include_failed_runs=INCLUDE_FAILED_RUNS,
        )
        print(f" found {len(runs)} runs")
        jobs_data = udpate_jobs_data(jobs_data, runs, auth)

    df = pd.DataFrame(jobs_data)
    df.to_csv(OUTPUT_FILE, sep="\t", index=False)

    df = pd.read_csv(
        OUTPUT_FILE,
        sep="\t",
        parse_dates=["started_at", "completed_at"],
    )
    df["duration"] = df["completed_at"] - df["started_at"]
    print(df)

    fig = px.line(df, x="started_at", y="duration", color="name")
    fig.show()


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
    response = requests.get(url, auth=auth)
    if response.status_code != 200:
        print(response.status_code)
        print(response.json())
        return {}
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

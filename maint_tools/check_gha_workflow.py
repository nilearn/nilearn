"""foo."""
import warnings
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
from rich import print

user = "nilearn"
repo = "nilearn"

token_file = Path("/home/remi/Documents/tokens/gh_read_repo_for_orga.txt")
username = "Remi-Gau"

# can be found out at
# "https://api.github.com/repos/{user}/{repo}/actions/workflows"
workflow_id = "71549417"

output_file = Path(__file__).parent / "test_runs_timing.tsv"


def get_auth(username, token_file):
    """Get authentication with token."""
    token = None
    if token_file.exists():
        with open(token_file) as f:
            token = f.read().strip()
    else:
        warnings.warn(f"Token file not found.\n{str(token_file)}")
    return None if username is None or token is None else (username, token)


def get_runs(workflow_id, auth=None, page=1):
    """Get list of runs for a workflow."""
    source = f"https://api.github.com/repos/{user}/{repo}/actions/workflows"
    query = f"?per_page=100&status=completed&branch=main&page={page}"
    url = f"{source}/{workflow_id}/runs{query}"

    response = requests.get(url, auth=auth)
    if response.status_code != 200:
        print(response.status_code)
        print(response.json())
    content = response.json()

    return [
        i for i in content["workflow_runs"] if i["conclusion"] == "success"
    ]


def udpate_jobs_data(jobs_data, runs, auth=None):
    """Collect info for each job in a run."""
    for run in runs:
        print(f'{run["id"]}: {run["display_title"]}')

        response = requests.get(run["jobs_url"], auth=auth)
        if response.status_code != 200:
            print(response.status_code)
            print(response.json())
        content = response.json()

        for job in content["jobs"]:
            for key in jobs_data:
                jobs_data[key].append(job[key])

    return jobs_data


def main():
    """Collect duration of each test job and plots them."""
    auth = get_auth(username, token_file)

    jobs_data = {"name": [], "started_at": [], "completed_at": []}

    for page in [1, 2, 3]:
        runs = get_runs(workflow_id, auth, page=page)
        jobs_data = udpate_jobs_data(jobs_data, runs, auth, output_file)

    df = pd.DataFrame(jobs_data)
    df.to_csv(output_file, sep="\t", index=False)

    df = pd.read_csv(
        output_file,
        sep="\t",
        parse_dates=["started_at", "completed_at"],
    )
    df["duration"] = df["completed_at"] - df["started_at"]
    print(df)

    fig = px.line(df, x="started_at", y="duration", color="name")
    fig.show()


if __name__ == "__main__":
    main()

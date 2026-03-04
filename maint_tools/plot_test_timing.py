"""Get output from pytest run and plot duration with different grouping."""

from pathlib import Path
from warnings import warn

import pandas as pd
import plotly.express as px


def main():
    """Get output from pytest run and plot duration with different grouping."""
    for file in ["pytest_output.csv", "pytest_output_slow_tests.csv"]:
        input_file = (
            Path(__file__).parents[1] / "results" / "pytest_output" / file
        )
        if not input_file.exists():
            warn(f"{input_file} not found.", stacklevel=2)
        else:
            plot_tests_timing(input_file)


def plot_tests_timing(input_file: Path):
    """Plot timing of tests based on pytest_csv output."""
    tests_data = pd.read_csv(input_file)

    tests_data["subpackage"] = tests_data["module"].str.split(
        ".", expand=True
    )[1]

    tests_data["id_no_param"] = tests_data["id"].str.split("[", expand=True)[0]

    for column, title in zip(
        ["subpackage", "module", "id_no_param", "id"],
        ["Subpackage", "Module", "Test", "Parametrization"],
        strict=False,
    ):
        durations = tests_data.groupby(column)["duration"].sum().reset_index()
        durations = durations.sort_values(by="duration", ascending=True)
        fig = px.bar(
            durations,
            x="duration",
            y=column,
            orientation="h",
            title=f"{title} duration",
            labels={"duration": "Total Duration (s)", column: title},
        )
        output_file = (
            input_file.parent / f"{input_file.stem}_{title.lower()}"
        ).with_suffix(".html")
        fig.write_html(output_file)
        print(f"File saved at: {output_file}")


if __name__ == "__main__":
    main()

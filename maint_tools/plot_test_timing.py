"""Get output from pytest run and plot duration with different grouping."""

from pathlib import Path

import pandas as pd
import plotly.express as px

input_file = (
    Path(__file__).parents[1]
    / "results"
    / "pytest_output"
    / "pytest_output.csv"
)

tests_data = pd.read_csv(input_file)

tests_data["subpackage"] = tests_data["module"].apply(
    lambda x: x.split(".")[1]
)

# get name of test without any parametrization
# from
#   test_resampling_result_axis_permutation[axis_permutation3-False]
# to
#   test_resampling_result_axis_permutation
tests_data["id_no_param"] = tests_data["id"].apply(lambda x: x.split("[")[0])

for column, title in zip(
    ["subpackage", "module", "id_no_param", "id"],
    ["Subpackage", "Module", "Test", "Parametrization"],
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

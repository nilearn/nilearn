# /// script
# requires-python = ">=3.8"
# dependencies = [
#    "pandas",
#    "plotly[express]",
#    "kaleido @ file:///${PROJECT_ROOT}/",
#    ]
# ///
"""Runs the examples in the documentation. Use `mkdir output/; uv run`."""

import asyncio

import plotly.express as px

import kaleido

### SAMPLE DATA ###

fig = px.scatter(
    px.data.iris(),
    x="sepal_length",
    y="sepal_width",
    color="species",
)

fig2 = px.line(
    px.data.gapminder().query("country=='Canada'"),
    x="year",
    y="lifeExp",
    title="Life expectancy in Canada",
)

figures = [fig, fig2]

### WRITE FIGURES ###

# Simple one image synchronous write

kaleido.write_fig_sync(fig, path="./output/")


# Multiple image write with error collection

error_log = []

kaleido.write_fig_sync(
    figures,
    path="./output/",
    opts={"format": "jpg"},
    error_log=error_log,
)

# Dump the error_log

if error_log:
    for e in error_log:
        print(str(e))  # noqa: T201
    raise RuntimeError("{len(error_log)} images failed.")


# async/await style of above

asyncio.run(
    kaleido.write_fig(
        figures,
        path="./output/",
        opts={"format": "jpg"},
        error_log=error_log,
    ),
)

### Make a figure generator


def generate_figures():  # can be async as well
    """Generate plotly figures for each country in gapminder."""
    data = px.data.gapminder()
    for country in data["country"].unique():  # list all countries in dataset
        # yield unique plot for each country
        yield px.line(
            data.query(f'country=="{country}"'),
            x="year",
            y="lifeExp",
            title=f"Life expectancy in {country}",
        )


kaleido.write_fig_sync(generate_figures(), path="./output/", n=15)
# file names will be taken from figure title


### If you need more control, use an object


def generate_figure_objects():
    """Generate plotly figure objects for each country in gapminder."""
    data = px.data.gapminder()
    for country in data["country"].unique():  # list all countries in dataset
        fig = px.line(
            data.query(f'country=="{country}"'),
            x="year",
            y="lifeExp",
            title=f"Life expectancy in {country}",
        )
        yield {"fig": fig, "path": f"./output/{country}.jpg"}
        # customize file name


# use 15 processes
kaleido.write_fig_from_object_sync(generate_figure_objects(), n=15)

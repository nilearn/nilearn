import asyncio
import re
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, Phase, given, settings
from hypothesis import strategies as st

from kaleido import Kaleido


@pytest.fixture
async def simple_figure_with_bytes():
    """Create a simple figure with calculated bytes and PNG assertion."""
    import plotly.express as px  # noqa: PLC0415

    fig = px.line(x=[1, 2, 3], y=[1, 2, 3])

    async with Kaleido() as k:
        bytes_data = await k.calc_fig(
            fig,
            opts={"format": "png", "width": 400, "height": 300},
        )

    # Assert it's a PNG by checking the PNG signature
    assert bytes_data[:8] == b"\x89PNG\r\n\x1a\n", "Generated data is not a valid PNG"

    return {
        "fig": fig,
        "bytes": bytes_data,
        "opts": {"format": "png", "width": 400, "height": 300},
    }


async def test_write_fig_from_object_sync_generator(simple_figure_with_bytes, tmp_path):
    """Test write_fig_from_object with sync generator."""

    file_paths = []

    def fig_generator():
        for i in range(2):
            path = tmp_path / f"test_sync_{i}.png"
            file_paths.append(path)
            yield {
                "fig": simple_figure_with_bytes["fig"],
                "path": path,
                "opts": simple_figure_with_bytes["opts"],
            }

    async with Kaleido() as k:
        await k.write_fig_from_object(fig_generator())

    # Assert that each created file matches the fixture bytes
    for path in file_paths:
        assert path.exists(), f"File {path} was not created"
        created_bytes = path.read_bytes()
        assert created_bytes == simple_figure_with_bytes["bytes"], (
            f"File {path} bytes don't match fixture bytes"
        )


async def test_write_fig_from_object_async_generator(
    simple_figure_with_bytes,
    tmp_path,
):
    """Test write_fig_from_object with async generator."""

    file_paths = []

    async def fig_async_generator():
        for i in range(2):
            path = tmp_path / f"test_async_{i}.png"
            file_paths.append(path)
            yield {
                "fig": simple_figure_with_bytes["fig"],
                "path": path,
                "opts": simple_figure_with_bytes["opts"],
            }

    async with Kaleido() as k:
        await k.write_fig_from_object(fig_async_generator())

    # Assert that each created file matches the fixture bytes
    for path in file_paths:
        assert path.exists(), f"File {path} was not created"
        created_bytes = path.read_bytes()
        assert created_bytes == simple_figure_with_bytes["bytes"], (
            f"File {path} bytes don't match fixture bytes"
        )


async def test_write_fig_from_object_iterator(simple_figure_with_bytes, tmp_path):
    """Test write_fig_from_object with iterator."""

    fig_list = []
    file_paths = []
    for i in range(2):
        path = tmp_path / f"test_iter_{i}.png"
        file_paths.append(path)
        fig_list.append(
            {
                "fig": simple_figure_with_bytes["fig"],
                "path": path,
                "opts": simple_figure_with_bytes["opts"],
            },
        )

    async with Kaleido() as k:
        await k.write_fig_from_object(fig_list)

    # Assert that each created file matches the fixture bytes
    for path in file_paths:
        assert path.exists(), f"File {path} was not created"
        created_bytes = path.read_bytes()
        assert created_bytes == simple_figure_with_bytes["bytes"], (
            f"File {path} bytes don't match fixture bytes"
        )


async def test_write_fig_from_object_bare_dictionary(
    simple_figure_with_bytes,
    tmp_path,
):
    """Test write_fig_from_object with bare dictionary list."""

    path1 = tmp_path / "test_dict_1.png"
    path2 = tmp_path / "test_dict_2.png"

    fig_data = [
        {
            "fig": simple_figure_with_bytes["fig"],
            "path": path1,
            "opts": simple_figure_with_bytes["opts"],
        },
        {
            "fig": simple_figure_with_bytes["fig"].to_dict(),
            "path": path2,
            "opts": simple_figure_with_bytes["opts"],
        },
    ]

    async with Kaleido() as k:
        await k.write_fig_from_object(fig_data)

    # Assert that each created file matches the fixture bytes
    for path in [path1, path2]:
        assert path.exists(), f"File {path} was not created"
        created_bytes = path.read_bytes()
        assert created_bytes == simple_figure_with_bytes["bytes"], (
            f"File {path} bytes don't match fixture bytes"
        )


# In the refactor, all figure generation methods are really just wrappers
# for the most flexible, tested above, generate_fig_from_object.
# So we test that one, and then test to make sure its receiving arguments
# properly for the other tests.


# Uncomment these settings after refactor.
# @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@settings(
    phases=[Phase.generate],
    max_examples=1,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    path=st.text(
        min_size=1,
        max_size=50,
        alphabet=st.characters(whitelist_categories=["L", "N"]),
    ),
    width=st.integers(min_value=100, max_value=2000),
    height=st.integers(min_value=100, max_value=2000),
    format_type=st.sampled_from(["png", "svg", "pdf", "html"]),
    topojson=st.one_of(st.none(), st.text(min_size=1, max_size=20)),
)
async def test_write_fig_argument_passthrough(  #  noqa: PLR0913
    simple_figure_with_bytes,
    tmp_path,
    path,
    width,
    height,
    format_type,
    topojson,
):
    """Test that write_fig properly passes arguments to write_fig_from_object."""
    pytest.skip("Remove this failure line and the comment above after the refactor!")
    test_path = tmp_path / f"{path}.{format_type}"
    opts = {"format": format_type, "width": width, "height": height}

    # Mock write_fig_from_object to capture arguments
    with patch.object(Kaleido, "write_fig_from_object") as mock_write_fig_from_object:
        async with Kaleido() as k:
            await k.write_fig(
                simple_figure_with_bytes["fig"],
                path=test_path,
                opts=opts,
                topojson=topojson,
            )

        # Verify write_fig_from_object was called
        mock_write_fig_from_object.assert_called_once()

        # Extract the generator that was passed as first argument
        args, _kwargs = mock_write_fig_from_object.call_args  # not sure.
        assert len(args) == 1, "Expected exactly one argument (the generator)"

        generator = args[0]

        # Convert generator to list to inspect its contents
        generated_args_list = list(generator)
        assert len(generated_args_list) == 1, (
            "Expected generator to yield exactly one item"
        )

        generated_args = generated_args_list[0]

        # Validate that the generated arguments match what we passed to write_fig
        assert "fig" in generated_args, "Generated args should contain 'fig'"
        assert "path" in generated_args, "Generated args should contain 'path'"
        assert "opts" in generated_args, "Generated args should contain 'opts'"
        assert "topojson" in generated_args, "Generated args should contain 'topojson'"

        # Check that the values match
        assert generated_args["fig"] == simple_figure_with_bytes["fig"], (
            "Figure should match"
        )
        assert str(generated_args["path"]) == str(test_path), "Path should match"
        assert generated_args["opts"] == opts, "Options should match"
        assert generated_args["topojson"] == topojson, "Topojson should match"


async def test_kaleido_instantiate_no_hang():
    """Test that instantiating Kaleido doesn't hang."""
    _ = Kaleido()


async def test_kaleido_instantiate_and_close():
    """Test that instantiating and closing Kaleido works."""
    # Maybe there should be a warning or error when closing without opening?
    k = Kaleido()
    await k.close()


async def test_all_methods_context(simple_figure_with_bytes, tmp_path):
    """Test write, write_from_object, and calc with context."""
    fig = simple_figure_with_bytes["fig"]
    opts = simple_figure_with_bytes["opts"]
    expected_bytes = simple_figure_with_bytes["bytes"]

    # Test with context manager
    async with Kaleido() as k:
        # Test calc_fig
        calc_bytes = await k.calc_fig(fig, opts=opts)
        assert calc_bytes == expected_bytes, "calc_fig bytes don't match fixture"

        # Test write_fig
        write_path = tmp_path / "context_write.png"
        await k.write_fig(fig, path=write_path, opts=opts)
        assert write_path.exists(), "write_fig didn't create file"
        write_bytes = write_path.read_bytes()
        assert write_bytes == expected_bytes, "write_fig bytes don't match fixture"

        # Test write_fig_from_object
        obj_path = tmp_path / "context_obj.png"
        await k.write_fig_from_object([{"fig": fig, "path": obj_path, "opts": opts}])
        assert obj_path.exists(), "write_fig_from_object didn't create file"
        obj_bytes = obj_path.read_bytes()
        assert obj_bytes == expected_bytes, (
            "write_fig_from_object bytes don't match fixture"
        )


async def test_all_methods_non_context(simple_figure_with_bytes, tmp_path):
    """Test write, write_from_object, and calc with non-context."""
    fig = simple_figure_with_bytes["fig"]
    opts = simple_figure_with_bytes["opts"]
    expected_bytes = simple_figure_with_bytes["bytes"]

    # Test without context manager
    k = await Kaleido()
    try:
        # Test calc_fig
        calc_bytes = await k.calc_fig(fig, opts=opts)
        assert calc_bytes == expected_bytes, (
            "Non-context calc_fig bytes don't match fixture"
        )

        # Test write_fig

        write_path2 = tmp_path / "non_context_write.png"
        await k.write_fig(fig, path=write_path2, opts=opts)

        assert write_path2.exists(), "Non-context write_fig didn't create file"
        write_bytes2 = write_path2.read_bytes()
        assert write_bytes2 == expected_bytes, (
            "Non-context write_fig bytes don't match fixture"
        )

        obj_path2 = tmp_path / "non_context_obj.png"
        await k.write_fig_from_object([{"fig": fig, "path": obj_path2, "opts": opts}])
        assert obj_path2.exists(), (
            "Non-context write_fig_from_object didn't create file"
        )
        obj_bytes2 = obj_path2.read_bytes()
        assert obj_bytes2 == expected_bytes, (
            "Non-context write_fig_from_object bytes don't match fixture"
        )

    finally:
        await k.close()


@pytest.mark.parametrize("n_tabs", [1, 2, 3])
async def test_tab_count_verification(n_tabs):
    """Test that Kaleido creates the correct number of tabs."""
    async with Kaleido(n=n_tabs) as k:
        # Check the queue size matches expected tabs
        assert k.tabs_ready.qsize() == n_tabs, (
            f"Queue size {k.tabs_ready.qsize()} != {n_tabs}"
        )

        # Use devtools protocol to verify tab count
        # Send getTargets command directly to Kaleido (which is a Browser/Target)
        result = await k.send_command("Target.getTargets")
        # Count targets that are pages (not service workers, etc.)
        page_targets = [
            t for t in result["result"]["targetInfos"] if t.get("type") == "page"
        ]
        assert len(page_targets) >= n_tabs, (
            f"Found {len(page_targets)} page targets, expected at least {n_tabs}"
        )


async def test_unreasonable_timeout(simple_figure_with_bytes):
    """Test that an unreasonably small timeout actually times out."""

    fig = simple_figure_with_bytes["fig"]
    opts = simple_figure_with_bytes["opts"]

    # Use an infinitely small timeout
    async with Kaleido(timeout=0.000001) as k:
        with pytest.raises((asyncio.TimeoutError, TimeoutError)):
            await k.calc_fig(fig, opts=opts)


@pytest.mark.parametrize(
    ("plotlyjs", "mathjax"),
    [
        ("https://cdn.plot.ly/plotly-latest.min.js", None),
        (
            None,
            "https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.0/es5/tex-chtml.min.js",
        ),
        (
            "https://cdn.plot.ly/plotly-latest.min.js",
            "https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.0/es5/tex-chtml.min.js",
        ),
    ],
)  # THESE STRINGS DON'T ACTUALLY MATTER!
async def test_plotlyjs_mathjax_injection(plotlyjs, mathjax):
    """Test that plotlyjs and mathjax URLs are properly injected."""

    async with Kaleido(plotlyjs=plotlyjs, mathjax=mathjax) as k:
        # Get a tab from the public queue to check the page source
        tab = await k.tabs_ready.get()
        try:
            # Get the page source using devtools protocol
            result = await tab.tab.send_command(
                "Runtime.evaluate",
                {
                    "expression": "document.documentElement.outerHTML",
                },
            )
            source = result["result"]["result"]["value"]

            if plotlyjs:
                # Check if plotlyjs URL is in the source
                plotly_pattern = re.escape(plotlyjs)
                assert re.search(plotly_pattern, source), (
                    f"Plotlyjs URL {plotlyjs} not found in page source"
                )

            if mathjax:
                # Check if mathjax URL is in the source
                mathjax_pattern = re.escape(mathjax)
                assert re.search(mathjax_pattern, source), (
                    f"Mathjax URL {mathjax} not found in page source"
                )

        finally:
            # Put the tab back in the queue
            await k.tabs_ready.put(tab)

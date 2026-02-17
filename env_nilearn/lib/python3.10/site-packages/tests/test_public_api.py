"""Integrative tests for all public API functions in __init__.py using basic figures."""

from pathlib import Path
from unittest.mock import patch

import pytest

import kaleido


@pytest.fixture(params=["figure", "dict"])
def simple_figure(request):
    """Create a simple plotly figure for testing, either as figure or dict."""
    # ruff: noqa: PLC0415
    import plotly.express as px

    fig = px.line(x=[1, 2, 3, 4], y=[1, 2, 3, 4])

    if request.param == "dict":
        return fig.to_dict()
    return fig


async def test_async_api_functions(simple_figure, tmp_path):
    """Test calc_fig, write_fig, and write_fig_from_object with cross-validation."""
    # Test calc_fig and get reference bytes
    calc_result = await kaleido.calc_fig(simple_figure)
    assert isinstance(calc_result, bytes)
    assert calc_result.startswith(b"\x89PNG\r\n\x1a\n"), "Not a PNG file"

    # Test write_fig and compare with calc_fig output
    write_fig_output = tmp_path / "test_write_fig.png"
    await kaleido.write_fig(simple_figure, path=str(write_fig_output))

    with Path(write_fig_output).open("rb") as f:  # noqa: ASYNC230
        write_fig_bytes = f.read()

    assert write_fig_bytes == calc_result
    assert write_fig_bytes.startswith(b"\x89PNG\r\n\x1a\n"), "Not a PNG file"

    # Test write_fig_from_object and compare with calc_fig output
    write_fig_from_object_output = tmp_path / "test_write_fig_from_object.png"
    await kaleido.write_fig_from_object(
        [
            {
                "fig": simple_figure,
                "path": write_fig_from_object_output,
            },
        ],
    )

    with Path(write_fig_from_object_output).open("rb") as f:  # noqa: ASYNC230
        write_fig_from_object_bytes = f.read()

    assert write_fig_from_object_bytes == calc_result
    assert write_fig_from_object_bytes.startswith(
        b"\x89PNG\r\n\x1a\n",
    ), "Not a PNG file"

    # Cross-validate all results are identical
    assert write_fig_bytes == write_fig_from_object_bytes == calc_result


async def test_sync_api_functions(simple_figure, tmp_path):
    """Test sync wrappers with cross-validation."""
    # Get expected bytes from calc_fig for comparison
    expected_bytes = await kaleido.calc_fig(simple_figure)
    assert isinstance(expected_bytes, bytes)
    assert expected_bytes.startswith(b"\x89PNG\r\n\x1a\n"), "Not a PNG file"

    # Test scenario 1: server running
    write_fig_output_1 = tmp_path / "test_write_fig_server_running.png"
    write_fig_from_object_output_1 = tmp_path / "test_from_object_server_running.png"

    with patch(
        "kaleido._sync_server.oneshot_async_run",
        wraps=kaleido._sync_server.oneshot_async_run,  # noqa: SLF001 internal
    ) as mock_oneshot, patch(
        "kaleido._global_server.call_function",
        wraps=kaleido._global_server.call_function,  # noqa: SLF001 internal
    ) as mock_call:
        kaleido.start_sync_server(silence_warnings=True)
        try:
            # Test calc_fig_sync
            calc_result_1 = kaleido.calc_fig_sync(simple_figure)
            assert isinstance(calc_result_1, bytes)
            assert calc_result_1.startswith(b"\x89PNG\r\n\x1a\n"), "Not a PNG file"
            assert calc_result_1 == expected_bytes

            # Test write_fig_sync
            kaleido.write_fig_sync(simple_figure, path=str(write_fig_output_1))

            with Path(write_fig_output_1).open("rb") as f:  # noqa: ASYNC230
                write_fig_bytes_1 = f.read()
            assert write_fig_bytes_1 == expected_bytes
            assert write_fig_bytes_1.startswith(b"\x89PNG\r\n\x1a\n"), "Not a PNG file"

            # Test write_fig_from_object_sync
            kaleido.write_fig_from_object_sync(
                [
                    {
                        "fig": simple_figure,
                        "path": write_fig_from_object_output_1,
                    },
                ],
            )

            with Path(write_fig_from_object_output_1).open("rb") as f:  # noqa: ASYNC230
                from_object_bytes_1 = f.read()
            assert from_object_bytes_1 == expected_bytes
            assert from_object_bytes_1.startswith(
                b"\x89PNG\r\n\x1a\n",
            ), "Not a PNG file"

            # Should have been called three times (once for each function)
            assert mock_call.call_count == 3  # noqa: PLR2004
            assert mock_oneshot.call_count == 0

            # Cross-validate all server running results are identical
            assert (
                calc_result_1
                == write_fig_bytes_1
                == from_object_bytes_1
                == expected_bytes
            )

        finally:
            kaleido.stop_sync_server(silence_warnings=True)

        # Test scenario 2: server not running
        write_fig_output_2 = tmp_path / "test_write_fig_server_not_running.png"
        write_fig_from_object_output_2 = (
            tmp_path / "test_from_object_server_not_running.png"
        )

        # Test calc_fig_sync
        calc_result_2 = kaleido.calc_fig_sync(simple_figure)
        assert isinstance(calc_result_2, bytes)
        assert calc_result_2.startswith(b"\x89PNG\r\n\x1a\n"), "Not a PNG file"
        assert calc_result_2 == expected_bytes

        # Test write_fig_sync
        kaleido.write_fig_sync(simple_figure, path=str(write_fig_output_2))

        with Path(write_fig_output_2).open("rb") as f:  # noqa: ASYNC230
            write_fig_bytes_2 = f.read()
        assert write_fig_bytes_2 == expected_bytes
        assert write_fig_bytes_2.startswith(b"\x89PNG\r\n\x1a\n"), "Not a PNG file"

        # Test write_fig_from_object_sync
        kaleido.write_fig_from_object_sync(
            [
                {
                    "fig": simple_figure,
                    "path": write_fig_from_object_output_2,
                },
            ],
        )

        with Path(write_fig_from_object_output_2).open("rb") as f:  # noqa: ASYNC230
            from_object_bytes_2 = f.read()
        assert from_object_bytes_2 == expected_bytes
        assert from_object_bytes_2.startswith(b"\x89PNG\r\n\x1a\n"), "Not a PNG file"

        # Should have been called three times (once for each function)
        assert mock_call.call_count == 3  # noqa: PLR2004
        assert mock_oneshot.call_count == 3  # noqa: PLR2004

        # Cross-validate all server not running results are identical
        assert (
            calc_result_2 == write_fig_bytes_2 == from_object_bytes_2 == expected_bytes
        )


def test_start_stop_sync_server_integration():
    """Test start_sync_server and stop_sync_server with warning behavior."""
    # Test starting and stopping with warnings silenced
    kaleido.start_sync_server(silence_warnings=False)

    # Test starting already started server - should warn
    with pytest.warns(RuntimeWarning, match="already"):
        kaleido.start_sync_server(silence_warnings=False)

    kaleido.start_sync_server(silence_warnings=True)

    kaleido.stop_sync_server(silence_warnings=False)

    # Test stopping already stopped server - should warn
    with pytest.warns(RuntimeWarning, match="closed"):
        kaleido.stop_sync_server(silence_warnings=False)

    kaleido.stop_sync_server(silence_warnings=True)

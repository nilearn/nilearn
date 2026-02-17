"""Tests for wrapper functions in __init__.py that test argument passing."""

from unittest.mock import AsyncMock, patch

import pytest

import kaleido

# Pretty complicated for basically testing a bunch of wrappers, but it works.
# Integration tests seem more important.
# I much prefer the public_api file, this set of tests can be considered
# for deletion.


@pytest.fixture
def args():
    """Basic args for sync wrapper tests."""
    return ({"data": []}, "test.png")


@pytest.fixture
def kwargs():
    """Basic kwargs for sync wrapper tests."""
    return {"width": 800}


@patch("kaleido._sync_server.GlobalKaleidoServer.open")
def test_start_sync_server_passes_args(mock_open, args, kwargs):
    """Test that start_sync_server passes args and silence_warnings correctly."""
    # Test with silence_warnings=False (default)
    kaleido.start_sync_server(*args, **kwargs)
    mock_open.assert_called_with(*args, silence_warnings=False, **kwargs)

    # Reset mock and test with silence_warnings=True
    mock_open.reset_mock()
    kaleido.start_sync_server(*args, silence_warnings=True, **kwargs)
    mock_open.assert_called_with(*args, silence_warnings=True, **kwargs)


@patch("kaleido._sync_server.GlobalKaleidoServer.close")
def test_stop_sync_server_passes_args(mock_close):
    """Test that stop_sync_server passes silence_warnings correctly."""
    # Test with silence_warnings=False (default)
    kaleido.stop_sync_server()
    mock_close.assert_called_with(silence_warnings=False)

    # Reset mock and test with silence_warnings=True
    mock_close.reset_mock()
    kaleido.stop_sync_server(silence_warnings=True)
    mock_close.assert_called_with(silence_warnings=True)


@patch("kaleido.Kaleido")
async def test_async_wrapper_functions(mock_kaleido_class):
    """Test all async wrapper functions pass arguments correctly.

    Note: This test uses fixed args rather than fixtures due to specific
    requirements with topojson and kopts that don't match the simple fixture pattern.
    """
    # Create a mock that doesn't need the context fixture
    mock_kaleido_class.return_value = mock_kaleido = AsyncMock()
    mock_kaleido.__aenter__.return_value = mock_kaleido
    mock_kaleido.__aexit__.return_value = None
    mock_kaleido.calc_fig.return_value = b"test_bytes"

    fig = {"data": []}

    # Test calc_fig with full arguments and kopts forcing n=1
    path = "test.png"
    opts = {"width": 800}
    topojson = "test_topojson"
    kopts = {"some_option": "value"}

    result = await kaleido.calc_fig(fig, path, opts, topojson=topojson, kopts=kopts)

    expected_kopts = {"some_option": "value", "n": 1}
    mock_kaleido_class.assert_called_with(**expected_kopts)
    mock_kaleido.calc_fig.assert_called_with(
        fig,
        path=path,
        opts=opts,
        topojson=topojson,
    )
    assert result == b"test_bytes"

    # Reset mocks
    mock_kaleido_class.reset_mock()
    mock_kaleido.calc_fig.reset_mock()

    # Test calc_fig with empty kopts
    await kaleido.calc_fig(fig)
    mock_kaleido_class.assert_called_with(n=1)

    # Reset mocks
    mock_kaleido_class.reset_mock()
    mock_kaleido.write_fig.reset_mock()

    # Test write_fig with full arguments
    await kaleido.write_fig(fig, path, opts, topojson=topojson, kopts=kopts)
    mock_kaleido_class.assert_called_with(**kopts)  # write_fig doesn't force n=1
    mock_kaleido.write_fig.assert_called_with(
        fig,
        path=path,
        opts=opts,
        topojson=topojson,
    )

    # Reset mocks
    mock_kaleido_class.reset_mock()
    mock_kaleido.write_fig.reset_mock()

    # Test write_fig with empty kopts
    await kaleido.write_fig(fig)
    mock_kaleido_class.assert_called_with()

    # Reset mocks
    mock_kaleido_class.reset_mock()
    mock_kaleido.write_fig_from_object.reset_mock()

    # Test write_fig_from_object
    generator = [{"data": []}]
    await kaleido.write_fig_from_object(generator, kopts=kopts)
    mock_kaleido_class.assert_called_with(**kopts)
    mock_kaleido.write_fig_from_object.assert_called_with(generator)


@patch("kaleido._sync_server.GlobalKaleidoServer.is_running")
@patch("kaleido._sync_server.GlobalKaleidoServer.call_function")
def test_sync_wrapper_server(mock_call_function, mock_is_running, args, kwargs):
    """Test all sync wrapper functions when global server is running."""
    mock_is_running.return_value = True

    # Test calc_fig_sync
    kaleido.calc_fig_sync(*args, **kwargs)
    mock_call_function.assert_called_with("calc_fig", *args, **kwargs)

    mock_call_function.reset_mock()

    # Test write_fig_sync
    kaleido.write_fig_sync(*args, **kwargs)
    mock_call_function.assert_called_with("write_fig", *args, **kwargs)

    mock_call_function.reset_mock()

    # Test write_fig_from_object_sync
    kaleido.write_fig_from_object_sync(*args, **kwargs)
    mock_call_function.assert_called_with("write_fig_from_object", *args, **kwargs)


@patch("kaleido._sync_server.GlobalKaleidoServer.is_running")
@patch("kaleido._sync_server.oneshot_async_run")
def test_sync_wrapper_oneshot(mock_oneshot_run, mock_is_running, args, kwargs):
    """Test all sync wrapper functions when no server is running."""
    mock_is_running.return_value = False

    # Test calc_fig_sync
    kaleido.calc_fig_sync(*args, **kwargs)
    mock_oneshot_run.assert_called_with(kaleido.calc_fig, args=args, kwargs=kwargs)

    mock_oneshot_run.reset_mock()

    # Test write_fig_sync
    kaleido.write_fig_sync(*args, **kwargs)
    mock_oneshot_run.assert_called_with(kaleido.write_fig, args=args, kwargs=kwargs)

    mock_oneshot_run.reset_mock()

    # Test write_fig_from_object_sync
    kaleido.write_fig_from_object_sync(*args, **kwargs)
    mock_oneshot_run.assert_called_with(
        kaleido.write_fig_from_object,
        args=args,
        kwargs=kwargs,
    )

import pytest

from kaleido._sync_server import GlobalKaleidoServer


class TestGlobalKaleidoServer:
    """Test the GlobalKaleidoServer singleton class."""

    def test_singleton_behavior(self):
        """Test that creating the object twice returns the same instance."""
        # Should be the same object
        assert GlobalKaleidoServer() is GlobalKaleidoServer()

    def test_is_running_open_close_cycle(self):
        """Test is_running, open, and close in a loop three times."""
        server = GlobalKaleidoServer()

        # Initial state should be not running
        assert not server.is_running()

        for i in range(2):
            # Check not running
            assert not server.is_running(), (
                f"Iteration {i}: Should not be running initially"
            )

            server.open()

            # Check is running
            assert server.is_running(), f"Iteration {i}: Should be running after open"

            # Call open again - should warn
            with pytest.warns(RuntimeWarning, match="Server already open"):
                server.open()

            server.open(silence_warnings=True)
            server.close()

            # Call close again - should warn
            with pytest.warns(RuntimeWarning, match="Server already closed"):
                server.close()

            server.close(silence_warnings=True)

            # Check not running
            assert not server.is_running(), (
                f"Iteration {i}: Should not be running after close"
            )

    def test_call_function_when_not_running_raises_error(self):
        """Test that calling function when server is not running raises RuntimeError."""
        server = GlobalKaleidoServer()

        # Ensure server is closed
        if server.is_running():
            server.close(silence_warnings=True)

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Can't call function on stopped server"):
            server.call_function("some_function")

    def test_getattr_call_function_integration(self):
        """Test __getattr__ integration with call_function."""
        server = GlobalKaleidoServer()
        method_name = "random_method"
        test_args = ("arg1", "arg2")
        test_kwargs = {"kwarg1": "value1"}

        def method_checker(_self, name):
            assert name == method_name

            def dummy_method(*args, **kwargs):
                assert args == test_args
                assert kwargs == test_kwargs

            return dummy_method

        # Temporarily add __getattr__ to the class
        GlobalKaleidoServer.__getattr__ = method_checker

        try:
            # Call a random method with some args and kwargs
            server.random_method(*test_args, **test_kwargs)

        finally:
            # Clean up - remove __getattr__ from the class
            delattr(GlobalKaleidoServer, "__getattr__")

    def teardown_method(self):
        """Clean up after each test."""
        server = GlobalKaleidoServer()
        if server.is_running():
            server.close(silence_warnings=True)

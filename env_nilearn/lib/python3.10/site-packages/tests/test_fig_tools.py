from pathlib import Path

import pytest

from kaleido import _fig_tools

sources = ["argument", "layout", "template", "default"]
values = [None, 150, 800, 1500]


@pytest.mark.parametrize("width_source", sources)
@pytest.mark.parametrize("height_source", sources)
@pytest.mark.parametrize("width_value", values)
@pytest.mark.parametrize("height_value", [x * 1.5 if x else x for x in values])
def test_get_figure_dimensions(width_source, height_source, width_value, height_value):
    """Test _get_figure_dimensions with all combinations of width/height sources."""

    layout = {}
    width_arg = None
    expected_width = width_value

    if width_source == "argument":
        width_arg = width_value
    elif width_source == "layout":
        layout["width"] = width_value
    elif width_source == "template":
        layout.setdefault("template", {}).setdefault("layout", {})["width"] = (
            width_value
        )
    else:  # default
        expected_width = None

    # Set to default if None
    if expected_width is None:
        expected_width = _fig_tools.DEFAULT_WIDTH

    # Do for height what I did for width
    height_arg = None
    expected_height = height_value

    if height_source == "argument":
        height_arg = height_value
    elif height_source == "layout":
        layout["height"] = height_value
    elif height_source == "template":
        layout.setdefault("template", {}).setdefault("layout", {})["height"] = (
            height_value
        )
    else:  # default
        expected_height = None

    # Set to default if None
    if expected_height is None:
        expected_height = _fig_tools.DEFAULT_HEIGHT

    # Call the function
    r_width, r_height = _fig_tools._get_figure_dimensions(  # noqa: SLF001
        layout,
        width_arg,
        height_arg,
    )

    # Assert results
    assert r_width == expected_width, (
        f"Width mismatch: got {r_width}, expected {expected_width}, "
        f"source: {width_source}, value: {width_value}"
    )
    assert r_height == expected_height, (
        f"Height mismatch: got {r_height}, expected {expected_height}, "
        f"source: {height_source}, value: {height_value}"
    )


def test_next_filename_no_existing_files(tmp_path):
    """Test _next_filename when no files exist."""
    result = _fig_tools._next_filename(tmp_path, "test", "png")  # noqa: SLF001
    assert result == "test.png"


def test_next_filename_base_file_exists(tmp_path):
    """Test _next_filename when base file exists."""
    # Create the base file
    (tmp_path / "test.png").touch()

    result = _fig_tools._next_filename(tmp_path, "test", "png")  # noqa: SLF001
    assert result == "test-2.png"


def test_next_filename_numbered_files_exist(tmp_path):
    """Test _next_filename when numbered files exist."""
    # Create various numbered files
    (tmp_path / "test.png").touch()
    (tmp_path / "test-2.png").touch()
    (tmp_path / "test-3.png").touch()
    (tmp_path / "test-5.png").touch()  # Gap in numbering

    result = _fig_tools._next_filename(tmp_path, "test", "png")  # noqa: SLF001
    assert result == "test-6.png"  # Should be max + 1


def test_next_filename_similar_names_ignored(tmp_path):
    """Test _next_filename ignores files with similar but different names."""
    # Create files that shouldn't match the pattern
    (tmp_path / "test.png").touch()
    (tmp_path / "test-2.png").touch()
    (tmp_path / "testing-3.png").touch()  # Different prefix
    (tmp_path / "test-2.jpg").touch()  # Different extension
    (tmp_path / "test-abc.png").touch()  # Non-numeric suffix

    result = _fig_tools._next_filename(tmp_path, "test", "png")  # noqa: SLF001
    assert result == "test-3.png"  # Should only count test.png and test-2.png


def test_next_filename_special_characters(tmp_path):
    """Test _next_filename with special characters in prefix and extension."""
    prefix = "test-file_name"
    ext = "svg"  # set up to be parameterized but not

    # Create some files
    (tmp_path / f"{prefix}.{ext}").touch()
    (tmp_path / f"{prefix}-2.{ext}").touch()

    result = _fig_tools._next_filename(tmp_path, prefix, ext)  # noqa: SLF001
    assert result == f"{prefix}-3.{ext}"


def test_next_filename_only_numbered_files(tmp_path):
    """Test _next_filename when only numbered files exist (no base file)."""
    # Create only numbered files, no base file
    (tmp_path / "test-2.png").touch()
    (tmp_path / "test-3.png").touch()
    (tmp_path / "test-10.png").touch()

    result = _fig_tools._next_filename(tmp_path, "test", "png")  # noqa: SLF001
    assert result == "test-11.png"  # Should be max + 1


# Fixtures for _build_full_path tests - testing various title scenarios
@pytest.fixture(
    params=[
        (
            {
                "layout": {
                    "title": {"text": "My-Test!@#$%^&*()Chart_with[lots]of{symbols}"},
                },
            },
            "My_TestChart_withlotsofsymbols",
        ),  # Complex title
        (
            {"layout": {"title": {"text": "Simple Title"}}},
            "Simple_Title",
        ),  # Simple title
        ({"layout": {}}, "fig"),  # No title
    ],
)
def fig_fixture(request):
    """Parameterized fixture for fig with various title scenarios."""
    return request.param


def test_build_full_path_no_path_input(fig_fixture):
    """Test _build_full_path with no path input uses current path."""
    fig_dict, expected_prefix = fig_fixture
    result = _fig_tools._build_full_path(None, fig_dict, "ext")  # noqa: SLF001

    # Should use current directory
    assert result.parent.resolve() == Path().cwd().resolve()
    assert result.parent.is_dir()

    assert result.name == f"{expected_prefix}.ext"


def test_build_full_path_no_suffix_directory(tmp_path, fig_fixture):
    """Test _build_full_path with path having no suffix."""
    fig_dict, expected_prefix = fig_fixture

    # Test directory no suffix
    test_dir = tmp_path
    result = _fig_tools._build_full_path(test_dir, fig_dict, "ext")  # noqa: SLF001

    # Should use provided directory
    assert result.parent == test_dir
    assert result.name == f"{expected_prefix}.ext"

    # Test error
    nonexistent_dir = Path("/nonexistent/directory")
    with pytest.raises(ValueError, match=r"Directory .* not found. Please create it."):
        _fig_tools._build_full_path(nonexistent_dir, fig_dict, "ext")  # noqa: SLF001


def test_build_full_path_directory_with_suffix(tmp_path, fig_fixture):
    """Test _build_full_path with path that is directory even with suffix."""
    fig_dict, expected_prefix = fig_fixture

    # Create a directory with a suffix-like name
    dir_with_suffix = tmp_path / "mydir.png"
    dir_with_suffix.mkdir()

    result = _fig_tools._build_full_path(dir_with_suffix, fig_dict, "ext")  # noqa: SLF001

    # Should treat as directory
    assert result.parent == dir_with_suffix
    assert result.name == f"{expected_prefix}.ext"


def test_build_full_path_file_with_suffix(tmp_path, fig_fixture):
    """Test _build_full_path with file path having suffix."""
    fig_dict, _expected_prefix = fig_fixture

    # Exists
    file_path = tmp_path / "output.png"
    result = _fig_tools._build_full_path(file_path, fig_dict, "ext")  # noqa: SLF001

    # Should return the exact path provided
    assert result == file_path

    # Doesn't exist
    file_path = Path("/nonexistent/directory/output.png")
    with pytest.raises(
        RuntimeError,
        match=r"Cannot reach path .* Are all directories created?",
    ):
        _fig_tools._build_full_path(file_path, fig_dict, "ext")  # noqa: SLF001

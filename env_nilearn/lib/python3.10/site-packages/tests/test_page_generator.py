import re
import sys
import tempfile
from html.parser import HTMLParser
from importlib.util import find_spec
from pathlib import Path

import logistro
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from kaleido import PageGenerator
from kaleido._page_generator import DEFAULT_MATHJAX, DEFAULT_PLOTLY

# allows to create a browser pool for tests
pytestmark = pytest.mark.asyncio(loop_scope="function")

_logger = logistro.getLogger(__name__)


# Expected boilerplate HTML (without script tags with src)
EXPECTED_BOILERPLATE = """<!DOCTYPE html>
<html>
    <head>
        <style id="head-style"></style>
        <title>Kaleido-fier</title>
        <script>
          window.PlotlyConfig = {MathJaxConfig: 'local'}
        </script>
        <script type="text/x-mathjax-config">
          MathJax.Hub.Config({ "SVG": { blacker: 0 }})
        </script>

    </head>
    <body style="{margin: 0; padding: 0;}"><img id="kaleido-image" /></body>
</html>"""


class HTMLAnalyzer(HTMLParser):
    """Extract script tags with src attributes and return HTML without them."""

    def __init__(self):
        super().__init__()
        self.scripts = []
        self.encodings = []
        self.boilerplate = []
        self._in_script = False

    def handle_starttag(self, tag, attrs):
        if tag == "script" and "src" in (attr_dict := dict(attrs)):
            self._in_script = True
            self.scripts.append(attr_dict["src"])
            self.encodings.append(attr_dict.get("charset"))
            return
        self.boilerplate.append(self.get_starttag_text())

    def handle_endtag(self, tag):
        if self._in_script and tag == "script":
            self._in_script = False
            return
        self.boilerplate.append(f"</{tag}>")

    def handle_data(self, data):
        if not self._in_script:
            self.boilerplate.append(data)


def normalize_whitespace(html):
    """Normalize whitespace by collapsing multiple newlines and extra spaces."""
    # Collapse multiple newlines to single newlines
    html = re.sub(r"\n\s*\n", "\n", html)
    # Remove extra whitespace between tags
    html = re.sub(r">\s*<", "><", html)
    return html.strip()


# Create boilerplate reference by parsing expected HTML
_reference_analyzer = HTMLAnalyzer()
_reference_analyzer.feed(EXPECTED_BOILERPLATE)
_REFERENCE_BOILERPLATE = normalize_whitespace("".join(_reference_analyzer.boilerplate))


def get_scripts_from_html(generated_html):
    """
    Parse generated HTML, assert boilerplate matches reference, and return script URLs.

    Returns:
        list: script src URLs found in generated HTML
    """
    analyzer = HTMLAnalyzer()
    analyzer.feed(generated_html)

    generated_boilerplate = normalize_whitespace("".join(analyzer.boilerplate))

    # Assert boilerplate matches with diff on failure
    assert generated_boilerplate == _REFERENCE_BOILERPLATE, (
        f"Boilerplate mismatch:\n"
        f"Expected:\n{_REFERENCE_BOILERPLATE}\n\n"
        f"Got:\n{generated_boilerplate}"
    )

    return analyzer.scripts, analyzer.encodings


# Fixtures for user supplied input scenarios
@pytest.fixture
def temp_js_file():
    """Create a temporary JavaScript file that exists."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
        f.write("console.log('test');")
        temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink()


@pytest.fixture
def existing_file_path():
    """Return path to current test file (guaranteed to exist)."""
    return Path(__file__)


@pytest.fixture
def nonexistent_file_uri():
    """Return path to file that doesn't exist."""
    return "file:///nonexistent/path/file.js"


@pytest.fixture
def nonexistent_file_path():
    """Return path to file that doesn't exist."""
    return Path("/nonexistent/path/file.js")


def st_valid_path(dir_path: Path):
    file_path = dir_path / "foo.foo"
    file_path.touch()
    assert file_path.resolve().exists()
    _valid_file_string = str(file_path.resolve())

    _h_file_str = st.just(_valid_file_string)
    _h_file_path = st.just(Path(_valid_file_string))
    _h_file_uri = st.just(Path(_valid_file_string).as_uri())

    _h_url = st.tuples(
        st.sampled_from(["s", ""]),
        st.text(
            min_size=1,
            max_size=20,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
        ),
    ).map(lambda x: f"http{x[0]}://example.com/{x[1]}.js")

    _h_uri = st.one_of(_h_url, _h_file_str, _h_file_path, _h_file_uri)

    _h_encoding = st.sampled_from(["utf-8", "utf-16", "ascii", "latin1"])

    return st.one_of(_h_uri, st.tuples(_h_uri, _h_encoding))


# Variable length list strategy for 'others' parameter
def st_others_list(dir_path: Path):
    return st.lists(st_valid_path(dir_path), min_size=0, max_size=3)


# Mathjax strategy (includes None, False, True, and path options)
def st_mathjax(dir_path: Path):
    return st.one_of(
        st.none(),
        st.just(False),  #  noqa: FBT003
        st_valid_path(dir_path),
    )


# Test default combinations
@pytest.mark.order(1)
async def test_defaults_no_plotly_available():
    """
    Test defaults when plotly package is not available.

    When we generate_index(), if we don't have plotly in path, we use a CDN.
    """
    _old_path = sys.path
    try:
        sys.path = []
        _plotly_mo = sys.modules.pop("plotly", None)

        # Test no imports (plotly not available)
        no_imports = PageGenerator().generate_index()
        scripts, _encodings = get_scripts_from_html(no_imports)

        # Should have mathjax, plotly default, and kaleido_scopes
        assert len(scripts) == 3  # noqa: PLR2004
        assert scripts[0] == DEFAULT_MATHJAX
        assert scripts[1] == DEFAULT_PLOTLY
        assert scripts[2].endswith("kaleido_scopes.js")
    finally:
        sys.path = _old_path
        if _plotly_mo:
            sys.modules.update({"plotly": _plotly_mo})


async def test_defaults_with_plotly_available():
    """Test defaults when plotly package is available."""
    all_defaults = PageGenerator().generate_index()
    scripts, _encodings = get_scripts_from_html(all_defaults)

    # Should have mathjax, plotly package data, and kaleido_scopes
    assert len(scripts) == 3  # noqa: PLR2004
    assert scripts[0] == DEFAULT_MATHJAX
    assert scripts[1].endswith("package_data/plotly.min.js")
    assert scripts[2].endswith("kaleido_scopes.js")


async def test_force_cdn():
    """Test force_cdn=True forces use of CDN plotly even when plotly is available."""
    # Verify plotly is available first
    if not find_spec("plotly"):
        pytest.skip("Plotly not available - cannot test force_cdn override")

    forced_cdn = PageGenerator(force_cdn=True).generate_index()
    scripts, _encodings = get_scripts_from_html(forced_cdn)

    assert len(scripts) == 3  # noqa: PLR2004
    assert scripts[0] == DEFAULT_MATHJAX
    assert scripts[1] == DEFAULT_PLOTLY
    assert scripts[2].endswith("kaleido_scopes.js")


# Test boolean mathjax functionality
async def test_mathjax_false():
    """Test that mathjax=False disables mathjax."""
    without_mathjax = PageGenerator(mathjax=False).generate_index()
    scripts, _encodings = get_scripts_from_html(without_mathjax)

    assert len(scripts) == 2  # noqa: PLR2004
    assert scripts[0].endswith("package_data/plotly.min.js")
    assert scripts[1].endswith("kaleido_scopes.js")


# Test user overrides
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.data())
async def test_custom_plotly_url(tmp_path, data):
    """Test custom plotly URL override."""
    custom_plotly = data.draw(st_valid_path(tmp_path))
    with_custom = PageGenerator(plotly=custom_plotly).generate_index()
    scripts, encodings = get_scripts_from_html(with_custom)

    assert len(scripts) == 3  # noqa: PLR2004
    assert scripts[0] == DEFAULT_MATHJAX
    if isinstance(custom_plotly, tuple):
        assert scripts[1] == str(custom_plotly[0])
        assert encodings[1] == custom_plotly[1]
    else:
        assert scripts[1] == str(custom_plotly)
    assert scripts[2].endswith("kaleido_scopes.js")


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.data())
async def test_custom_mathjax_url(tmp_path, data):
    """Test custom mathjax URL override."""
    custom_mathjax = data.draw(st_valid_path(tmp_path))
    with_custom = PageGenerator(mathjax=custom_mathjax).generate_index()
    scripts, encodings = get_scripts_from_html(with_custom)

    assert len(scripts) == 3  # noqa: PLR2004
    if isinstance(custom_mathjax, tuple):
        assert scripts[0] == str(custom_mathjax[0])
        assert encodings[0] == custom_mathjax[1]
    else:
        assert scripts[0] == str(custom_mathjax)
    assert scripts[1].endswith("package_data/plotly.min.js")
    assert scripts[2].endswith("kaleido_scopes.js")


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.data())
async def test_other_scripts(tmp_path, data):
    """Test adding other scripts."""
    other_scripts = data.draw(st_others_list(tmp_path))
    with_others = PageGenerator(others=other_scripts).generate_index()
    scripts, encodings = get_scripts_from_html(with_others)

    # mathjax + plotly + others + kaleido_scopes
    expected_count = 2 + len(other_scripts) + 1
    assert len(scripts) == expected_count
    assert scripts[0] == DEFAULT_MATHJAX
    assert scripts[1].endswith("package_data/plotly.min.js")

    # Check all other scripts in order
    for i, script in enumerate(other_scripts):
        if isinstance(script, tuple):
            assert scripts[2 + i] == str(script[0])
            assert encodings[2 + i] == script[1]
        else:
            assert scripts[2 + i] == str(script)

    assert scripts[-1].endswith("kaleido_scopes.js")


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.data())
async def test_combined_overrides(tmp_path, data):
    """Test combination of multiple overrides."""
    custom_plotly = data.draw(st_valid_path(tmp_path))
    custom_mathjax = data.draw(st_mathjax(tmp_path))
    other_scripts = data.draw(st_others_list(tmp_path))

    combined = PageGenerator(
        plotly=custom_plotly,
        mathjax=custom_mathjax,
        others=other_scripts,
    ).generate_index()
    scripts, encodings = get_scripts_from_html(combined)

    # Calculate expected count
    expected_count = 0
    script_index = 0

    # Mathjax adds one (unless False)
    if custom_mathjax is not False:
        expected_count += 1
        if custom_mathjax is None:
            expected_mathjax = DEFAULT_MATHJAX
        elif isinstance(custom_mathjax, tuple):
            expected_mathjax = str(custom_mathjax[0])
            assert encodings[script_index] == custom_mathjax[1]
        else:
            expected_mathjax = str(custom_mathjax)
        assert scripts[script_index] == expected_mathjax
        script_index += 1

    # Plotly always adds one
    expected_count += 1
    if isinstance(custom_plotly, tuple):
        assert scripts[script_index] == str(custom_plotly[0])
        assert encodings[script_index] == custom_plotly[1]
    else:
        assert scripts[script_index] == str(custom_plotly)
    script_index += 1

    # Others adds however many are in the list
    expected_count += len(other_scripts)
    for script in other_scripts:
        if isinstance(script, tuple):
            assert scripts[script_index] == str(script[0])
            assert encodings[script_index] == script[1]
        else:
            assert scripts[script_index] == str(script)
        script_index += 1

    # Kaleido scopes always adds one
    expected_count += 1
    assert scripts[script_index].endswith("kaleido_scopes.js")

    assert len(scripts) == expected_count


# Test file path validation
async def test_existing_file_path(temp_js_file):
    """Test that existing file paths work with and without file:/// protocol."""
    # Test with regular path
    generator = PageGenerator(plotly=str(temp_js_file))
    html = generator.generate_index()
    scripts, _encodings = get_scripts_from_html(html)
    assert len(scripts) == 3  # noqa: PLR2004
    assert scripts[0] == DEFAULT_MATHJAX
    assert scripts[1] == str(temp_js_file)
    assert scripts[2].endswith("kaleido_scopes.js")

    # Test with file:/// protocol
    generator_uri = PageGenerator(plotly=temp_js_file.as_uri())
    html_uri = generator_uri.generate_index()
    scripts_uri, _encodings_uri = get_scripts_from_html(html_uri)
    assert len(scripts_uri) == 3  # noqa: PLR2004
    assert scripts_uri[0] == DEFAULT_MATHJAX
    assert scripts_uri[1] == temp_js_file.as_uri()
    assert scripts_uri[2].endswith("kaleido_scopes.js")


async def test_nonexistent_file_path_raises_error(
    nonexistent_file_path,
    nonexistent_file_uri,
):
    """Test that nonexistent file paths raise FileNotFoundError."""
    # Test with regular path
    with pytest.raises(FileNotFoundError):
        PageGenerator(plotly=str(nonexistent_file_path))

    with pytest.raises(FileNotFoundError):
        PageGenerator(plotly=Path(nonexistent_file_path))

    # Test with file:/// protocol
    with pytest.raises(FileNotFoundError):
        PageGenerator(plotly=nonexistent_file_uri)


async def test_mathjax_nonexistent_file_raises_error(
    nonexistent_file_path,
    nonexistent_file_uri,
):
    """Test that nonexistent mathjax file raises FileNotFoundError."""
    # Test with regular path
    with pytest.raises(FileNotFoundError):
        PageGenerator(mathjax=str(nonexistent_file_path))

    with pytest.raises(FileNotFoundError):
        PageGenerator(mathjax=nonexistent_file_path)

    # Test with file:/// protocol
    with pytest.raises(FileNotFoundError):
        PageGenerator(mathjax=nonexistent_file_uri)


async def test_others_nonexistent_file_raises_error(
    nonexistent_file_path,
    nonexistent_file_uri,
):
    """Test that nonexistent file in others list raises FileNotFoundError."""
    # Test with regular path
    with pytest.raises(FileNotFoundError):
        PageGenerator(others=[str(nonexistent_file_path)])

    with pytest.raises(FileNotFoundError):
        PageGenerator(others=[nonexistent_file_path])

    # Test with file:/// protocol
    with pytest.raises(FileNotFoundError):
        PageGenerator(others=[nonexistent_file_uri])


# Test HTTP URLs (should not raise FileNotFoundError)
async def test_http_urls_skip_file_validation():
    """Test that HTTP URLs skip file existence validation."""
    # These should not raise FileNotFoundError even if URLs don't exist
    generator = PageGenerator(
        plotly="https://nonexistent.example.com/plotly.js",
        mathjax="https://nonexistent.example.com/mathjax.js",
        others=["https://nonexistent.example.com/other.js"],
    )
    html = generator.generate_index()
    scripts, _encodings = get_scripts_from_html(html)

    assert len(scripts) == 4  # noqa: PLR2004
    assert scripts[0] == "https://nonexistent.example.com/mathjax.js"
    assert scripts[1] == "https://nonexistent.example.com/plotly.js"
    assert scripts[2] == "https://nonexistent.example.com/other.js"
    assert scripts[3].endswith("kaleido_scopes.js")

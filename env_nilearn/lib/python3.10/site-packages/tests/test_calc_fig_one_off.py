import warnings

import logistro
import pytest

import kaleido

# allows to create a browser pool for tests
pytestmark = pytest.mark.asyncio(loop_scope="function")

_logger = logistro.getLogger(__name__)


async def test_calc_fig():
    # ruff: noqa: PLC0415
    import plotly.express as px

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        fig = px.line(x=[1, 2, 3, 4], y=[1, 2, 3, 4])

    # Pixel by pixel testing
    img_awaited = await kaleido.calc_fig(fig)
    assert isinstance(img_awaited, bytes)

    # Make sure passing `mathjax` argument via `kopts` works properly
    img_sync = kaleido.calc_fig_sync(
        fig,
        kopts={"mathjax": "https://cdn.jsdelivr.net/npm/mathjax@3.2.1/es5/tex-svg.js"},
    )
    assert isinstance(img_sync, bytes)

    img_from_dict = kaleido.calc_fig_sync(fig.to_dict())
    assert isinstance(img_from_dict, bytes)

    assert img_awaited == img_sync == img_from_dict

    with pytest.raises(TypeError):
        # can't accept iterables
        _ = kaleido.calc_fig_sync([fig, fig])

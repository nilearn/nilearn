import numpy as np
import plotly.graph_objects as go
import pytest
from plotly.subplots import make_subplots

import kaleido

pytestmark = pytest.mark.asyncio(loop_scope="function")

rng = np.random.default_rng()  # creates a Generator instance


async def test_complex_plotly_encoder():
    """Test that kaleido can handle complex Plotly figures with numpy arrays."""

    # Create complex numpy arrays
    x = np.linspace(0, 4 * np.pi, 100)
    y1 = np.sin(x) * np.exp(-x / 10)

    # Create a 2D array for heatmap
    z = np.outer(
        np.sin(np.linspace(0, np.pi, 20)),
        np.cos(np.linspace(0, np.pi, 30)),
    )

    # Create subplot figure
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Complex Scatter",
            "Heatmap",
            "Bar with Color",
            "3D Surface",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "heatmap"}],
            [{"type": "bar"}, {"type": "surface"}],
        ],
    )

    # Scatter with numpy marker sizes and complex styling
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y1,
            mode="lines+markers",
            line={"color": "blue", "width": 2},
            marker={
                "size": [rng.integers(2, 12) for _ in range(len(x))],
                "color": np.abs(y1),
                "colorscale": "Viridis",
                "showscale": True,
            },
        ),
        row=1,
        col=1,
    )

    # Heatmap with 2D numpy array
    fig.add_trace(
        go.Heatmap(
            z=z,
            colorscale="Plasma",
        ),
        row=1,
        col=2,
    )

    # Bar chart with numpy data and color mapping
    categories = np.array(["A", "B", "C", "D", "E", "F"])
    values = rng.normal(50, 20, len(categories))

    fig.add_trace(
        go.Bar(
            x=categories,
            y=values,
            marker={
                "color": np.abs(values),
                "colorscale": "Blues",
                "line": {"width": 2, "color": "black"},
            },
        ),
        row=2,
        col=1,
    )

    # 3D surface with complex numpy operations
    x_surf = np.linspace(-3, 3, 30)
    y_surf = np.linspace(-3, 3, 30)
    X, Y = np.meshgrid(x_surf, y_surf)  # noqa: N806
    Z = np.sin(np.sqrt(X**2 + Y**2)) * np.exp(-(X**2 + Y**2) / 10)  # noqa: N806

    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale="Cividis",
        ),
        row=2,
        col=2,
    )

    # Complex layout with numpy-based annotations
    fig.update_layout(
        title="Complex Numpy Figure for Encoder Testing",
        height=800,
        width=1200,
    )

    # Render with kaleido
    img_bytes = await kaleido.calc_fig(fig)

    assert isinstance(img_bytes, bytes)

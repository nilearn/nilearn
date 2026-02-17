"""Create pickled fig for use in integration tests."""

import pickle
from pathlib import Path

import datashader as ds
import datashader.transfer_functions as tf
import pandas as pd
import plotly.express as px
import zstandard as zstd
from colorcet import fire

cctx = zstd.ZstdCompressor(level=20)

df = pd.read_csv(
    "https://raw.githubusercontent.com/plotly/datasets/master/uber-rides-data1.csv",
)
dff = (
    df.query("Lat < 40.82")
    .query("Lat > 40.70")
    .query("Lon > -74.02")
    .query("Lon < -73.91")
)


cvs = ds.Canvas(plot_width=1000, plot_height=1000)
agg = cvs.points(dff, x="Lon", y="Lat")
# agg is an xarray object, see http://xarray.pydata.org/en/stable/ for more details
coords_lat, coords_lon = agg.coords["Lat"].to_numpy(), agg.coords["Lon"].to_numpy()
# Corners of the image
coordinates = [
    [coords_lon[0], coords_lat[0]],
    [coords_lon[-1], coords_lat[0]],
    [coords_lon[-1], coords_lat[-1]],
    [coords_lon[0], coords_lat[-1]],
]


img = tf.shade(agg, cmap=fire)[::-1].to_pil()


# Trick to create rapidly a figure with map axes
fig = px.scatter_map(dff[:1], lat="Lat", lon="Lon", zoom=12)
# Add the datashader image as a tile map layer image
fig.update_layout(
    map_style="carto-darkmatter",
    map_layers=[{"sourcetype": "image", "source": img, "coordinates": coordinates}],
)

raw = pickle.dumps(fig, protocol=5)  # >=3.8
compressed = cctx.compress(raw)
with Path(f"./figs/{Path(__file__).stem}.pkl.zst").open("wb") as f:
    f.write(compressed)

print(  # noqa: T201
    f"{Path(__file__).stem}.pkl: "
    f"{len(raw) / 1024:.1f} -> {len(compressed) / 1024:.1f} KB",
)

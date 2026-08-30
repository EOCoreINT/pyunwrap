"""
pyunwrap.visualization.maps
==============================

Interactive 2D raster maps for InSAR phase results, built on `folium`.

`folium` (Leaflet.js) requires image overlays to be georeferenced in
WGS84 (lat/lon) bounds, so every function here reprojects the source
raster's bounds from its native CRS via `rasterio.warp.transform_bounds`
before building the map.

Rather than a literal slider widget switching between 3 independent layers
(no standard Leaflet/folium plugin does an N-way slider), this module uses
`folium.plugins.SideBySideLayers` -- a swipe/slider comparison between
exactly *two* layers -- for the most diagnostically useful pair (wrapped vs.
unwrapped phase), and exposes the remaining layers (coherence, residue
probability, etc.) as ordinary togglable overlays via `folium.LayerControl`.
This is noted explicitly here since it's a deliberate adaptation of the
"slider between layers" requirement to what the underlying mapping library
actually supports well.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np

try:
    import folium
    from folium.plugins import SideBySideLayers

    _HAS_FOLIUM = True
except ImportError:  # pragma: no cover
    _HAS_FOLIUM = False

try:
    from rasterio.warp import transform_bounds

    _HAS_RASTERIO = True
except ImportError:  # pragma: no cover
    _HAS_RASTERIO = False

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Raster -> PNG data URI
# --------------------------------------------------------------------------- #


def array_to_png_data_uri(
    array: np.ndarray,
    cmap: str = "twilight",
    vmin: float | None = None,
    vmax: float | None = None,
) -> str:
    """Render a 2D array to a base64-encoded PNG data URI, for use as a
    `folium.raster_layers.ImageOverlay` source.

    Args:
        array: 2D data array.
        cmap: Matplotlib colormap name.
        vmin: Color scale minimum; defaults to `array.min()`.
        vmax: Color scale maximum; defaults to `array.max()`.

    Returns:
        A `"data:image/png;base64,..."` URI string.
    """
    vmin = float(array.min()) if vmin is None else vmin
    vmax = float(array.max()) if vmax is None else vmax
    if vmax <= vmin:
        vmax = vmin + 1e-6

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colormap = matplotlib.colormaps.get_cmap(cmap)
    rgba = colormap(norm(array))  # HxWx4, float in [0, 1]

    buf = io.BytesIO()
    plt.imsave(buf, rgba, format="png")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _wgs84_bounds_from_profile(profile: dict) -> list[list[float]]:
    """Compute `[[south, west], [north, east]]` WGS84 bounds from a rasterio
    profile (CRS + affine transform + array shape), for `folium.ImageOverlay`.

    Args:
        profile: A rasterio dataset profile dict (as returned by
            `rasterio.open(...).profile`, e.g.
            `UnwrapResult.geotiff_profile`).

    Returns:
        `[[south, west], [north, east]]`, suitable for
        `folium.raster_layers.ImageOverlay(bounds=...)`.

    Raises:
        ImportError: If rasterio is not installed.
    """
    if not _HAS_RASTERIO:
        raise ImportError("rasterio is required to georeference map overlays.")
    transform = profile["transform"]
    crs = profile["crs"]
    height, width = profile["height"], profile["width"]

    left, top = transform @ (0, 0)
    right, bottom = transform @ (width, height)

    west, south, east, north = transform_bounds(crs, "EPSG:4326", left, bottom, right, top)
    return [[south, west], [north, east]]


# --------------------------------------------------------------------------- #
# Map builders
# --------------------------------------------------------------------------- #


def build_phase_comparison_map(
    wrapped_phase: np.ndarray,
    unwrapped_phase: np.ndarray,
    geotiff_profile: dict,
    coherence: np.ndarray | None = None,
    residue_prob: np.ndarray | None = None,
    zoom_start: int = 12,
) -> folium.Map:
    """Build an interactive map with a swipe slider comparing wrapped vs.
    unwrapped phase, plus optional togglable coherence / residue-probability
    overlays.

    Args:
        wrapped_phase: 2D wrapped phase array, radians.
        unwrapped_phase: 2D unwrapped phase array, radians, same shape.
        geotiff_profile: Source raster's rasterio profile (CRS + transform),
            e.g. `UnwrapResult.geotiff_profile`, used to georeference the
            overlay in WGS84.
        coherence: Optional 2D coherence array, [0, 1], added as a togglable
            overlay layer.
        residue_prob: Optional 2D residue probability array, [0, 1], added
            as a togglable overlay layer.
        zoom_start: Initial Leaflet zoom level.

    Returns:
        A `folium.Map` with a wrapped/unwrapped swipe control and any
        additional layers in the layer-control panel.

    Raises:
        ImportError: If folium is not installed.
    """
    if not _HAS_FOLIUM:
        raise ImportError("folium is required for interactive maps (`pip install folium`).")

    bounds = _wgs84_bounds_from_profile(geotiff_profile)
    center = [(bounds[0][0] + bounds[1][0]) / 2.0, (bounds[0][1] + bounds[1][1]) / 2.0]

    fmap = folium.Map(location=center, zoom_start=zoom_start, tiles="OpenStreetMap")

    wrapped_uri = array_to_png_data_uri(wrapped_phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    vmax_unwrap = float(np.abs(unwrapped_phase).max())
    unwrapped_uri = array_to_png_data_uri(
        unwrapped_phase, cmap="jet", vmin=-vmax_unwrap, vmax=vmax_unwrap
    )

    left_layer = folium.raster_layers.ImageOverlay(
        wrapped_uri,
        bounds=bounds,
        name="Wrapped phase",
        opacity=1.0,
    )
    right_layer = folium.raster_layers.ImageOverlay(
        unwrapped_uri,
        bounds=bounds,
        name="Unwrapped phase",
        opacity=1.0,
    )
    left_layer.add_to(fmap)
    right_layer.add_to(fmap)
    SideBySideLayers(layer_left=left_layer, layer_right=right_layer).add_to(fmap)

    if coherence is not None:
        coherence_uri = array_to_png_data_uri(coherence, cmap="viridis", vmin=0.0, vmax=1.0)
        folium.raster_layers.ImageOverlay(
            coherence_uri,
            bounds=bounds,
            name="Coherence",
            opacity=0.8,
            show=False,
        ).add_to(fmap)

    if residue_prob is not None:
        residue_uri = array_to_png_data_uri(residue_prob, cmap="inferno", vmin=0.0, vmax=1.0)
        folium.raster_layers.ImageOverlay(
            residue_uri,
            bounds=bounds,
            name="Residue probability",
            opacity=0.8,
            show=False,
        ).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


def save_map(fmap: folium.Map, out_path: str | Path) -> Path:
    """Save a folium map to a standalone HTML file.

    Args:
        fmap: The map to save.
        out_path: Output `.html` path.

    Returns:
        The path written to.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_path))
    return out_path

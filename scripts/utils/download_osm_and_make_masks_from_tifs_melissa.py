#!/usr/bin/env python3
import os
import glob
import warnings

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.enums import Resampling
from PIL import Image
from tqdm import tqdm
from shapely.geometry import box

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
TIF_DIR = "/media/volume/data_2d/semseg_2d/data/melissa_tif1"
OUT_DIR = "/media/volume/data_2d/semseg_2d/data/melissa_imagery/melissa_tif_building_masks1"
os.makedirs(OUT_DIR, exist_ok=True)

FOOTPRINTS_FILE = os.path.join(OUT_DIR, "osm_buildings.geojson")
DOWNLOAD_OSM = True  # set False to reuse cached FOOTPRINTS_FILE

WRITE_MASK_GEOTIFF = True
WRITE_MASK_PNG = True
WRITE_OVERLAY_PNG = True

MASK_VALUE = 255
ALL_TOUCHED = False  # True makes thicker masks

OVERLAY_ALPHA = 0.35  # 0..1


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def bbox4326_from_dataset_bounds(bounds, src_crs):
    """Convert dataset bounds (in src_crs) to WGS84 bbox tuple."""
    geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    gs = gpd.GeoSeries([geom], crs=src_crs)  # GeoSeries from list is widely supported [web:164]
    gs = gs.to_crs("EPSG:4326")              # reproject GeoSeries [web:162]
    return tuple(gs.total_bounds)            # (minx,miny,maxx,maxy)


def union_bbox_of_tifs(tif_paths):
    """Return union bbox in EPSG:4326: (minx,miny,maxx,maxy)."""
    minx = miny = 1e30
    maxx = maxy = -1e30

    for p in tif_paths:
        with rasterio.open(p) as src:
            if src.crs is None:
                raise ValueError(f"{p} has no CRS")
            bb = bbox4326_from_dataset_bounds(src.bounds, src.crs)
            minx = min(minx, bb[0]); miny = min(miny, bb[1])
            maxx = max(maxx, bb[2]); maxy = max(maxy, bb[3])

    return (minx, miny, maxx, maxy)


def download_osm_buildings(bbox4326):
    """
    Download OSM building footprints once for a bbox (EPSG:4326).
    Uses OSMnx features module with tags={'building': True}. [web:103]
    """
    import osmnx as ox

    tags = {"building": True}
    gdf = ox.features.features_from_bbox(bbox4326, tags)  # [web:103]
    if gdf is None or len(gdf) == 0:
        return gpd.GeoDataFrame()

    gdf = gdf.reset_index()
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    gdf = gdf[["geometry"]].copy()
    return gdf


def to_uint8_rgb(arr):
    """arr: (bands,H,W) uint8 -> (H,W,3) uint8"""
    if arr.ndim == 3 and arr.shape[0] >= 3:
        return np.stack([arr[0], arr[1], arr[2]], axis=-1)
    raise ValueError("Expected at least 3 bands")


def save_overlay_png(rgb_u8, mask_u8, out_path):
    img = rgb_u8.astype(np.float32) / 255.0
    m = (mask_u8 > 0).astype(np.float32)

    red = np.zeros_like(img)
    red[..., 0] = 1.0

    out = img * (1 - OVERLAY_ALPHA * m[..., None]) + red * (OVERLAY_ALPHA * m[..., None])
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(out_path)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    tif_paths = sorted(glob.glob(os.path.join(TIF_DIR, "*.tif")))
    if not tif_paths:
        print(f"ERROR: No .tif found in {TIF_DIR}")
        return

    print(f"Found {len(tif_paths)} GeoTIFFs")

    # 1) Compute union bbox in WGS84 for one-shot footprint download
    bbox4326 = union_bbox_of_tifs(tif_paths)
    print(f"Union bbox (EPSG:4326): {bbox4326}")

    # 2) Get footprints (download or read cached)
    if DOWNLOAD_OSM or not os.path.exists(FOOTPRINTS_FILE):
        print("Downloading OSM buildings...")
        footprints = download_osm_buildings(bbox4326)
        if footprints.empty:
            print("ERROR: OSM returned no buildings for union bbox")
            return
        footprints.to_file(FOOTPRINTS_FILE, driver="GeoJSON")
        print(f"Saved footprints: {FOOTPRINTS_FILE}")
    else:
        print(f"Reading cached footprints: {FOOTPRINTS_FILE}")
        footprints = gpd.read_file(FOOTPRINTS_FILE)
        if footprints.crs is None:
            footprints = footprints.set_crs("EPSG:4326")
        else:
            footprints = footprints.to_crs("EPSG:4326")
        footprints = footprints[["geometry"]].copy()

    # 3) Rasterize per tile
    for tif_path in tqdm(tif_paths, desc="Rasterizing"):
        name = os.path.splitext(os.path.basename(tif_path))[0]

        with rasterio.open(tif_path) as src:
            if src.crs is None:
                continue

            # Reproject footprints to this tile CRS
            fp = footprints.to_crs(src.crs)

            # Clip footprints to tile bounds to speed rasterization
            b = src.bounds
            tile_box = box(b.left, b.bottom, b.right, b.top)
            fp = fp[fp.intersects(tile_box)].copy()
            if fp.empty:
                continue

            shapes = [(geom, MASK_VALUE) for geom in fp.geometry if geom is not None and not geom.is_empty]
            mask = rasterize(  # burns polygons into raster pixel grid [web:151]
                shapes=shapes,
                out_shape=(src.height, src.width),
                transform=src.transform,
                fill=0,
                dtype=np.uint8,
                all_touched=ALL_TOUCHED,
            )

            if WRITE_MASK_GEOTIFF:
                out_mask_tif = os.path.join(OUT_DIR, f"{name}_mask.tif")
                meta = src.meta.copy()
                meta.update(driver="GTiff", count=1, dtype=rasterio.uint8, compress="deflate")
                with rasterio.open(out_mask_tif, "w", **meta) as dst:
                    dst.write(mask, 1)

            if WRITE_MASK_PNG:
                out_mask_png = os.path.join(OUT_DIR, f"{name}_mask.png")
                Image.fromarray(mask).save(out_mask_png)

            if WRITE_OVERLAY_PNG:
                max_dim = 2048
                scale = max(src.width / max_dim, src.height / max_dim, 1.0)
                if scale > 1.0:
                    out_w = int(src.width / scale)
                    out_h = int(src.height / scale)
                    rgb = src.read([1, 2, 3], out_shape=(3, out_h, out_w), resampling=Resampling.bilinear)
                    mask_small = Image.fromarray(mask).resize((out_w, out_h), resample=Image.NEAREST)
                    mask_small = np.array(mask_small, dtype=np.uint8)
                else:
                    rgb = src.read([1, 2, 3])
                    mask_small = mask

                rgb_u8 = to_uint8_rgb(rgb)
                out_overlay = os.path.join(OUT_DIR, f"{name}_overlay.png")
                save_overlay_png(rgb_u8, mask_small, out_overlay)

    print("Done.")
    print(f"Outputs in: {OUT_DIR}")
    print("Reminder: OSM data is ODbL; attribute OpenStreetMap contributors when you share results.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import glob
import re
import warnings

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from PIL import Image, ImageDraw
from tqdm import tqdm
from pyproj import Transformer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
RAW_DIR = "/media/volume/data_2d/semseg_2d/data/melissa_imagery/20251103a_RGB/raw"

# Folder containing per-image footprints produced earlier:
#   {image_id}_buildings.geojson
FOOTPRINTS_DIR = "/media/volume/data_2d/semseg_2d/data/melissa_imagery/20251103a_RGB/building_footprints"
FOOTPRINT_SUFFIX = "_buildings.geojson"   # adjust if needed

OUT_MASK_DIR = os.path.join(FOOTPRINTS_DIR, "masks_utm_affine")
OUT_OVERLAY_DIR = os.path.join(FOOTPRINTS_DIR, "overlays_utm_affine")
os.makedirs(OUT_MASK_DIR, exist_ok=True)
os.makedirs(OUT_OVERLAY_DIR, exist_ok=True)

MASK_VALUE = 255
OVERLAY_COLOR = (255, 0, 0)   # red
OVERLAY_ALPHA = 120           # 0..255


# ---------------------------------------------------------------------
# .geom parsing
# ---------------------------------------------------------------------
def _read_geom_text(geom_path: str) -> str:
    with open(geom_path, "r", errors="replace") as f:
        return f.read()

def _get_val(txt: str, keys, cast=float):
    if isinstance(keys, str):
        keys = [keys]
    for k in keys:
        m = re.search(rf"{re.escape(k)}\s*:?\s*([\-0-9\.eE]+)", txt)
        if m:
            return cast(m.group(1))
    raise ValueError(f"Missing keys={keys}")

def parse_geom_corners_zone(geom_path: str):
    """
    Returns:
      corners_lonlat: (ul, ur, lr, ll) each (lon,lat)
      utm_epsg: like 'EPSG:32618' or 'EPSG:32718'
    """
    txt = _read_geom_text(geom_path)

    ul_lat = _get_val(txt, ["ul_lat", "ullat"])
    ul_lon = _get_val(txt, ["ul_lon", "ullon"])
    ur_lat = _get_val(txt, ["ur_lat", "urlat"])
    ur_lon = _get_val(txt, ["ur_lon", "urlon"])
    lr_lat = _get_val(txt, ["lr_lat", "lrlat"])
    lr_lon = _get_val(txt, ["lr_lon", "lrlon"])
    ll_lat = _get_val(txt, ["ll_lat", "lllat"])
    ll_lon = _get_val(txt, ["ll_lon", "lllon"])

    utm_zone = _get_val(txt, ["utm_zone", "utmzone", "zone"], int)
    hem = None
    m = re.search(r"(?:utm_hemisphere|utmhemisphere|hemisphere)\s*:?\s*([NS])", txt, re.IGNORECASE)
    if m:
        hem = m.group(1).upper()
    else:
        # fallback by latitude
        hem = "N" if ul_lat >= 0 else "S"

    utm_epsg = f"EPSG:{32600 + utm_zone}" if hem == "N" else f"EPSG:{32700 + utm_zone}"

    corners_lonlat = ((ul_lon, ul_lat), (ur_lon, ur_lat), (lr_lon, lr_lat), (ll_lon, ll_lat))
    return corners_lonlat, utm_epsg


# ---------------------------------------------------------------------
# UTM affine mapping
# ---------------------------------------------------------------------
def fit_affine_EN_to_xy(corners_lonlat, utm_epsg, W, H):
    """
    Fit affine:
      [x, y] = [E, N, 1] @ M
    where M is 3x2.
    """
    # always_xy=True ensures lon,lat order for EPSG:4326 transforms
    tf = Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)

    # Convert corners to UTM EN
    EN = []
    for (lon, lat) in corners_lonlat:
        e, n = tf.transform(lon, lat)
        EN.append((e, n))
    EN = np.asarray(EN, dtype=np.float64)  # ul,ur,lr,ll

    # Pixel coords for those corners
    # ul=(0,0), ur=(W-1,0), lr=(W-1,H-1), ll=(0,H-1)
    XY = np.asarray([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float64)

    A = np.column_stack([EN[:, 0], EN[:, 1], np.ones((4,), dtype=np.float64)])  # 4x3
    # Solve least squares A @ M = XY
    M, _, _, _ = np.linalg.lstsq(A, XY, rcond=None)  # 3x2
    return M, tf


def lonlat_coords_to_xy(coords_lonlat, M, tf):
    """
    coords_lonlat: Nx2 (lon,lat)
    Returns Nx2 (x,y)
    """
    coords = np.asarray(coords_lonlat, dtype=np.float64)
    lon = coords[:, 0]
    lat = coords[:, 1]
    E, N = tf.transform(lon, lat)
    A = np.column_stack([E, N, np.ones((len(E),), dtype=np.float64)])  # Nx3
    XY = A @ M  # Nx2
    return XY


def polygon_lonlat_to_pixel_parts(geom, M, tf):
    """
    Convert Polygon/MultiPolygon lon/lat -> pixel parts
    Returns list of (ext_xy Nx2, holes_xy_list)
    """
    parts = []
    if geom is None or geom.is_empty:
        return parts

    if isinstance(geom, Polygon):
        polys = [geom]
    elif isinstance(geom, MultiPolygon):
        polys = list(geom.geoms)
    else:
        return parts

    for p in polys:
        if p.is_empty:
            continue
        ext = np.asarray(p.exterior.coords, dtype=np.float64)
        ext_xy = lonlat_coords_to_xy(ext, M, tf)

        holes_xy = []
        for ring in p.interiors:
            rr = np.asarray(ring.coords, dtype=np.float64)
            rr_xy = lonlat_coords_to_xy(rr, M, tf)
            holes_xy.append(rr_xy)

        parts.append((ext_xy, holes_xy))
    return parts


# ---------------------------------------------------------------------
# Rasterization + overlay
# ---------------------------------------------------------------------
def rasterize_parts(W, H, parts):
    """
    Create uint8 mask (H,W), fill building with 255 and holes with 0.
    """
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)

    for ext_xy, holes_xy in parts:
        ext = [(float(x), float(y)) for x, y in ext_xy]
        draw.polygon(ext, fill=MASK_VALUE)

        for hole in holes_xy:
            hh = [(float(x), float(y)) for x, y in hole]
            draw.polygon(hh, fill=0)

    return np.array(mask, dtype=np.uint8)


def overlay(img_rgb, mask_u8, color=(255, 0, 0), alpha=120):
    img = img_rgb.convert("RGBA")
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))

    m = Image.fromarray(mask_u8, mode="L")
    colored = Image.new("RGBA", img.size, (*color, alpha))
    over.paste(colored, (0, 0), mask=m)

    return Image.alpha_composite(img, over)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    jpgs = sorted(glob.glob(os.path.join(RAW_DIR, "*.jpg")))
    print(f"Found {len(jpgs)} JPG files in {RAW_DIR}")

    made = 0
    for jpg_path in tqdm(jpgs, desc="Masks"):
        image_id = os.path.splitext(os.path.basename(jpg_path))[0]
        geom_path = os.path.join(RAW_DIR, image_id + ".geom")
        footprints_path = os.path.join(FOOTPRINTS_DIR, image_id + FOOTPRINT_SUFFIX)

        if not os.path.exists(geom_path):
            continue
        if not os.path.exists(footprints_path):
            continue

        # Load image
        img = Image.open(jpg_path).convert("RGB")
        W, H = img.size

        # Build affine UTM->pixel
        corners_lonlat, utm_epsg = parse_geom_corners_zone(geom_path)
        M, tf = fit_affine_EN_to_xy(corners_lonlat, utm_epsg, W, H)

        # Load footprints (assume EPSG:4326 if missing)
        gdf = gpd.read_file(footprints_path)
        if gdf.empty:
            continue
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        else:
            gdf = gdf.to_crs("EPSG:4326")

        # Convert all polygons to pixel space parts
        parts_all = []
        for geom in gdf.geometry:
            parts_all.extend(polygon_lonlat_to_pixel_parts(geom, M, tf))

        if not parts_all:
            continue

        # Rasterize + save
        mask = rasterize_parts(W, H, parts_all)
        out_mask = os.path.join(OUT_MASK_DIR, f"{image_id}_mask.png")
        Image.fromarray(mask).save(out_mask)

        out_overlay = os.path.join(OUT_OVERLAY_DIR, f"{image_id}_overlay.png")
        overlay(img, mask, color=OVERLAY_COLOR, alpha=OVERLAY_ALPHA).save(out_overlay)

        made += 1

    print(f"Done. Created {made} masks.")
    print(f"Masks:    {OUT_MASK_DIR}")
    print(f"Overlays: {OUT_OVERLAY_DIR}")


if __name__ == "__main__":
    main()

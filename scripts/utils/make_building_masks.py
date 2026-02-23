#!/usr/bin/env python3
import os
import glob
import re
import json
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from PIL import Image, ImageDraw
from tqdm import tqdm

RAW_DIR = "/media/volume/data_2d/semseg_2d/data/melissa_imagery/20251103a_RGB/raw"

# Folder that contains your clipped footprints per image, e.g.
# building_footprints/178480_1103251414210_029_RGB2_buildings.geojson
FOOTPRINTS_DIR = "/media/volume/data_2d/semseg_2d/data/melissa_imagery/20251103a_RGB/building_footprints"

OUT_MASK_DIR = os.path.join(FOOTPRINTS_DIR, "masks")
OUT_OVERLAY_DIR = os.path.join(FOOTPRINTS_DIR, "overlays")
os.makedirs(OUT_MASK_DIR, exist_ok=True)
os.makedirs(OUT_OVERLAY_DIR, exist_ok=True)

MASK_SUFFIX = "_mask.png"
OVERLAY_SUFFIX = "_overlay.png"
FOOTPRINT_SUFFIX = "_buildings.geojson"  # change if your files are named differently


def parse_geom_corners(geom_path: str):
    """Read ul/ur/lr/ll lon/lat from .geom (WGS84)."""
    with open(geom_path, "r", errors="replace") as f:
        txt = f.read()

    def get_val(keys, cast=float):
        if isinstance(keys, str):
            keys = [keys]
        for k in keys:
            m = re.search(rf"{re.escape(k)}\s*:?\s*([\-0-9\.eE]+)", txt)
            if m:
                return cast(m.group(1))
        raise ValueError(f"Missing {keys} in {geom_path}")

    ul_lat = get_val(["ul_lat", "ullat"])
    ul_lon = get_val(["ul_lon", "ullon"])
    ur_lat = get_val(["ur_lat", "urlat"])
    ur_lon = get_val(["ur_lon", "urlon"])
    lr_lat = get_val(["lr_lat", "lrlat"])
    lr_lon = get_val(["lr_lon", "lrlon"])
    ll_lat = get_val(["ll_lat", "lllat"])
    ll_lon = get_val(["ll_lon", "lllon"])

    # Useful but not required for mask generation:
    # number_lines = get_val(["number_lines", "numberlines"], int)
    # number_samples = get_val(["number_samples", "numbersamples"], int)

    return (ul_lon, ul_lat), (ur_lon, ur_lat), (lr_lon, lr_lat), (ll_lon, ll_lat)


def homography_from_4pts(src_xy, dst_uv):
    """
    Compute homography H such that [u v 1]^T ~ H [x y 1]^T
    src_xy: 4x2 pixels, dst_uv: 4x2 lon/lat
    """
    A = []
    for (x, y), (u, v) in zip(src_xy, dst_uv):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.asarray(A, dtype=np.float64)

    # Solve Ah=0 using SVD
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    H = h.reshape(3, 3)
    return H / H[2, 2]


def apply_homography(H, pts):
    """pts Nx2 -> Nx2"""
    pts = np.asarray(pts, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    p = np.hstack([pts, ones])
    q = (H @ p.T).T
    q = q[:, :2] / q[:, 2:3]
    return q


def polygon_lonlat_to_pixel(poly, H_lonlat_to_px):
    """
    Transform shapely Polygon/MultiPolygon in lon/lat to pixel coords using homography.
    Returns list of (exterior, [holes]) in pixel coordinates.
    """
    parts = []
    if isinstance(poly, Polygon):
        polys = [poly]
    elif isinstance(poly, MultiPolygon):
        polys = list(poly.geoms)
    else:
        return parts

    for p in polys:
        if p.is_empty:
            continue

        ext = np.asarray(p.exterior.coords, dtype=np.float64)  # lon/lat
        ext_px = apply_homography(H_lonlat_to_px, ext)         # x/y

        holes_px = []
        for ring in p.interiors:
            rr = np.asarray(ring.coords, dtype=np.float64)
            rr_px = apply_homography(H_lonlat_to_px, rr)
            holes_px.append(rr_px)

        parts.append((ext_px, holes_px))
    return parts


def draw_mask_from_polygons(img_w, img_h, poly_parts):
    """
    poly_parts: list of (ext_px Nx2, holes_px list)
    Returns uint8 mask [H,W] in {0,255}
    """
    mask = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)

    for ext_px, holes_px in poly_parts:
        ext = [(float(x), float(y)) for x, y in ext_px]
        draw.polygon(ext, fill=255)

        # Carve holes
        for hole in holes_px:
            hh = [(float(x), float(y)) for x, y in hole]
            draw.polygon(hh, fill=0)

    return np.array(mask, dtype=np.uint8)


def overlay_mask_on_image(img_rgb, mask_u8, color=(255, 0, 0), alpha=120):
    """
    Create a quick visualization overlay: red where mask==255.
    """
    img = img_rgb.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))

    m = Image.fromarray(mask_u8, mode="L")
    colored = Image.new("RGBA", img.size, (*color, alpha))
    overlay.paste(colored, (0, 0), mask=m)

    return Image.alpha_composite(img, overlay)


def main():
    jpgs = sorted(glob.glob(os.path.join(RAW_DIR, "*.jpg")))
    print(f"Found {len(jpgs)} JPGs")

    for jpg_path in tqdm(jpgs, desc="Masking"):
        base = os.path.splitext(os.path.basename(jpg_path))[0]
        geom_path = os.path.join(RAW_DIR, base + ".geom")
        footprints_path = os.path.join(FOOTPRINTS_DIR, base + FOOTPRINT_SUFFIX)

        if not os.path.exists(geom_path):
            continue
        if not os.path.exists(footprints_path):
            # No footprints for this image; skip
            continue

        # Load image to get width/height
        img = Image.open(jpg_path).convert("RGB")
        W, H = img.size

        # Build pixel->lonlat homography from 4 corners in .geom
        (ul, ur, lr, ll) = parse_geom_corners(geom_path)

        src_px = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float64)
        dst_ll = np.array([ul, ur, lr, ll], dtype=np.float64)  # lon/lat

        H_px_to_ll = homography_from_4pts(src_px, dst_ll)
        H_ll_to_px = np.linalg.inv(H_px_to_ll)

        # Read footprints (assumed lon/lat EPSG:4326)
        gdf = gpd.read_file(footprints_path)
        if gdf.empty:
            continue
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        else:
            gdf = gdf.to_crs("EPSG:4326")

        # Transform all polygons to pixel space and rasterize
        all_parts = []
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            all_parts.extend(polygon_lonlat_to_pixel(geom, H_ll_to_px))

        mask = draw_mask_from_polygons(W, H, all_parts)

        out_mask = os.path.join(OUT_MASK_DIR, base + MASK_SUFFIX)
        Image.fromarray(mask).save(out_mask)

        out_overlay = os.path.join(OUT_OVERLAY_DIR, base + OVERLAY_SUFFIX)
        overlay = overlay_mask_on_image(img, mask, color=(255, 0, 0), alpha=120)
        overlay.save(out_overlay)

    print("Done.")
    print(f"Masks:    {OUT_MASK_DIR}")
    print(f"Overlays: {OUT_OVERLAY_DIR}")


if __name__ == "__main__":
    main()

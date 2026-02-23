#!/usr/bin/env python
"""
Compute approximate flood water depth from a DEM (.tif) and
a FloodNet segmentation mask (PNG or TIF).

Includes specific logic to treat buildings (classes 1 and 2) and 
water/flood (classes 3, 5, 8) as a single connected region to 
correctly determine the water level from the surrounding terrain.

Usage:
  python -m scripts.water_depth_floodnet \
    --dem path/to/dem.tif \
    --mask path/to/prediction_mask.png \
    --out path/to/output_depth.png \
    --out-buildings path/to/output_building_depth.png
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import rasterio
from scipy import ndimage
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# FloodNet Classes:
# 0: background
# 1: building flooded       (INCLUDE)
# 2: building non-flooded   (INCLUDE - to handle "surrounded by water" case)
# 3: road flooded           (INCLUDE)
# 4: road non-flooded
# 5: water                  (INCLUDE)
# 6: tree
# 7: vehicle
# 8: pool                   (INCLUDE)
# 9: grass

FLOOD_CLASSES = [1, 2, 3, 5, 8]
BUILDING_CLASSES = [3,5,8]  # Target classes for the second output image

# ---------------- IO helpers ----------------

def depth_to_color(depth: np.ndarray, vmax: float | None = None) -> np.ndarray:
    """Map depth (in meters) to an RGB image (Blue heatmap)."""
    d = depth.copy()
    d[d < 0] = 0

    if vmax is None:
        positive = d[d > 0]
        vmax = np.percentile(positive, 99) if positive.size > 0 else 1.0
    if vmax <= 0: vmax = 1.0

    d_norm = np.clip(d / vmax, 0.0, 1.0)
    h, w = d_norm.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    
    # Simple blue scale
    rgb[..., 2] = 255 * (0.3 + 0.7 * d_norm)
    rgb[..., 1] = 255 * (0.2 + 0.8 * d_norm)
    rgb[..., 0] = 255 * (0.0 + 0.2 * d_norm)
    
    rgb[d == 0] = 0
    return rgb.astype(np.uint8)

def load_dem(dem_path: str):
    with rasterio.open(dem_path) as src:
        arr = src.read(1, masked=True)
        meta = src.meta.copy()
        return arr, meta

def load_mask(mask_path: str) -> np.ndarray:
    """Load flood mask (PNG/JPG/TIF). Returns 2D array of class IDs."""
    p = Path(mask_path)
    if p.suffix.lower() in [".tif", ".tiff"]:
        with rasterio.open(mask_path) as src:
            return src.read(1).astype(np.int64)
    
    # Assume PNG/JPG
    img = Image.open(mask_path).convert("L")
    return np.array(img, dtype=np.int64)

def save_depth(depth: np.ndarray, dem_meta: dict, out_path: str, title: str = "Depth"):
    """Save depth as TIF (data) or PNG (visualization)."""
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if out_p.suffix.lower() in [".tif", ".tiff"]:
        meta = dem_meta.copy()
        meta.update({"dtype": "float32", "count": 1, "nodata": 0.0})
        with rasterio.open(out_p, "w", **meta) as dst:
            dst.write(depth.astype(np.float32), 1)
        print(f"[water_depth] Saved GeoTIFF -> {out_p}")
        
    elif out_p.suffix.lower() == ".png":
        rgb = depth_to_color(depth)
        positive = depth[depth > 0]
        vmax = np.percentile(positive, 99) if positive.size > 0 else 1.0
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(rgb)
        ax.axis("off")
        ax.set_title(title)
        
        # Colorbar
        cax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
        norm = plt.Normalize(vmin=0, vmax=vmax)
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.Blues), cax=cax)
        cb.set_label("Depth (meters)", fontsize=12)
        
        plt.savefig(out_p, dpi=150, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print(f"[water_depth] Saved PNG -> {out_p}")

# ------------- Core Algorithm ---------------

def compute_flood_depth(
    dem_path: str,
    mask_path: str,
    depth_out_path: str,
    building_out_path: str = None,
    boundary_percentile: float = 95.0,
):
    print(f"[water_depth] DEM : {dem_path}")
    print(f"[water_depth] MASK: {mask_path}")

    dem, dem_meta = load_dem(dem_path)
    mask = load_mask(mask_path)

    # Resize mask if needed
    if mask.shape != dem.shape:
        print(f"[water_depth] WARNING: Resizing mask {mask.shape} to DEM {dem.shape}")
        mask_img = Image.fromarray(mask.astype(np.int32))
        mask_img = mask_img.resize((dem.shape[1], dem.shape[0]), Image.NEAREST)
        mask = np.array(mask_img, dtype=np.int64)

    # --- KEY CHANGE: Define 'flood' as Water, Pools, Roads, AND Buildings ---
    is_flood = np.isin(mask, FLOOD_CLASSES)

    if np.sum(is_flood) == 0:
        print("No flooded pixels found (Classes 1, 2, 3, 5, 8).")
        return

    # Connected components
    structure = ndimage.generate_binary_structure(2, 2)
    labels, n_labels = ndimage.label(is_flood, structure=structure)
    print(f"[water_depth] Found {n_labels} connected water/building region(s).")

    depth = np.zeros(dem.shape, dtype=np.float32)
    dem_data = dem.data if np.ma.isMaskedArray(dem) else dem
    dem_mask = dem.mask if np.ma.isMaskedArray(dem) else np.zeros_like(dem, dtype=bool)

    for lab in range(1, n_labels + 1):
        region = (labels == lab)
        if not np.any(region): continue

        # Inner boundary: pixels in region touching non-region
        eroded = ndimage.binary_erosion(region, structure=structure, border_value=0)
        inner_boundary = region & (~eroded)
        inner_boundary = inner_boundary & (~dem_mask)

        boundary_elev = dem_data[inner_boundary]
        # Remove invalid data
        boundary_elev = boundary_elev[np.isfinite(boundary_elev)]

        if boundary_elev.size == 0:
            continue

        # Determine Water Level from boundary elevation
        if boundary_elev.size >= 10:
            water_level = np.percentile(boundary_elev, boundary_percentile)
        else:
            water_level = np.max(boundary_elev)

        # Calculate depth
        region_dem = dem_data[region]
        region_depth = water_level - region_dem
        region_depth[region_depth < 0] = 0.0

        depth[region] = region_depth

        print(f"  Region {lab}: bound_px={boundary_elev.size}, Level={water_level:.2f}m, MaxDepth={region_depth.max():.2f}m")

    # Mask out nodata
    depth[dem_mask] = 0.0
    
    # --- OUTPUT 1: Full Flood Depth ---
    save_depth(depth, dem_meta, depth_out_path, title="Flood Depth (All Classes)")

    # --- OUTPUT 2: Building Depth Only ---
    if building_out_path:
        # Filter: Keep depth ONLY where mask is a Building (Flooded or Non-Flooded)
        is_building = np.isin(mask, BUILDING_CLASSES)
        
        building_depth = np.zeros_like(depth)
        # Apply mask: pixels that are buildings AND have calculated depth
        building_depth[is_building] = depth[is_building]
        
        save_depth(building_depth, dem_meta, building_out_path, title="Flood Extent + Depth")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem", required=True, help="Path to DEM GeoTIFF.")
    parser.add_argument("--mask", required=True, help="Path to raw mask PNG (indices).")
    parser.add_argument("--out", required=True, help="Output depth filename (All Flooded Areas).")
    parser.add_argument("--out-buildings", default=None, help="Output depth filename (Buildings Only).")
    parser.add_argument("--boundary-percentile", type=float, default=95.0, help="Percentile for water level.")
    args = parser.parse_args()

    compute_flood_depth(
        dem_path=args.dem,
        mask_path=args.mask,
        depth_out_path=args.out,
        building_out_path=args.out_buildings,
        boundary_percentile=args.boundary_percentile,
    )

if __name__ == "__main__":
    main()
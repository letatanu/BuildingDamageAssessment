#!/usr/bin/env python
"""
Compute approximate flood water depth from a DEM (.tif) and
a flood mask (PNG or TIF).

Updates:
  - New Output (_mask_vis.png):
      * Background (0) -> Green
      * Buildings -> Red
      * Flood -> Yellow
      * Boundary -> Orange
  - Restored Output (_boundary.png): Just Red lines on Black.
  - Retains all depth calculation fixes.

Usage example:
  python -m scripts.water_depth \
    --dem path/to/dem.tif \
    --mask path/to/mask.png \
    --out path/to/depth_out.png \
    --flood-labels 3 5 \
    --vis-max 5.0
"""

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import rasterio
from scipy import ndimage
import matplotlib.pyplot as plt

# ---------------- IO helpers ----------------

def depth_to_color(depth: np.ndarray, vmax: float | None = None) -> np.ndarray:
    """
    Map depth (in meters) to an RGB image.
    Scheme: Black (0) -> Dark Blue -> Bright Cyan
    """
    d = depth.copy()
    
    # Mask out strict 0.0 (Background)
    is_water = d > 0

    if vmax is None:
        positive = d[d > 0]
        if positive.size == 0:
            vmax = 1.0
        else:
            vmax = np.percentile(positive, 95)
    
    if vmax <= 0:
        vmax = 1.0

    d_norm = np.clip(d / vmax, 0.0, 1.0)
    h, w = d_norm.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    # Blue colormap (starts at Dark Blue)
    rgb[..., 2] = 255 * (0.3 + 0.7 * d_norm)      # B
    rgb[..., 1] = 255 * (0.2 + 0.8 * d_norm)      # G
    rgb[..., 0] = 255 * (0.0 + 0.2 * d_norm)      # R

    # Apply Background Mask (Black)
    rgb[~is_water] = 0

    return rgb.astype(np.uint8)


def dem_to_color(dem: np.ndarray) -> np.ndarray:
    """
    Map DEM elevation to RGB using the SAME Blue-Cyan scheme as depth.
    Low Elevation -> Dark Blue
    High Elevation -> Bright Cyan
    """
    d_min, d_max = dem.min(), dem.max()
    if d_max == d_min:
        d_norm = np.zeros_like(dem)
    else:
        d_norm = (dem - d_min) / (d_max - d_min)
    
    d_norm = np.clip(d_norm, 0, 1)
    
    h, w = d_norm.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    rgb[..., 2] = 255 * (0.3 + 0.7 * d_norm)      # B
    rgb[..., 1] = 255 * (0.2 + 0.8 * d_norm)      # G
    rgb[..., 0] = 255 * (0.0 + 0.2 * d_norm)      # R
    
    return rgb.astype(np.uint8)


def load_dem(dem_path: str):
    ds = rasterio.open(dem_path)
    arr = ds.read(1, masked=True)
    meta = ds.meta.copy()
    return arr, meta, ds.transform, ds.crs


def load_mask(mask_path: str) -> np.ndarray:
    p = Path(mask_path)
    suf = p.suffix.lower()
    if suf in [".tif", ".tiff"]:
        with rasterio.open(mask_path) as src:
            m = src.read(1)
        return m.astype(np.int64)
    img = Image.open(mask_path).convert("L")
    arr = np.array(img, dtype=np.int64)
    return arr


def save_depth(depth: np.ndarray, dem_meta: dict, out_path: str, vis_max: float = None):
    out_p = Path(out_path)
    suf = out_p.suffix.lower()

    if suf in [".tif", ".tiff"]:
        meta = dem_meta.copy()
        meta.update({"dtype": "float32", "count": 1, "nodata": 0.0})
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_p, "w", **meta) as dst:
            dst.write(depth.astype(np.float32), 1)
        print(f"[water_depth] Saved depth GeoTIFF → {out_p}")
        
    elif suf == ".png":
        import matplotlib.pyplot as plt
        rgb = depth_to_color(depth, vmax=vis_max)
        
        positive = depth[depth > 0]
        bar_max = vis_max if vis_max else (np.percentile(positive, 95) if positive.size > 0 else 1.0)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(rgb)
        ax.axis("off")

        cax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
        norm = plt.Normalize(vmin=0, vmax=bar_max)
        cmap = plt.cm.Blues
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                          cax=cax, orientation="vertical")
        cb.set_label("Depth (meters)", fontsize=12)

        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=150, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print(f"[water_depth] Saved color depth PNG → {out_p}")
    else:
        raise ValueError(f"Unsupported output extension: {suf}")


def save_boundary_vis(dem_shape, boundary_mask, out_path):
    """
    Restored Original: Red lines on Black background.
    """
    vis = np.zeros((dem_shape[0], dem_shape[1], 3), dtype=np.uint8)
    
    # Just Red Lines [255, 0, 0]
    vis[boundary_mask] = [255, 0, 0]
    
    p = Path(out_path)
    boundary_out = p.parent / (p.stem + "_boundary.png")
    Image.fromarray(vis).save(boundary_out)
    print(f"[water_depth] Saved boundary visualization → {boundary_out}")


def save_mask_vis(raw_mask, flood_labels, boundary_mask, out_path):
    """
    New Output: Multi-colored mask visualization.
    - Background (0) -> Green
    - Buildings (Non-Flood) -> Red
    - Flood -> Yellow
    - Boundary -> Orange
    """
    h, w = raw_mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 1. Background (0) -> Green [0, 255, 0]
    vis[:, :] = [0, 255, 0]

    # Identify classes
    is_flood = np.isin(raw_mask, flood_labels)
    is_background = (raw_mask == 0)
    # Buildings/Dryland = Not Flood AND Not Background
    is_building = (raw_mask == 1) | (raw_mask == 2)  # 1 and 2 are building classes

    # 2. Buildings -> Red [255, 0, 0]
    vis[is_building] = [255, 0, 0]

    # 3. Flood -> Yellow [255, 255, 0]
    vis[is_flood] = [255, 255, 0]

    # # 4. Boundary -> Orange [255, 165, 0] (Overlays Flood/Buildings)
    # vis[boundary_mask] = [255, 165, 0]
    
    p = Path(out_path)
    mask_out = p.parent / (p.stem + "_mask_vis.png")
    Image.fromarray(vis).save(mask_out)
    print(f"[water_depth] Saved mask visualization → {mask_out}")


def save_dem_vis(dem, out_path):
    rgb = dem_to_color(dem)
    p = Path(out_path)
    dem_out = p.parent / (p.stem + "_dem.png")
    Image.fromarray(rgb).save(dem_out)
    print(f"[water_depth] Saved DEM visualization → {dem_out}")


def fill_invalid_dem_nearest(dem_data, invalid_mask):
    if not np.any(invalid_mask):
        return dem_data
    from scipy.ndimage import distance_transform_edt
    indices = distance_transform_edt(invalid_mask, return_distances=False, return_indices=True)
    return dem_data[tuple(indices)]


# ------------- Core Algorithm ---------------

def compute_flood_depth(
    dem_path: str,
    mask_path: str,
    depth_out_path: str,
    flood_labels: list[int],
    min_boundary_pixels: int = 10,
    boundary_percentile: float = 95.0,
    vis_max: float = None
):
    print(f"[water_depth] DEM : {dem_path}")
    print(f"[water_depth] MASK: {mask_path}")

    # 1. Load DEM & Fix Holes
    dem_raw, dem_meta, _, _ = load_dem(dem_path)
    
    if np.ma.isMaskedArray(dem_raw):
        dem_data = dem_raw.data.copy()
        invalid_mask = dem_raw.mask
    else:
        dem_data = dem_raw.copy()
        invalid_mask = ~np.isfinite(dem_data)
        
    # Treat 0.0 as invalid/missing
    invalid_mask = invalid_mask | (~np.isfinite(dem_data)) | (dem_data == 0.0)

    if np.any(invalid_mask):
        print(f"[water_depth] Interpolating {np.sum(invalid_mask)} invalid DEM pixels...")
        dem_data = fill_invalid_dem_nearest(dem_data, invalid_mask)
    
    # 2. Load Mask
    mask = load_mask(mask_path)
    if mask.shape != dem_data.shape:
        print(f"[water_depth] WARNING: Resizing mask {mask.shape} to DEM {dem_data.shape}.")
        mask_img = Image.fromarray(mask.astype(np.int32))
        mask_img = mask_img.resize((dem_data.shape[1], dem_data.shape[0]), Image.NEAREST)
        mask = np.array(mask_img, dtype=np.int64)

    # 3. Identify Flood
    is_flood = np.isin(mask, flood_labels)
    
    # 4. Connected Components
    structure = ndimage.generate_binary_structure(2, 2)
    labels, n_labels = ndimage.label(is_flood, structure=structure)
    print(f"[water_depth] Found {n_labels} flooded region(s).")

    depth = np.zeros(dem_data.shape, dtype=np.float32)
    all_boundaries = np.zeros(dem_data.shape, dtype=bool)

    for lab in range(1, n_labels + 1):
        region = (labels == lab)
        if not np.any(region):
            continue

        # --- Attempt 1: Strict Dryland Boundary ---
        eroded_strict = ndimage.binary_erosion(region, structure=structure, border_value=1)
        strict_boundary = region & (~eroded_strict)
        
        boundary_elev = dem_data[strict_boundary]
        boundary_elev = boundary_elev[np.isfinite(boundary_elev)]

        final_boundary_mask = strict_boundary

        # --- Attempt 2: Fallback to Edge Boundary ---
        if boundary_elev.size == 0:
            print(f"[water_depth] Region {lab}: No dry-land boundary found. Attempting fallback to edges...")
            eroded_lax = ndimage.binary_erosion(region, structure=structure, border_value=0)
            lax_boundary = region & (~eroded_lax)
            
            boundary_elev_lax = dem_data[lax_boundary]
            boundary_elev_lax = boundary_elev_lax[np.isfinite(boundary_elev_lax)]
            
            if boundary_elev_lax.size > 0:
                print(f" -> Fallback successful. Using {boundary_elev_lax.size} edge pixels.")
                boundary_elev = boundary_elev_lax
                final_boundary_mask = lax_boundary
            else:
                print(" -> Fallback failed. Depth set to 0.")
                continue

        all_boundaries = all_boundaries | final_boundary_mask

        # Compute Water Level
        if boundary_elev.size >= min_boundary_pixels:
            water_level = np.percentile(boundary_elev, boundary_percentile)
        else:
            water_level = np.max(boundary_elev)

        region_dem = dem_data[region]
        region_depth = water_level - region_dem
        
        # Force min depth to 0.1m so "uphill" flood is visible
        region_depth = np.maximum(region_depth, 0.1)
        
        # Outlier Removal
        outliers = region_depth > 30.0
        if np.any(outliers):
            region_depth[outliers] = 0.1

        depth[region] = region_depth

    # 5. Save
    save_depth(depth, dem_meta, depth_out_path, vis_max=vis_max)
    
    # Save the standard Red boundary visualization
    save_boundary_vis(dem_data.shape, all_boundaries, depth_out_path)
    
    # Save the new Mask visualization (Green/Red/Yellow/Orange)
    save_mask_vis(mask, flood_labels, all_boundaries, depth_out_path)
    
    save_dem_vis(dem_data, depth_out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--flood-labels", type=int, nargs='+', default=[1])
    parser.add_argument("--boundary-percentile", type=float, default=95.0)
    parser.add_argument("--vis-max", type=float, default=None)
    args = parser.parse_args()

    compute_flood_depth(
        dem_path=args.dem,
        mask_path=args.mask,
        depth_out_path=args.out,
        flood_labels=args.flood_labels,
        boundary_percentile=args.boundary_percentile,
        vis_max=args.vis_max
    )


if __name__ == "__main__":
    main()
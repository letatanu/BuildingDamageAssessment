#!/usr/bin/env python3
"""
Downloads and aligns Copernicus 30m Global DEMs for xBD imagery.

Usage:
    python scripts/utils/download_xbd_dems.py \
        --xbd-root data/xBD/geotiffs \
        --split tier1 \
        --api-key 28411dcb1b4e26ddcd95574cacecc91f
"""

import os
import argparse
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import rasterio
from rasterio.warp import transform_bounds, reproject, Resampling
import tempfile

def download_and_align_dem(rgb_path, out_dem_path, api_key):
    """
    Reads the bounds of the xBD RGB image, downloads the corresponding DEM,
    and warps the DEM to exactly match the RGB image's dimensions and transform.
    """
    if os.path.exists(out_dem_path):
        return True

    try:
        # 1. Get bounds and CRS from the xBD image
        with rasterio.open(rgb_path) as src:
            rgb_crs = src.crs
            rgb_transform = src.transform
            rgb_width = src.width
            rgb_height = src.height
            bounds = src.bounds

            # Convert bounds to EPSG:4326 (Lat/Lon) for the API
            if rgb_crs.to_string() != "EPSG:4326":
                min_lon, min_lat, max_lon, max_lat = transform_bounds(rgb_crs, 'EPSG:4326', *bounds)
            else:
                min_lon, min_lat, max_lon, max_lat = bounds

        # Add a 0.01 degree buffer (~1km) to ensure the API doesn't fail on tiny bounding boxes
        buffer = 0.01
        min_lon, max_lon = min_lon - buffer, max_lon + buffer
        min_lat, max_lat = min_lat - buffer, max_lat + buffer

        # 2. Download from OpenTopography (COP30 = Copernicus 30m Global DEM)
        url = (
            f"https://portal.opentopography.org/API/globaldem"
            f"?demtype=COP30"
            f"&south={min_lat}&north={max_lat}&west={min_lon}&east={max_lon}"
            f"&outputFormat=GTiff"
            f"&API_Key={api_key}"
        )

        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return False

        # Save downloaded DEM to a temporary file
        fd, temp_dem_path = tempfile.mkstemp(suffix=".tif")
        os.close(fd)
        
        with open(temp_dem_path, 'wb') as f:
            f.write(response.content)

        # 3. Reproject and crop the downloaded DEM to perfectly match the xBD RGB image
        with rasterio.open(temp_dem_path) as dem_src:
            kwargs = dem_src.meta.copy()
            kwargs.update({
                'crs': rgb_crs,
                'transform': rgb_transform,
                'width': rgb_width,
                'height': rgb_height,
                'dtype': rasterio.float32,
                'nodata': 0.0
            })

            with rasterio.open(out_dem_path, 'w', **kwargs) as dem_dst:
                reproject(
                    source=rasterio.band(dem_src, 1),
                    destination=rasterio.band(dem_dst, 1),
                    src_transform=dem_src.transform,
                    src_crs=dem_src.crs,
                    dst_transform=rgb_transform,
                    dst_crs=rgb_crs,
                    resampling=Resampling.bilinear # Smooths the 30m pixels over the 0.3m grid
                )

        # Cleanup temp file
        os.remove(temp_dem_path)
        return True

    except Exception as e:
        if os.path.exists(temp_dem_path):
            os.remove(temp_dem_path)
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xbd-root", required=True, help="Path to xBD dataset")
    parser.add_argument("--split", default="tier1", choices=["tier1", "tier3", "test", "hold"])
    parser.add_argument("--api-key", required=True, help="OpenTopography API Key")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent downloads")
    args = parser.parse_args()

    img_dir = Path(args.xbd_root) / args.split / "images"
    
    # Create the dems/ directory inside the split (e.g., tier1/dems/)
    dem_dir = Path(args.xbd_root) / args.split / "dems"
    os.makedirs(dem_dir, exist_ok=True)

    post_files = sorted(img_dir.glob("*_post_disaster.tif"))
    print(f"Found {len(post_files)} images. Preparing to download DEMs...")

    tasks = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for f in post_files:
            # Output filename matches the xBD image name exactly
            out_dem = dem_dir / f"{f.stem}.tif"
            if not out_dem.exists():
                tasks[executor.submit(download_and_align_dem, str(f), str(out_dem), args.api_key)] = f.name

        successful = 0
        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Downloading DEMs"):
            if future.result():
                successful += 1

    print(f"\nDone! Successfully downloaded and aligned {successful}/{len(tasks)} DEMs.")
    print(f"Saved to: {dem_dir}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
tile_crasar_ultimate_fixed.py

Purpose:
    The "Holy Grail" script for CRASAR data fusion.
    
    FIXES:
    1. Contextily Error: Updated 'provider' -> 'source' for newer versions.
    2. Robust JSON Parsing: Handles both list/dict formats for coordinates.
    3. Satellite Registration: Warps Sat image to match Drone rotation.
    4. OSM Integration: Projects OSM vectors onto the Drone image.

    OUTPUTS:
    - images/      (Drone RGB)
    - sat_images/  (Aligned Satellite RGB)
    - osm_masks/   (OpenStreetMap Binary Mask)
    - labels/      (Damage Labels)
    - depth/       (Water Depth)

Prerequisites:
    pip install contextily rasterio shapely scikit-image transformers torch osmnx geopandas
"""

import argparse
import json
import os
import math
import warnings
from pathlib import Path
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds, from_gcps
from rasterio.control import GroundControlPoint
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import Window
from shapely.geometry import Polygon, box
from shapely.affinity import translate, scale, affine_transform
from tqdm import tqdm
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from scipy import ndimage
import contextily as cx 
import geopandas as gpd
import osmnx as ox

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
SAT_PROVIDER = cx.providers.Esri.WorldImagery
LABEL_TO_ID = {"no damage": 1, "minor damage": 2, "major damage": 3, "destroyed": 4, "un-classified": 255}
FLOODNET_WATER_IDS = [5, 3] 

# --- ROBUST GCP EXTRACTION ---

def flatten_coords(obj):
    """Recursively extracts numbers from nested lists/dicts."""
    coords = []
    if isinstance(obj, (int, float)):
        coords.append(obj)
    elif isinstance(obj, list):
        for item in obj:
            coords.extend(flatten_coords(item))
    elif isinstance(obj, dict):
        if 'x' in obj and 'y' in obj:
            coords.extend([obj['x'], obj['y']])
        elif 'lon' in obj and 'lat' in obj:
            coords.extend([obj['lon'], obj['lat']])
        else:
            for v in obj.values():
                coords.extend(flatten_coords(v))
    return coords

def extract_gcps_from_json(json_path):
    """Extracts matched pairs of (Pixel, Lat/Lon) to serve as GCPs."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        pixel_coords = []
        world_coords = []
        
        def find_pairs(obj):
            if isinstance(obj, dict):
                has_pixels = "pixels" in obj
                has_epsg = "EPSG:4326" in obj
                
                if has_pixels and has_epsg:
                    p_raw = obj["pixels"]
                    w_raw = obj["EPSG:4326"]
                    
                    p_arr = np.array(flatten_coords(p_raw))
                    w_arr = np.array(flatten_coords(w_raw))
                    
                    if len(p_arr) >= 6 and len(w_arr) >= 6:
                        px = np.mean(p_arr[0::2])
                        py = np.mean(p_arr[1::2])
                        wx = np.mean(w_arr[0::2]) # Lon
                        wy = np.mean(w_arr[1::2]) # Lat
                        
                        pixel_coords.append((px, py))
                        world_coords.append((wx, wy))
                
                for v in obj.values(): find_pairs(v)
            elif isinstance(obj, list):
                for v in obj: find_pairs(v)
        
        find_pairs(data)
        
        if len(pixel_coords) < 3:
            return None
            
        return pixel_coords, world_coords
        
    except Exception as e:
        print(f"GCP Extract Error for {json_path}: {e}")
        return None

def warp_satellite_and_fetch_osm(drone_path, json_path, sat_out_path, osm_out_path):
    """
    1. Calculates Transform using GCPs.
    2. Downloads & Warps Satellite Image.
    3. Downloads OSM Vectors.
    """
    if os.path.exists(sat_out_path) and os.path.exists(osm_out_path): 
        return True
    
    res = extract_gcps_from_json(json_path)
    if not res: return False
    px_list, wld_list = res
    
    try:
        lons = [p[0] for p in wld_list]
        lats = [p[1] for p in wld_list]
        w, s, e, n = min(lons), min(lats), max(lons), max(lats)
        buf = 0.002 # Buffer for rotation
        
        # --- DOWNLOAD OSM ---
        if not os.path.exists(osm_out_path):
            try:
                tags = {'building': True}
                gdf = ox.features_from_bbox(bbox=(n+buf, s-buf, e+buf, w-buf), tags=tags)
                if gdf.empty:
                    gdf = gpd.GeoDataFrame(columns=['geometry', 'building'], geometry='geometry')
                else:
                    gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
                gdf.to_file(osm_out_path, driver='GeoJSON')
            except:
                gdf = gpd.GeoDataFrame(columns=['geometry', 'building'], geometry='geometry')
                gdf.to_file(osm_out_path, driver='GeoJSON')

        # --- DOWNLOAD & WARP SATELLITE ---
        if not os.path.exists(sat_out_path):
            # FIX: Updated 'provider' -> 'source'
            sat_img, sat_ext = cx.bounds2img(w-buf, s-buf, e+buf, n+buf, source=SAT_PROVIDER, zoom=19, ll=True)
            
            sat_H, sat_W = sat_img.shape[0], sat_img.shape[1]
            src_transform = from_bounds(sat_ext[0], sat_ext[2], sat_ext[1], sat_ext[3], sat_W, sat_H)
            src_crs = "EPSG:3857"
            
            with rasterio.open(drone_path) as drone_src:
                dst_height = drone_src.height
                dst_width = drone_src.width
                
                gcps = []
                for (px, py), (lon, lat) in zip(px_list, wld_list):
                    xs, ys = transform_bounds("EPSG:4326", "EPSG:3857", lon, lat, lon, lat)
                    gcps.append(GroundControlPoint(row=py, col=px, x=xs[0], y=ys[0]))
                
                dst_transform = from_gcps(gcps)
                dst_crs = "EPSG:3857"
                
                aligned_sat = np.zeros((3, dst_height, dst_width), dtype=np.uint8)
                reproject(
                    source=np.moveaxis(sat_img, -1, 0),
                    destination=aligned_sat,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear
                )
                
                with rasterio.open(
                    sat_out_path, 'w', driver='GTiff',
                    height=dst_height, width=dst_width, count=3, dtype=np.uint8,
                    crs=dst_crs, transform=dst_transform
                ) as dst:
                    dst.write(aligned_sat)

        return True

    except Exception as e:
        print(f"Warp/DL Failed: {e}")
        return False

# --- STANDARD POLYGON LOAD ---
def load_polygons(json_path):
    entries = []
    try:
        data = json.loads(Path(json_path).read_text())
        def visit(x):
            if isinstance(x, dict):
                if "geometry" in x or "pixels" in x: entries.append(x)
                for v in x.values(): visit(v)
            elif isinstance(x, list):
                for v in x: visit(v)
        visit(data)
    except: return [], []

    polys, classes = [], []
    for e in entries:
        pix = e.get("pixels")
        pts = None
        if isinstance(pix, dict) and "geometry" in pix:
            pts = pix["geometry"].get("coordinates", [[]])[0]
        elif isinstance(pix, list):
            flat = flatten_coords(pix)
            if flat: pts = list(zip(flat[0::2], flat[1::2]))
                 
        if pts and len(pts) >= 3:
            lbl = e.get("label", "").lower().strip()
            cid = LABEL_TO_ID.get(lbl, 255)
            if cid != 255:
                p = Polygon(pts).buffer(0)
                if not p.is_empty:
                    polys.append(p)
                    classes.append(cid)
    return polys, classes

# --- FLOOD DEPTH ---
def compute_flood_depth_tile(dem, water_mask, boundary_percentile=95.0, min_boundary_pixels=5):
    depth = np.zeros_like(dem, dtype=np.float32)
    structure = ndimage.generate_binary_structure(2, 2)
    labels, n_labels = ndimage.label(water_mask, structure=structure)
    dem_valid = dem > -100
    
    for lab in range(1, n_labels + 1):
        region = (labels == lab)
        if not np.any(region): continue
        eroded = ndimage.binary_erosion(region, structure=structure, border_value=1)
        inner_boundary = region & (~eroded) & dem_valid
        boundary_elev = dem[inner_boundary]
        wse = 0
        if boundary_elev.size >= min_boundary_pixels:
             wse = np.percentile(boundary_elev, boundary_percentile)
        elif boundary_elev.size > 0:
             wse = np.max(boundary_elev)
        if boundary_elev.size > 0:
            region_depth = wse - dem[region]
            region_depth[region_depth < 0] = 0.0
            depth[region] = region_depth
    depth[~dem_valid] = 0.0
    return depth

# --- STAGE 1: SCANNING ---

def scan_scene(args):
    tif_path, anno_path, dem_path, tile_size, zoom, cache_dir = args
    candidates = []
    
    if not os.path.exists(tif_path) or not os.path.exists(anno_path): return []
    
    stem = Path(tif_path).stem
    sat_aligned_path = cache_dir / f"{stem}_sat_aligned.tif"
    osm_path = cache_dir / f"{stem}_osm.geojson"
    
    # Register & Download
    warp_satellite_and_fetch_osm(tif_path, anno_path, str(sat_aligned_path), str(osm_path))
    
    polys, _ = load_polygons(anno_path)
    if not polys: return []
    
    try:
        with rasterio.open(tif_path) as src:
            H, W = src.height, src.width
    except: return []
        
    read_size = int(tile_size * zoom)
    seen_centers = set()
    
    for p in polys:
        cx, cy = int(p.centroid.x), int(p.centroid.y)
        grid_step = tile_size // 2 
        grid_x, grid_y = cx // grid_step, cy // grid_step
        if (grid_x, grid_y) in seen_centers: continue
        seen_centers.add((grid_x, grid_y))
        
        col_off = cx - (read_size // 2)
        row_off = cy - (read_size // 2)
        
        if col_off < 0 or row_off < 0: continue
        if (col_off + read_size) > W or (row_off + read_size) > H: continue
            
        candidates.append({
            "tif": str(tif_path),
            "anno": str(anno_path),
            "dem": str(dem_path),
            "sat_aligned": str(sat_aligned_path),
            "osm": str(osm_path),
            "window": (col_off, row_off, read_size, read_size),
            "zoom": zoom
        })
    return candidates

# --- STAGE 2: GPU WORKER ---

def gpu_worker(worker_id, gpu_id, candidates, args, output_root):
    device = f"cuda:{gpu_id}"
    
    try:
        flood_proc = Mask2FormerImageProcessor.from_pretrained(args.floodnet_model)
        flood_model = Mask2FormerForUniversalSegmentation.from_pretrained(args.floodnet_model)
        flood_model.to(device).eval()
    except Exception as e:
        print(f"Model Error: {e}")
        return

    for p in ["images", "labels", "depth", "sat_images", "osm_masks"]: 
        os.makedirs(output_root / p, exist_ok=True)
    
    poly_cache = {} 
    osm_cache = {}
    
    for cand in tqdm(candidates, desc=f"GPU {gpu_id}", position=worker_id):
        tif_path = cand["tif"]
        dem_path = cand["dem"]
        sat_path = cand["sat_aligned"]
        osm_path = cand["osm"]
        
        if tif_path not in poly_cache:
            poly_cache[tif_path] = load_polygons(cand["anno"])
        scene_polys, scene_classes = poly_cache[tif_path]
        
        # Load OSM
        if osm_path and osm_path not in osm_cache:
            try:
                gdf = gpd.read_file(osm_path)
                if not gdf.empty:
                    gdf = gdf.to_crs("EPSG:3857")
                    osm_cache[osm_path] = list(gdf.geometry)
                else: osm_cache[osm_path] = []
            except: osm_cache[osm_path] = []
        scene_osm = osm_cache.get(osm_path, [])
        
        col, row, w, h = cand["window"]
        win = Window(col, row, w, h)
        tile_size = args.tile_size
        
        # A. Read Drone
        try:
            with rasterio.open(tif_path) as src:
                rgb = src.read([1, 2, 3], window=win, boundless=True, fill_value=0)
                rgb_img = Image.fromarray(np.moveaxis(rgb, 0, -1))
                if w != tile_size:
                    rgb_img = rgb_img.resize((tile_size, tile_size), resample=Image.BILINEAR)
        except: continue

        # B. Read Sat (Aligned) & Create OSM Mask
        sat_crop = Image.new("RGB", (tile_size, tile_size), (0,0,0))
        osm_mask_img = np.zeros((tile_size, tile_size), dtype=np.uint8)

        if sat_path and os.path.exists(sat_path):
            try:
                with rasterio.open(sat_path) as sat_src:
                    sat_data = sat_src.read([1, 2, 3], window=win, boundless=True, fill_value=0)
                    tmp_sat = Image.fromarray(np.moveaxis(sat_data, 0, -1))
                    if w != tile_size:
                        tmp_sat = tmp_sat.resize((tile_size, tile_size), resample=Image.BILINEAR)
                    sat_crop = tmp_sat
                    
                    inv_tf = ~sat_src.transform
                    mat = [inv_tf.a, inv_tf.b, inv_tf.d, inv_tf.e, inv_tf.c, inv_tf.f]
                    
                    scale_f = tile_size / w 
                    win_box = box(col, row, col+w, row+h)
                    
                    local_osm = []
                    for p in scene_osm:
                        px_poly = affine_transform(p, mat)
                        if px_poly.intersects(win_box):
                            clipped = px_poly.intersection(win_box)
                            shifted = translate(clipped, xoff=-col, yoff=-row)
                            final_poly = scale(shifted, xfact=scale_f, yfact=scale_f, origin=(0,0))
                            local_osm.append(final_poly)
                    
                    if local_osm:
                         m = rasterize([(g, 1) for g in local_osm], out_shape=(tile_size, tile_size), fill=0)
                         osm_mask_img = (m * 255).astype(np.uint8)
            except Exception: pass

        # C. Water/Depth
        f_inputs = flood_proc(images=rgb_img, return_tensors="pt").to(device)
        with torch.no_grad():
            f_out = flood_model(**f_inputs)
        f_seg = flood_proc.post_process_semantic_segmentation(f_out, target_sizes=[(tile_size, tile_size)])[0].cpu().numpy()
        water_mask = np.isin(f_seg, FLOODNET_WATER_IDS)
        
        depth_map = np.zeros((tile_size, tile_size), dtype=np.float32)
        if os.path.exists(dem_path) and np.sum(water_mask) > 100:
            try:
                with rasterio.open(dem_path) as dsrc:
                    dem = dsrc.read(1, window=win, boundless=True, fill_value=-9999)
                    if dem.shape != (tile_size, tile_size):
                         dem = np.array(Image.fromarray(dem).resize((tile_size, tile_size), resample=Image.NEAREST))
                    depth_map = compute_flood_depth_tile(dem.astype(np.float32), water_mask)
            except: pass 

        # D. Labels
        scale_f = tile_size / w 
        win_box = box(col, row, col+w, row+h)
        local_polys, local_classes = [], []
        for i, p in enumerate(scene_polys):
            if p.intersects(win_box):
                clipped = p.intersection(win_box)
                shifted = translate(clipped, xoff=-col, yoff=-row)
                final_poly = scale(shifted, xfact=scale_f, yfact=scale_f, origin=(0,0))
                local_polys.append(final_poly)
                local_classes.append(scene_classes[i])

        lbl_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
        for c in [1, 2, 3, 4]:
            shps = [g for i, g in enumerate(local_polys) if local_classes[i] == c]
            if shps:
                m = rasterize([(g, 1) for g in shps], out_shape=(tile_size, tile_size), fill=0)
                lbl_mask[m == 1] = c
        
        # Save
        fname = f"w{worker_id}_{cand['window'][0]}_{cand['window'][1]}"
        rgb_img.save(output_root / "images" / f"{fname}.png")
        sat_crop.save(output_root / "sat_images" / f"{fname}.png")
        Image.fromarray(osm_mask_img).save(output_root / "osm_masks" / f"{fname}.png")
        Image.fromarray(lbl_mask).save(output_root / "labels" / f"{fname}.png")
        with rasterio.open(output_root / "depth" / f"{fname}.tif", 'w', driver='GTiff', 
                           height=tile_size, width=tile_size, count=1, dtype=rasterio.float32) as dst:
            dst.write(depth_map, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--floodnet-model", default="nvidia/mask2former-swin-large-cityscapes-semantic")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--zoom", type=float, default=3.0)
    parser.add_argument("--scan-workers", type=int, default=16)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)
    
    root = Path(args.data_root)
    img_dir = root / args.split / "imagery" / "UAS"
    anno_dir = root / args.split / "annotations" / "building_damage_assessment"
    dem_dir = root / "dem" / args.split / "imagery" / "UAS"
    
    cache_dir = root / "fusion_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    files = sorted(list(img_dir.glob("*.tif")))
    
    print("--- Scanning & Registering Scenes ---")
    tasks = []
    for f in tqdm(files):
        if "mask" in f.name: continue
        an_path = anno_dir / f"{f.name}.json"
        d_path = dem_dir / f"{f.stem}_DEM.tif"
        if not d_path.exists(): d_path = dem_dir / f.name 

        if an_path.exists():
            tasks.append((str(f), str(an_path), str(d_path), args.tile_size, args.zoom, cache_dir))

    print("--- Generating Tiles ---")
    all_candidates = []
    with mp.Pool(args.scan_workers) as pool:
        for res in tqdm(pool.imap_unordered(scan_scene, tasks), total=len(tasks)):
            all_candidates.extend(res)
            
    print(f"Found {len(all_candidates)} tiles.")

    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    if num_gpus == 0: return

    total_workers = num_gpus * args.workers_per_gpu
    out_root = Path(args.out_dir) / args.split
    
    chunk_size = math.ceil(len(all_candidates) / total_workers)
    processes = []
    
    for i in range(total_workers):
        subset = all_candidates[i*chunk_size : (i+1)*chunk_size]
        if not subset: continue
        p = mp.Process(target=gpu_worker, args=(i, i//args.workers_per_gpu, subset, args, out_root))
        p.start()
        processes.append(p)
    
    for p in processes: p.join()

if __name__ == "__main__":
    main()
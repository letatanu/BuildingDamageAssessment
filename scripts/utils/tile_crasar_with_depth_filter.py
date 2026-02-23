#!/usr/bin/env python
"""
tile_crasar_water_filtered_multi_gpu.py

Purpose:
    Generate a complete training dataset (RGB + Label + Depth) for CRASAR.
    
    Features:
    - Parallel Multi-GPU processing.
    - Water Detection using pre-trained FloodNet Mask2Former.
    - Physics-based Water Depth calculation (Bathtub model).
      * UPDATE: Ignores water boundaries at the image edge to prevent false depth references.
    - Strict filtering (no black edges, centered on buildings).

Usage:
    python scripts/tile_crasar_water_filtered_multi_gpu.py \
        --data-root data/crarsar \
        --split train \
        --out-dir data/crarsar_water_filtered \
        --floodnet-model path/to/floodnet_checkpoint \
        --num-gpus 8 \
        --workers-per-gpu 2
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
from rasterio.windows import Window
from rasterio.features import rasterize
from shapely.geometry import Polygon, box
from shapely.affinity import translate, scale
from shapely.strtree import STRtree
from tqdm import tqdm
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from scipy import ndimage

# Suppress warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---

LABEL_TO_ID = {
    "no damage": 1, 
    "minor damage": 2, 
    "major damage": 3, 
    "destroyed": 4, 
    "un-classified": 255
}

# FloodNet ID Mapping: 5=Water, 3=Flooded Road
FLOODNET_WATER_IDS = [5, 3] 

# --- UTILS ---

def load_polygons(json_path):
    """Loads building polygons and their damage class from CRASAR JSON."""
    entries = []
    try:
        data = json.loads(json_path.read_text())
        def visit(x):
            if isinstance(x, dict):
                if "geometry" in x or "pixels" in x: entries.append(x)
                for v in x.values(): visit(v)
            elif isinstance(x, list):
                for v in x: visit(v)
        visit(data)
    except: 
        return [], []

    polys = []
    classes = []
    
    for e in entries:
        pix = e.get("pixels")
        pts = None
        if isinstance(pix, dict) and "geometry" in pix:
            pts = pix["geometry"].get("coordinates", [[]])[0]
        elif isinstance(pix, list) and len(pix) > 0:
            if isinstance(pix[0], list): pts = pix
            elif isinstance(pix[0], dict): pts = [(p['x'], p['y']) for p in pix]
            
        if pts and len(pts) >= 3:
            lbl = e.get("label", "").lower().strip()
            cid = LABEL_TO_ID.get(lbl, 255)
            if cid == 255: continue
            
            p = Polygon(pts).buffer(0)
            if not p.is_empty:
                polys.append(p)
                classes.append(cid)
                
    return polys, classes

# --- DEPTH LOGIC (Updated for Edge Handling) ---
def compute_flood_depth_tile(dem, water_mask, boundary_percentile=95.0, min_boundary_pixels=5):
    """
    Computes depth map for a tile using the 'Bathtub' model.
    dem: (H, W) float array
    water_mask: (H, W) boolean array
    """
    depth = np.zeros_like(dem, dtype=np.float32)
    
    # 1. Identify connected water bodies
    structure = ndimage.generate_binary_structure(2, 2)
    labels, n_labels = ndimage.label(water_mask, structure=structure)
    
    # Handle NoData in DEM (assuming < -100 is invalid/nodata)
    dem_valid = dem > -100
    
    for lab in range(1, n_labels + 1):
        region = (labels == lab)
        if not np.any(region): continue

        # 2. Find Inner Boundary (Water touching Dry)
        # UPDATE: border_value=1 ensures that pixels at the image edge are treated 
        # as touching "more water" (outside), not "dry land".
        # This prevents the image edge from being detected as a boundary.
        eroded = ndimage.binary_erosion(region, structure=structure, border_value=1)
        inner_boundary = region & (~eroded)
        
        # Only use boundary pixels where DEM is valid
        inner_boundary = inner_boundary & dem_valid
        
        boundary_elev = dem[inner_boundary]
        
        if boundary_elev.size == 0:
            # If no shoreline found within the tile (e.g. 100% water or water flows off edges)
            # we cannot estimate depth reliably from this tile alone.
            continue

        # 3. Determine Water Surface Elevation (WSE)
        if boundary_elev.size >= min_boundary_pixels:
            wse = np.percentile(boundary_elev, boundary_percentile)
        else:
            wse = np.max(boundary_elev)

        # 4. Calculate Depth
        region_dem = dem[region]
        region_depth = wse - region_dem
        region_depth[region_depth < 0] = 0.0
        
        # Assign to output
        depth[region] = region_depth

    # Ensure depth is 0 where DEM was invalid
    depth[~dem_valid] = 0.0
    
    return depth

# --- STAGE 1: SCANNING (CPU) ---

def scan_scene(args):
    tif_path, anno_path, dem_path, tile_size, zoom = args
    candidates = []
    
    if not os.path.exists(tif_path) or not os.path.exists(anno_path): return []

    polys, classes = load_polygons(Path(anno_path))
    if not polys: return []
    
    try:
        with rasterio.open(tif_path) as src:
            H, W = src.height, src.width
    except: return []
        
    read_size = int(tile_size * zoom)
    seen_centers = set()
    
    for i, p in enumerate(polys):
        cx, cy = int(p.centroid.x), int(p.centroid.y)
        
        # Grid snap
        grid_step = tile_size // 2 
        grid_x, grid_y = cx // grid_step, cy // grid_step
        if (grid_x, grid_y) in seen_centers: continue
        seen_centers.add((grid_x, grid_y))
        
        col_off = cx - (read_size // 2)
        row_off = cy - (read_size // 2)
        
        # Strict Bounds Check
        if col_off < 0 or row_off < 0: continue
        if (col_off + read_size) > W or (row_off + read_size) > H: continue
            
        candidates.append({
            "tif": str(tif_path),
            "anno": str(anno_path),
            "dem": str(dem_path),
            "window": (col_off, row_off, read_size, read_size),
            "zoom": zoom
        })
        
    return candidates

# --- STAGE 2: GPU WORKER ---

def gpu_worker(worker_id, gpu_id, candidates, args, output_root):
    device = f"cuda:{gpu_id}"
    
    # Load FloodNet Model
    try:
        processor = Mask2FormerImageProcessor.from_pretrained(args.floodnet_model)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(args.floodnet_model)
        model.to(device).eval()
    except Exception as e:
        print(f"[Worker {worker_id}] Error loading model: {e}")
        return

    # Create Dirs
    path_img = output_root / "images"
    path_lbl = output_root / "labels"
    path_dep = output_root / "depth"
    for p in [path_img, path_lbl, path_dep]: os.makedirs(p, exist_ok=True)
    
    poly_cache = {} 
    saved_count = 0
    
    iterator = tqdm(candidates, desc=f"GPU {gpu_id} (W{worker_id})", position=worker_id) if len(candidates) > 0 else []
    
    for cand in iterator:
        tif_path = cand["tif"]
        dem_path = cand["dem"]
        
        if tif_path not in poly_cache:
            poly_cache[tif_path] = load_polygons(Path(cand["anno"]))
        scene_polys, scene_classes = poly_cache[tif_path]
        if not scene_polys: continue

        col, row, w, h = cand["window"]
        win = Window(col, row, w, h)
        tile_size = args.tile_size
        
        # 1. READ RGB & CHECK BLACK
        try:
            with rasterio.open(tif_path) as src:
                rgb = src.read([1, 2, 3], window=win, boundless=True, fill_value=0)
                is_black = np.all(rgb == 0, axis=0)
                if (np.sum(is_black) / is_black.size) > 0.01: continue
                
                rgb_img = Image.fromarray(np.moveaxis(rgb, 0, -1))
                if w != tile_size:
                    rgb_img = rgb_img.resize((tile_size, tile_size), resample=Image.BILINEAR)
        except: continue

        # 2. PREDICT WATER (GPU)
        inputs = processor(images=rgb_img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        seg_map = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(tile_size, tile_size)]
        )[0].cpu().numpy()
        
        water_mask = np.isin(seg_map, FLOODNET_WATER_IDS)
        
        if np.sum(water_mask) < (tile_size * tile_size * 0.01):
            continue

        # 3. COMPUTE DEPTH (CPU/Numpy)
        depth_map = np.zeros((tile_size, tile_size), dtype=np.float32)
        
        if os.path.exists(dem_path):
            try:
                with rasterio.open(dem_path) as dsrc:
                    dem = dsrc.read(1, window=win, boundless=True, fill_value=-9999)
                    
                    if dem.shape != (tile_size, tile_size):
                         dem_pil = Image.fromarray(dem)
                         dem = np.array(dem_pil.resize((tile_size, tile_size), resample=Image.NEAREST))
                    
                    depth_map = compute_flood_depth_tile(dem.astype(np.float32), water_mask)

            except Exception:
                pass 

        # 4. GENERATE DAMAGE LABEL
        scale_f = tile_size / w 
        local_polys = []
        local_classes = []
        
        win_box = box(col, row, col+w, row+h)
        tree = STRtree(scene_polys)
        indices = tree.query(win_box)
        idx_list = indices if isinstance(indices, list) else list(indices)
        
        for idx in idx_list:
            p = scene_polys[idx]
            cid = scene_classes[idx]
            if p.intersects(win_box):
                clipped = p.intersection(win_box)
                shifted = translate(clipped, xoff=-col, yoff=-row)
                final_poly = scale(shifted, xfact=scale_f, yfact=scale_f, origin=(0,0))
                local_polys.append(final_poly)
                local_classes.append(cid)
        
        if not local_polys: continue

        lbl_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
        for c in [1, 2, 3, 4]:
            shps = [g for i, g in enumerate(local_polys) if local_classes[i] == c]
            if shps:
                m = rasterize([(g, 1) for g in shps], out_shape=(tile_size, tile_size), fill=0)
                lbl_mask[m == 1] = c
                
        # 5. SAVE
        fname = f"w{worker_id}_{saved_count:06d}"
        saved_count += 1
        
        rgb_img.save(path_img / f"{fname}.png")
        Image.fromarray(lbl_mask).save(path_lbl / f"{fname}.png")
        with rasterio.open(
            path_dep / f"{fname}.tif", 'w', driver='GTiff', 
            height=tile_size, width=tile_size, count=1, dtype=rasterio.float32
        ) as dst:
            dst.write(depth_map, 1)

# --- MAIN ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--floodnet-model", required=True)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--zoom", type=float, default=3.0)
    parser.add_argument("--scan-workers", type=int, default=32)
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--workers-per-gpu", type=int, default=8)
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)

    root = Path(args.data_root)
    img_dir = root / args.split / "imagery" / "UAS"
    dem_dir = root / "dem" / args.split / "imagery" / "UAS"
    anno_dir = root / args.split / "annotations" / "building_damage_assessment"
    
    files = sorted(list(img_dir.glob("*.tif")))
    tasks = []
    
    print(f"--- Stage 1: CPU Scanning ({len(files)} scenes) ---")
    for f in files:
        if "mask" in f.name: continue
        an_path = anno_dir / f"{f.name}.json"
        
        d_path = dem_dir / f.name 
        if not d_path.exists(): d_path = dem_dir / f"{f.stem}_DEM.tif"
        
        d_path_str = str(d_path) if d_path.exists() else ""
        
        if an_path.exists():
            tasks.append((str(f), str(an_path), d_path_str, args.tile_size, args.zoom))

    all_candidates = []
    with mp.Pool(args.scan_workers) as pool:
        for res in tqdm(pool.imap_unordered(scan_scene, tasks), total=len(tasks)):
            all_candidates.extend(res)
            
    print(f"Found {len(all_candidates)} candidates.")
    
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    if num_gpus == 0:
        print("Error: No GPUs detected!")
        return

    total_workers = num_gpus * args.workers_per_gpu
    print(f"--- Stage 2: Parallel Processing ({total_workers} workers) ---")
    
    out_root = Path(args.out_dir) / args.split
    chunk_size = math.ceil(len(all_candidates) / total_workers)
    processes = []
    
    for i in range(total_workers):
        assigned_gpu = i // args.workers_per_gpu 
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(all_candidates))
        subset = all_candidates[start_idx:end_idx]
        
        if not subset: continue
        
        p = mp.Process(target=gpu_worker, args=(i, assigned_gpu, subset, args, out_root))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

    print("Done.")

if __name__ == "__main__":
    main()
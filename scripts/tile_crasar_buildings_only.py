#!/usr/bin/env python
"""
tile_crasar_multi_count.py — Generates centered tiles for Hurricane events only (Train & Test sets included).
"""
from __future__ import annotations
import argparse, json, sys, os
import multiprocessing
from pathlib import Path
import numpy as np
from PIL import Image
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.enums import Resampling
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.affinity import translate, scale
from shapely.strtree import STRtree

# --- CONFIG ---
ANNO_SUBFOLDER = "building_damage_assessment"
LABEL_TO_ID = {"no damage": 1, "minor damage": 2, "major damage": 3, "destroyed": 4, "un-classified": 255}
BURN_ORDER = [1, 2, 3, 4]
COLOR_RGB = {0: (0,0,0), 1: (0,255,0), 2: (255,255,0), 3: (255,140,0), 4: (255,0,0), 255: (0,0,0)}

# --- HURRICANE FILTER LIST ---
# Expanded to include specific locations and dates found in your Train/Test file lists.
HURRICANE_KEYWORDS = [
    # 1. Generic Terms & Major Names
    "hurricane", "harvey", "michael", "ida", "laura", "ian", 
    "idalia", "ike", "irma", "maria", "matthew", "florence", 
    "katrina", "wilma", "sandy",

    # 2. Hurricane Ian (2022) Locations (Files starting 1001-, 1002-)
    "ft-myers", "fort-myers", "san-carlos", "summerlin", 
    "boca-grande", "palm-acers", "sanibel", "kelly-road",
    "harlem-heights", "iona-point", "kennedy-green", "palmeto-palms",

    # 3. Hurricane Michael (2018) Locations
    "mexicobeach", 

    # 4. Hurricane Idalia (2023) Locations
    "steinhatchee", "horseshoebeach", "suwannee", "cedarkey", "jena",

    # 5. Hurricane Ida (2021) Locations & Dates
    "cocodrie", "grandisle", "laplace", "lockport", "la-div",
    # (Ida dates in filename often appear as 20210831, 20210901, etc. which are caught if we look for patterns, 
    # but 'cocodrie' and 'la-div' cover most of the ones in your screenshot)

    # 6. Hurricane Harvey (2017) Locations
    "pecan-grove", "westpark", "sienna-village", "lancaster-canyon",

    # 7. Hurricane Laura (2020)
    # Your screenshot shows '0827-A-01'. 0827 (Aug 27) is Laura's landfall date.
    "0827-a", "0827-b" 
]

# --- WORKER INIT ---
def init_worker(shared_val):
    global counter_lock
    counter_lock = shared_val

# --- UTILS ---
def ensure_dir(p): p.mkdir(parents=True, exist_ok=True)
def class_id(lbl): return LABEL_TO_ID.get(str(lbl).strip().lower(), 255)

def to_xy_pairs(pix):
    if not pix: return None
    if isinstance(pix, dict):
        g = pix.get("geometry", pix)
        if isinstance(g, dict) and g.get("type", "").lower() == "polygon":
            return [(float(x), float(y)) for x, y in g.get("coordinates", [[]])[0]]
    if isinstance(pix, list) and pix:
        if isinstance(pix[0], (list, tuple)): return [(float(p[0]), float(p[1])) for p in pix]
        if isinstance(pix[0], dict): return [(float(d.get("x", d.get("X"))), float(d.get("y", d.get("Y")))) for d in pix]
    return None

def fix_poly(poly): return poly.buffer(0) if not poly.is_valid else poly

def apply_nearest_align(poly, A, V):
    if A.size == 0: return poly
    c = np.array([poly.centroid.x, poly.centroid.y])
    j = np.argmin(np.sum((A - c) ** 2, axis=1))
    vx, vy = V[j]
    return translate(poly, xoff=float(vx), yoff=float(vy))

def load_alignment_vectors(path: Path):
    try:
        segs = json.loads(path.read_text())
        A, V = [], []
        for (x0, y0), (x1, y1) in segs:
            A.append((float(x0), float(y0)))
            V.append((float(x1)-float(x0), float(y1)-float(y0)))
        return (np.asarray(A, float), np.asarray(V, float)) if A else (np.zeros((0, 2)), np.zeros((0, 2)))
    except:
        return (np.zeros((0, 2)), np.zeros((0, 2)))

# --- WORKER FUNCTION ---
def process_one_scene(args):
    # Unpack arguments
    tif_path, dem_path, entries, align_path, paths, th, tw, min_buildings, zoom, max_black_ratio = args
    
    filename = tif_path.name
    # Calculate read window size based on zoom
    read_w = int(tw * zoom)
    read_h = int(th * zoom)
    scale_factor = 1.0 / zoom

    # Load Alignment
    A, V = np.zeros((0, 2)), np.zeros((0, 2))
    if align_path and align_path.exists():
        A, V = load_alignment_vectors(align_path)

    try:
        with rasterio.open(tif_path) as src:
            # 1. Collect ALL Valid Polygons
            all_polys = []
            poly_class_map = [] 

            for e in entries:
                pts = to_xy_pairs(e.get("pixels"))
                if pts and len(pts) >= 3:
                    cid = class_id(e.get("label", ""))
                    if cid == 255: continue
                    
                    poly = fix_poly(apply_nearest_align(fix_poly(Polygon(pts)), A, V))
                    if not poly.is_empty: 
                        all_polys.append(poly)
                        poly_class_map.append(cid)
            
            if not all_polys: return

            tree = STRtree(all_polys)
            
            src_dem = rasterio.open(dem_path) if (dem_path and dem_path.exists()) else None
            saved_local = 0
            seen_centers = set()

            # Iterate by Building
            for i, target_poly in enumerate(all_polys):
                
                cx, cy = target_poly.centroid.x, target_poly.centroid.y
                cx_int, cy_int = int(cx), int(cy)
                
                if (cx_int, cy_int) in seen_centers: continue
                seen_centers.add((cx_int, cy_int))

                # Calculate Window
                col_off = cx_int - (read_w // 2)
                row_off = cy_int - (read_h // 2)
                
                win = Window(col_off, row_off, read_w, read_h)
                win_box = box(col_off, row_off, col_off + read_w, row_off + read_h)

                # --- STEP 1: Read Image & Check Black Background ---
                rgb = src.read(
                    indexes=[1,2,3], 
                    window=win, 
                    boundless=True, 
                    fill_value=0,
                    out_shape=(3, th, tw), 
                    resampling=Resampling.bilinear
                ) if src.count >= 3 else \
                        np.repeat(
                            src.read(
                                indexes=1, 
                                window=win, 
                                boundless=True, 
                                out_shape=(1, th, tw), 
                                resampling=Resampling.bilinear
                            ), 3, axis=0
                        )

                # Black Check
                is_black = np.all(rgb == 0, axis=0)
                if (np.sum(is_black) / is_black.size) > max_black_ratio:
                    continue

                # Query R-Tree
                query = tree.query(win_box)
                if isinstance(query, (list, np.ndarray)) and len(query) > 0 and isinstance(query[0], (int, np.integer)):
                        candidates_indices = query
                else: candidates_indices = list(query)
                
                buckets = {1: [], 2: [], 3: [], 4: []}
                unique_buildings_count = 0
                
                for idx in candidates_indices:
                    p = all_polys[idx]
                    p_cid = poly_class_map[idx]
                    
                    if p.intersects(win_box):
                        clipped = p.intersection(win_box)
                        if not clipped.is_empty:
                            unique_buildings_count += 1
                            
                            shifted = translate(clipped, xoff=-col_off, yoff=-row_off)
                            scaled_poly = scale(shifted, xfact=scale_factor, yfact=scale_factor, origin=(0,0))
                            
                            if scaled_poly.geom_type == 'MultiPolygon':
                                for g in scaled_poly.geoms: buckets[p_cid].append(g)
                            else: buckets[p_cid].append(scaled_poly)

                if unique_buildings_count < min_buildings:
                    continue

                # Burn Mask
                mask = np.zeros((th, tw), dtype=np.uint8)
                for cls in BURN_ORDER:
                    if buckets[cls]:
                        r_mask = rasterize([(g, 1) for g in buckets[cls]], out_shape=(th, tw), fill=0, dtype="uint8")
                        mask[r_mask == 1] = cls
                
                # Save Data
                with counter_lock.get_lock():
                    curr_id = counter_lock.value
                    counter_lock.value += 1
                
                saved_local += 1
                
                Image.fromarray(np.moveaxis(rgb, 0, -1).astype(np.uint8)).save(paths["img"] / f"{curr_id}.png")
                Image.fromarray(mask).save(paths["lab"] / f"{curr_id}_lab.png")
                
                vis = np.zeros((th, tw, 3), dtype=np.uint8)
                for cls, col in COLOR_RGB.items(): vis[mask == cls] = col
                Image.fromarray(vis).save(paths["vis"] / f"{curr_id}_vis.png")

                # DEM
                if src_dem:
                    d = src_dem.read(
                        1, 
                        window=win, 
                        boundless=True, 
                        fill_value=-9999,
                        out_shape=(th, tw), 
                        resampling=Resampling.bilinear
                    )
                    meta = src_dem.meta.copy()
                    win_transform = rasterio.windows.transform(win, src_dem.transform)
                    new_transform = win_transform * win_transform.scale(zoom, zoom)
                    meta.update(height=th, width=tw, transform=new_transform) 
                    
                    with rasterio.open(paths["dem"] / f"{curr_id}_dem.tif", "w", **meta) as dst: 
                        dst.write(d, 1)
            
            if src_dem: src_dem.close()

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        import traceback
        traceback.print_exc()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--start-id", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel processes")
    ap.add_argument("--min-buildings", type=int, default=1, help="Min number of distinct buildings required per tile")
    ap.add_argument("--tile-size", type=int, default=1024, help="Output tile size (pixels)")
    ap.add_argument("--zoom-factor", type=float, default=2.0, help="Zoom out factor (2.0 = 2x altitude)")
    ap.add_argument("--max-black-ratio", type=float, default=0.2, help="Max ratio of black pixels (0.0-1.0)")
    
    args = ap.parse_args()

    root = Path(args.data_root)
    img_dir = root / args.split / "imagery" / "UAS"
    anno_dir = root / args.split / "annotations" / ANNO_SUBFOLDER
    align_dir = root / args.split / "annotations" / "alignment_adjustments"
    dem_dir = root / "dem" / args.split / "imagery" / "UAS"
    
    out = Path(args.out_dir) / args.split
    paths = {k: out/v for k,v in [("img",f"{args.split}-org-img"), ("lab",f"{args.split}-label-img"), ("dem",f"{args.split}-dem"), ("vis","colored-annotations")]}
    for p in paths.values(): ensure_dir(p)

    # --- FILTERING FILES ---
    all_files = list(img_dir.glob("*.tif"))
    filtered_files = []
    
    print(f"Scanning {args.split}...")
    
    for f in all_files:
        if "mask" in f.name: continue
        
        fname_lower = f.name.lower()
        
        # Check if ANY known hurricane keyword is in the filename
        if any(k in fname_lower for k in HURRICANE_KEYWORDS):
            filtered_files.append(f)

    files = sorted(filtered_files)
    
    print(f"Total files found: {len(all_files)}")
    print(f"Files confirmed as Hurricane events: {len(files)}")
    
    if len(files) == 0:
        print("ERROR: No files matched the hurricane filter. Check your filenames!")
        sys.exit(1)

    print(f"Config: Size {args.tile_size}px | Zoom {args.zoom_factor}x | Black Filter >{args.max_black_ratio*100:.0f}%")

    work_items = []
    
    for tif in files:
        json_path = anno_dir / f"{tif.name}.json"
        
        entries = []
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text())
                def visit(x):
                    if isinstance(x, dict):
                        if "geometry" in x or "pixels" in x: entries.append(x)
                        for v in x.values(): visit(v)
                    elif isinstance(x, list):
                        for v in x: visit(v)
                visit(data)
            except: pass
        else:
            print(f"Warning: No JSON for {tif.name}")
            continue

        dem_path = dem_dir / tif.name.replace(".tif", "_DEM.tif")
        if not dem_path.exists(): dem_path = dem_dir / tif.name
        
        align_path = align_dir / f"{tif.name}.json" if align_dir.exists() else None

        work_items.append((
            tif, dem_path, entries, align_path, paths, 
            args.tile_size, args.tile_size, args.min_buildings, 
            args.zoom_factor, args.max_black_ratio
        ))

    shared_counter = multiprocessing.Value('i', args.start_id)

    print(f"Starting {args.workers} workers...")
    
    with multiprocessing.Pool(processes=args.workers, initializer=init_worker, initargs=(shared_counter,)) as pool:
        for _ in pool.imap_unordered(process_one_scene, work_items):
            pass

    print("Done.")

if __name__ == "__main__":
    main()
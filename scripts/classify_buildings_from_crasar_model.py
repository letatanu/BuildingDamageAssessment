#!/usr/bin/env python3
import os
import glob
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, rasterize
from shapely.geometry import box
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from PIL import Image

JPG_QUALITY_IMAGE = 90
JPG_QUALITY_OVERLAY = 88

def save_rgb_jpg(rgb_u8, path, quality):
    Image.fromarray(rgb_u8, mode="RGB").save(path, format="JPEG", quality=quality, optimize=True)

def save_mask_palette_png(mask_u8, path):
    # mask_u8 values are 0..4, store as paletted PNG
    im = Image.fromarray(mask_u8, mode="L").convert("P")
    # optional: define a palette (5 colors + pad to 256*3)
    palette = [
        0,0,0,        # 0 background
        0,255,0,      # 1
        255,255,0,    # 2
        255,165,0,    # 3
        255,0,0,      # 4
    ] + [0,0,0] * (256 - 5)
    im.putpalette(palette)
    im.save(path, format="PNG", optimize=True)  # optimize enables best compression 
# -----------------------
# CONFIG (EDIT THESE)
# -----------------------
TIF_DIR = "/media/volume/data_2d/semseg_2d/data/melissa_tif1"   # folder with *.tif
FOOTPRINTS_GEOJSON = "/media/volume/data_2d/semseg_2d/data/melissa_imagery/melissa_tif_building_masks1/osm_buildings.geojson"         # building polygons
MODEL_DIR = "/media/volume/data_2d/semseg_2d/runs/segformer_crarsar/checkpoint-4527"       # HF checkpoint folder/repo id
OUT_DIR = "/media/volume/data_2d/semseg_2d/data/melissa_imagery/segformer_building_damage_vis1"
os.makedirs(OUT_DIR, exist_ok=True)

# Sliding window inference (avoid OOM)
WIN = 1024
OVERLAP = 128
BATCH = 18
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True

# Only 4 categories
VALID_IDS = {1, 2, 3, 4}

# Visualization palette
# 0 background (transparent if you write RGBA), 1..4 damage classes
PALETTE = {
    0: (0, 0, 0),
    1: (240, 249, 33),   # No Damage  (yellow)
    2: (252, 148, 65),   # Minor      (orange)
    3: (204, 71, 120),   # Major      (magenta)
    4: (126, 3, 168),    # Destroyed  (purple)
}

OVERLAY_ALPHA = 0.45

# Filters
MIN_BUILDINGS_IN_TILE = 10          # skip if fewer footprints intersect
MIN_VALID_FRAC = 0.98              # skip if <98% of pixels are valid (nodata/black border tiles)
FALLBACK_MAX_BLACK_FRAC = 0.20     # if no mask exists, skip if >2% pixels are pure black (0,0,0)

# Vote settings
MIN_DOMINANCE = 0.35
ALL_TOUCHED = False

torch.backends.cudnn.benchmark = True


# -----------------------
# Sliding-window helpers
# -----------------------
def windows_grid(width, height, win, overlap):
    step = win - overlap
    for row0 in range(0, height, step):
        for col0 in range(0, width, step):
            h = min(win, height - row0)
            w = min(win, width - col0)
            yield Window(col_off=col0, row_off=row0, width=w, height=h)

def pad_patch(patch, target_h, target_w):
    h, w, c = patch.shape
    out = np.zeros((target_h, target_w, c), dtype=patch.dtype)
    out[:h, :w] = patch
    return out, (h, w)

def unpad_2d(arr2d, orig_hw):
    h, w = orig_hw
    return arr2d[:h, :w]

@torch.no_grad()
def run_batch(model, processor, batch_imgs):
    inputs = processor(images=batch_imgs, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    if DEVICE == "cuda" and USE_AMP:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(**inputs)
    else:
        outputs = model(**inputs)
    H, W = batch_imgs[0].shape[:2]
    preds = processor.post_process_semantic_segmentation(outputs, target_sizes=[(H, W)] * len(batch_imgs))
    return [p.detach().cpu().numpy().astype(np.int64) for p in preds]

def tiled_predict_class_ids(src, model, processor):
    H, W = src.height, src.width
    ramp = np.linspace(0, 1, WIN, dtype=np.float32)
    w1d = np.minimum(ramp, ramp[::-1])
    w2d = np.outer(w1d, w1d)

    acc = np.zeros((H, W), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)
    batch_imgs, batch_meta = [], []

    for win in tqdm(list(windows_grid(W, H, WIN, OVERLAP)), desc="Inference tiles", leave=False):
        patch = src.read([1, 2, 3], window=win)          # (3,h,w)
        patch = np.transpose(patch, (1, 2, 0))           # (h,w,3)
        if patch.dtype != np.uint8:
            patch = np.clip(patch, 0, 255).astype(np.uint8)

        patch_pad, orig_hw = pad_patch(patch, WIN, WIN)
        batch_imgs.append(patch_pad)
        batch_meta.append((win, orig_hw))

        if len(batch_imgs) == BATCH:
            preds = run_batch(model, processor, batch_imgs)
            for pred_pad, (win2, orig_hw2) in zip(preds, batch_meta):
                pred = unpad_2d(pred_pad, orig_hw2)
                r0, c0 = int(win2.row_off), int(win2.col_off)
                h, w = int(win2.height), int(win2.width)
                ww = w2d[:h, :w]
                acc[r0:r0+h, c0:c0+w] += pred.astype(np.float32) * ww
                wsum[r0:r0+h, c0:c0+w] += ww
            batch_imgs, batch_meta = [], []

    if batch_imgs:
        preds = run_batch(model, processor, batch_imgs)
        for pred_pad, (win2, orig_hw2) in zip(preds, batch_meta):
            pred = unpad_2d(pred_pad, orig_hw2)
            r0, c0 = int(win2.row_off), int(win2.col_off)
            h, w = int(win2.height), int(win2.width)
            ww = w2d[:h, :w]
            acc[r0:r0+h, c0:c0+w] += pred.astype(np.float32) * ww
            wsum[r0:r0+h, c0:c0+w] += ww

    wsum = np.maximum(wsum, 1e-6)
    return np.rint(acc / wsum).astype(np.int64)


# -----------------------
# Footprints + filtering
# -----------------------
def clip_footprints_to_tile(fp_wgs84, src):
    fp = fp_wgs84.to_crs(src.crs)
    b = src.bounds
    tile_geom = box(b.left, b.bottom, b.right, b.top)
    return fp[fp.intersects(tile_geom)].copy()

def get_valid_mask(src):
    """
    Returns HxW boolean valid-data mask if available.
    Rasterio dataset masks indicate valid pixels (non-nodata). [web:320]
    """
    try:
        m = src.dataset_mask()  # 0 nodata, 255 valid [web:320]
        if m is not None and m.size > 0:
            return (m > 0)
    except Exception:
        pass
    return None

def tile_has_black_background(rgb_u8, black_frac_thresh):
    black = np.all(rgb_u8 == 0, axis=-1)
    frac = float(black.mean())
    return frac > black_frac_thresh, frac

def building_vote_ignore_unclassified(pred_ids, transform, geom):
    H, W = pred_ids.shape
    inside = ~geometry_mask([geom], out_shape=(H, W), transform=transform, all_touched=False, invert=False)
    vals = pred_ids[inside]
    if vals.size == 0:
        return 0
    vals = vals[np.isin(vals, list(VALID_IDS))]
    if vals.size == 0:
        return 0
    ids, counts = np.unique(vals, return_counts=True)
    win_id = int(ids[np.argmax(counts)])
    dom = float(counts.max() / max(int(counts.sum()), 1))
    return win_id if dom >= MIN_DOMINANCE else 0


# -----------------------
# Visualization helpers
# -----------------------
def colorize_ids(ids2d, palette):
    rgb = np.zeros((ids2d.shape[0], ids2d.shape[1], 3), dtype=np.uint8)
    for k, col in palette.items():
        rgb[ids2d == k] = col
    return rgb

def save_rgba(rgb_u8, alpha_bool, out_path):
    a = (alpha_bool.astype(np.uint8) * 255)
    rgba = np.dstack([rgb_u8, a])
    Image.fromarray(rgba, mode="RGBA").save(out_path)

def overlay_rgb(base_rgb_u8, color_rgb_u8, mask_bool, alpha=0.45):
    base = base_rgb_u8.astype(np.float32)
    col = color_rgb_u8.astype(np.float32)
    m = mask_bool.astype(np.float32)[..., None]
    out = base * (1 - alpha * m) + col * (alpha * m)
    return np.clip(out, 0, 255).astype(np.uint8)


# -----------------------
# MAIN
# -----------------------
def main():
    tif_paths = sorted(glob.glob(os.path.join(TIF_DIR, "*.tif")))
    if not tif_paths:
        print(f"ERROR: No .tif found in {TIF_DIR}")
        return

    fp = gpd.read_file(FOOTPRINTS_GEOJSON)
    if fp.crs is None:
        fp = fp.set_crs("EPSG:4326")
    else:
        fp = fp.to_crs("EPSG:4326")
    fp = fp[["geometry"]].copy()

    processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_DIR).to(DEVICE).eval()

    print(f"Found {len(tif_paths)} tiles. Running on {DEVICE}.")

    with torch.inference_mode():
        for tif_path in tqdm(tif_paths, desc="Tiles"):
            tile_id = os.path.splitext(os.path.basename(tif_path))[0]
            out_tile_dir = os.path.join(OUT_DIR, tile_id)
            os.makedirs(out_tile_dir, exist_ok=True)

            with rasterio.open(tif_path) as src:
                # Quick footprint filter
                fp_tile = clip_footprints_to_tile(fp, src)
                if len(fp_tile) < MIN_BUILDINGS_IN_TILE:
                    os.rmdir(out_tile_dir)
                    continue

                # Read full-res RGB once (for outputs + black check)
                rgb = src.read([1, 2, 3])
                rgb = np.transpose(rgb, (1, 2, 0))
                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

                # Valid-data mask (preferred) to avoid black background artifacts [web:320]
                valid = get_valid_mask(src)
                if valid is not None:
                    valid_frac = float(valid.mean())
                    if valid_frac < MIN_VALID_FRAC:
                        os.rmdir(out_tile_dir)
                        continue
                else:
                    # fallback heuristic if no mask present
                    too_black, black_frac = tile_has_black_background(rgb, FALLBACK_MAX_BLACK_FRAC)
                    if too_black:
                        os.rmdir(out_tile_dir)
                        continue
                    valid = np.ones((src.height, src.width), dtype=bool)

                # Save image (RGBA with transparency outside valid area)
                save_rgb_jpg(rgb, os.path.join(out_tile_dir, f"{tile_id}_image.jpg"), quality=JPG_QUALITY_IMAGE)

                # Predict per-pixel ids
                pred_ids = tiled_predict_class_ids(src, model, processor)

                # Vote per building (ignore unclassified)
                keep_geoms, bld_classes = [], []
                for geom in fp_tile.geometry:
                    if geom is None or geom.is_empty:
                        continue
                    cls = building_vote_ignore_unclassified(pred_ids, src.transform, geom)
                    if cls == 0:
                        continue
                    keep_geoms.append(geom)
                    bld_classes.append(cls)

                if not keep_geoms:
                    continue

                # Rasterize building-level class mask (0 background, 1..4 buildings) [web:237]
                bld_ids = rasterize(
                    shapes=list(zip(keep_geoms, bld_classes)),
                    out_shape=(src.height, src.width),
                    transform=src.transform,
                    fill=0,
                    dtype=np.uint8,
                    all_touched=ALL_TOUCHED,
                )

                # Apply valid mask so nodata stays 0 in mask
                bld_ids = np.where(valid, bld_ids, 0).astype(np.uint8)

                # Save mask PNG (single-channel)
                Image.fromarray(bld_ids).save(os.path.join(out_tile_dir, f"{tile_id}_mask.png"))

                # Save overlay (RGBA; transparent outside valid, and only buildings colored)
                bld_rgb = colorize_ids(bld_ids, PALETTE)
                over = overlay_rgb(rgb, bld_rgb, (bld_ids != 0), alpha=OVERLAY_ALPHA)
                save_rgb_jpg(over, os.path.join(out_tile_dir, f"{tile_id}_overlay.jpg"), quality=JPG_QUALITY_OVERLAY)
            if os.path.isdir(out_tile_dir) and not os.listdir(out_tile_dir):
                os.rmdir(out_tile_dir)

    print("Done.")
    print("Output folder:", OUT_DIR)

if __name__ == "__main__":
    main()
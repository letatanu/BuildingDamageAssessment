import os
import glob
import json
import numpy as np
from PIL import Image, ImageDraw
import shapely.wkt
import tifffile as tiff
from concurrent.futures import ProcessPoolExecutor, as_completed

XBD_CLASS_MAPPING = {
    "un-classified": 1,
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4
}

VIS_COLORS = {
    0: [0, 0, 0],         
    1: [0, 255, 0],       
    2: [255, 255, 0],     
    3: [255, 165, 0],     
    4: [255, 0, 0]        
}

def poly_to_mask(json_path, shape):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    mask = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(mask)
    
    if 'features' in data and 'xy' in data['features']:
        for feat in data['features']['xy']:
            wkt = feat['wkt']
            subtype = feat['properties'].get('subtype', 'un-classified')
            
            class_val = XBD_CLASS_MAPPING.get(subtype, 0)
            if class_val == 0:
                continue
                
            poly = shapely.wkt.loads(wkt)
            if poly.geom_type == 'Polygon':
                x, y = poly.exterior.coords.xy
                draw.polygon(list(zip(x, y)), fill=class_val)
            elif poly.geom_type == 'MultiPolygon':
                for p in poly.geoms:
                    x, y = p.exterior.coords.xy
                    draw.polygon(list(zip(x, y)), fill=class_val)
                    
    return np.array(mask)

def calculate_iou(boxA, boxB):
    yA0, yA1, xA0, xA1 = boxA
    yB0, yB1, xB0, xB1 = boxB

    x_left = max(xA0, xB0)
    y_top = max(yA0, yB0)
    x_right = min(xA1, xB1)
    y_bottom = min(yA1, yB1)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    boxA_area = (xA1 - xA0) * (yA1 - yA0)
    boxB_area = (xB1 - xB0) * (yB1 - yB0)

    return intersection_area / float(boxA_area + boxB_area - intersection_area)

def get_crop_bounds(cx, cy, img_w, img_h, th, tw):
    y0 = int(cy - (th / 2))
    y1 = y0 + th
    x0 = int(cx - (tw / 2))
    x1 = x0 + tw
    
    if y0 < 0:
        y0, y1 = 0, th
    elif y1 > img_h:
        y0, y1 = img_h - th, img_h
        
    if x0 < 0:
        x0, x1 = 0, tw
    elif x1 > img_w:
        x0, x1 = img_w - tw, img_w
        
    return y0, y1, x0, x1

def generate_visualization(img_tile, mask_tile, alpha=0.5):
    h, w = mask_tile.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.float32)
    
    for class_val, color in VIS_COLORS.items():
        if class_val == 0:
            continue
        mask_rgb[mask_tile == class_val] = color
        
    vis_img = img_tile.copy().astype(np.float32)
    mask_pixels = mask_tile > 0
    
    vis_img[mask_pixels] = (vis_img[mask_pixels] * (1 - alpha)) + (mask_rgb[mask_pixels] * alpha)
    return vis_img.astype(np.uint8)

def process_single_file(tif_path, output_dir, split, tile_size, iou_threshold=0.5):
    base_name = os.path.basename(tif_path).replace(".tif", "")
    is_pre = "pre_disaster" in base_name
    sub_split = f"{split}" if is_pre else f"{split}_post"
    sub_split = "train" if split == "tier1" else "test"
    sub_split = f"{sub_split}_pre" if is_pre else f"{sub_split}"    
        
    out_dir_img = os.path.join(output_dir, sub_split, f"{sub_split}-org-img")
    os.makedirs(out_dir_img, exist_ok=True)
    
    # Only create label directories if it's a post-disaster image
    if not is_pre:
        out_dir_lbl = os.path.join(output_dir, sub_split, f"{sub_split}-label-img")
        out_dir_vis = os.path.join(output_dir, sub_split, f"{sub_split}-label-vis") 
        os.makedirs(out_dir_lbl, exist_ok=True)
        os.makedirs(out_dir_vis, exist_ok=True)
    
    parent_dir = os.path.dirname(os.path.dirname(tif_path))
    json_path = os.path.join(parent_dir, "labels", f"{base_name}.json")
    
    if not os.path.exists(json_path):
        return f"Skipped {base_name}: No JSON label found."
        
    img_array = tiff.imread(tif_path)
    
    if len(img_array.shape) == 3 and img_array.shape[0] < 10:
        img_array = np.transpose(img_array, (1, 2, 0))
        
    if img_array.dtype != np.uint8:
        if img_array.max() > 255:
            img_array = (img_array / img_array.max() * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)

    img_h, img_w = img_array.shape[:2]
    mask_array = poly_to_mask(json_path, (img_h, img_w))
    th, tw = tile_size
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    crop_boxes = []
    if 'features' in data and 'xy' in data['features']:
        for feat in data['features']['xy']:
            wkt = feat['wkt']
            poly = shapely.wkt.loads(wkt)
            cx, cy = poly.centroid.x, poly.centroid.y
            box = get_crop_bounds(cx, cy, img_w, img_h, th, tw)
            crop_boxes.append(box)
            
    final_crops = []
    for box in crop_boxes:
        is_duplicate = False
        for kept_box in final_crops:
            if calculate_iou(box, kept_box) > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            final_crops.append(box)
            
    tiles_created = 0
    for idx, (y0, y1, x0, x1) in enumerate(final_crops):
        img_tile = img_array[y0:y1, x0:x1]
        mask_tile = mask_array[y0:y1, x0:x1]
        
        if img_tile.shape[0] != th or img_tile.shape[1] != tw:
            continue
            
        if not np.any(mask_tile > 0):
            continue
            
        tile_name = f"{base_name}_crop_{idx}"
        img_out_path = os.path.join(out_dir_img, f"{tile_name}.jpg") 
        
        if len(img_tile.shape) == 3 and img_tile.shape[2] == 4:
            img_tile = img_tile[:, :, :3]
            
        # 1. ALWAYS Save Original Image (JPEG)
        Image.fromarray(img_tile).save(img_out_path, format="JPEG", quality=100) 
        
        # 2. ONLY Save Labels and Visualizations for POST_DISASTER
        if not is_pre:
            lbl_out_path = os.path.join(out_dir_lbl, f"{tile_name}_lab.png") 
            vis_out_path = os.path.join(out_dir_vis, f"{tile_name}_vis.jpg") 
            
            Image.fromarray(mask_tile).save(lbl_out_path, format="PNG")
            
            vis_tile = generate_visualization(img_tile, mask_tile, alpha=0.5)
            Image.fromarray(vis_tile).save(vis_out_path, format="JPEG", quality=90)
        
        tiles_created += 1
            
    return f"Processed {base_name}: {tiles_created} tiles."

def process_dataset_parallel(xbd_root, output_dir, split="train", tile_size=(512, 512), max_workers=None):
    img_dir = os.path.join(xbd_root, split, "images")
    tif_files = glob.glob(os.path.join(img_dir, "*.tif"))
    tif_files = [f for f in tif_files if "hurricane" in os.path.basename(f).lower()]
    
    if not tif_files:
        print(f"No hurricane .tif files found in {img_dir}.")
        return

    print(f"Centering tiles for {len(tif_files)} {split} hurricane images...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_file, tif, output_dir, split, tile_size)
            for tif in tif_files
        ]
        for future in as_completed(futures):
            try:
                print(future.result())
            except Exception as exc:
                print(f"Exception generated: {exc}")

if __name__ == "__main__":
    XBD_RAW_ROOT = "data/xBD_raw/geotiffs"
    OUTPUT_DATASET_ROOT = "data/xBD_tiled"
    
    process_dataset_parallel(XBD_RAW_ROOT, OUTPUT_DATASET_ROOT, split="tier1", tile_size=(512, 512))

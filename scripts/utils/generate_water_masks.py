import argparse
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

# FloodNet Class Mapping (ID 5 is usually Water)
# 0:Bg, 1:Bldg-Flood, 2:Bldg-NoFlood, 3:Road-Flood, 4:Road-NoFlood, 5:Water, ...
FLOODNET_WATER_ID = 5 
FLOODNET_ROAD_FLOOD_ID = 3 # Optional: Include flooded roads as water context

def generate_masks(args):
    # 1. Load Model Trained on FloodNet
    processor = Mask2FormerImageProcessor.from_pretrained(args.model_path)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model_path)
    model.eval().cuda()

    os.makedirs(args.output_dir, exist_ok=True)
    
    image_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(('.png', '.jpg', '.tif'))])

    print(f"Generating water masks for {len(image_files)} images...")

    for fname in tqdm(image_files):
        img_path = os.path.join(args.input_dir, fname)
        image = Image.open(img_path).convert("RGB")
        
        # Inference
        inputs = processor(images=image, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process to get semantic map
        # Target size = original image size
        pred_map = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0] # (H, W)
        
        pred_np = pred_map.cpu().numpy()
        
        # Extract Binary Water Mask
        # We consider 'Water' (5) and 'Road-Flooded' (3) as water surface
        water_mask = np.isin(pred_np, [FLOODNET_WATER_ID, FLOODNET_ROAD_FLOOD_ID]).astype(np.uint8) * 255
        
        # Save
        out_name = os.path.splitext(fname)[0] + ".png"
        Image.fromarray(water_mask).save(os.path.join(args.output_dir, out_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path to a checkpoint you trained on FloodNet (or a HuggingFace hub model)
    parser.add_argument("--model-path", type=str, required=True, help="Path to FloodNet-trained Mask2Former")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to CRASAR images")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to save generated water masks")
    args = parser.parse_args()
    generate_masks(args)
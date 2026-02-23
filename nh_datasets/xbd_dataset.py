import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape
import albumentations as A
from albumentations.pytorch import ToTensorV2

SUBTYPE_TO_CLASS = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 0,
}

class XBDSiameseDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=512, augment=True):
        """
        root_dir: path to xBD/geotiffs/
        split: 'train' (uses tier1), 'tier3', 'test', or 'hold'
        """
        self.img_size = img_size
        
        # --- PATH CORRECTION LOGIC ---
        # Map 'train' to 'tier1' (standard xBD practice for training)
        # You can also combine tier1+tier3 if you want more data
        if split == "train":
            # For simplicity, we just use tier1. 
            # To use both, you'd need to list multiple dirs.
            target_dirs = ["tier1"]
        elif split == "val":
            target_dirs = ["tier3"] # Use tier3 for validation if desired
        else:
            target_dirs = [split] # 'test', 'hold', 'tier3'

        self.samples = []
        for d in target_dirs:
            # Construct path: root_dir/tier1/images
            images_dir = os.path.join(root_dir, d, "images")
            labels_dir = os.path.join(root_dir, d, "labels")

            # Find all post-disaster images
            # NOTE: Your screenshot shows .tif extension.
            # Some xBD versions use .png. We check for .tif first.
            post_imgs = sorted(glob.glob(os.path.join(images_dir, "*_post_disaster.tif")))
            
            if len(post_imgs) == 0:
                # Fallback to PNG if TIF not found
                post_imgs = sorted(glob.glob(os.path.join(images_dir, "*_post_disaster.png")))

            for post_path in post_imgs:
                # Construct pre-disaster path
                # e.g. guatemala-volcano_00000003_post_disaster.tif 
                # -> guatemala-volcano_00000003_pre_disaster.tif
                pre_path = post_path.replace("_post_disaster", "_pre_disaster")
                
                # Construct label path
                # e.g. ../labels/guatemala-volcano_00000003_post_disaster.json
                basename = os.path.basename(post_path)
                label_name = basename.replace(".tif", ".json").replace(".png", ".json")
                label_path = os.path.join(labels_dir, label_name)

                if os.path.exists(pre_path) and os.path.exists(label_path):
                    self.samples.append((pre_path, post_path, label_path))
        
        print(f"Loaded {len(self.samples)} samples for split='{split}' from {target_dirs}")

        if augment:
            self.transform = A.Compose([
                # 1. RandomCrop: Takes a 512x512 patch from the 1024x1024 image
                # This preserves original resolution (buildings stay "big")
                A.RandomCrop(height=img_size, width=img_size),
                
                # 2. Standard augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ], additional_targets={"image2": "image"})
        else:
            # Validation: Resize or CenterCrop?
            # Resizing 1024 -> 512 loses small damage details.
            # Better to keep 1024 if memory allows, or CenterCrop.
            # If your GPU can handle 1024x1024 input, simply Normalize & ToTensor.
            # If not, use CenterCrop(512, 512).
            self.transform = A.Compose([
                # Option A: Full resolution (Best for accuracy, heavy on VRAM)
                # A.NoOp(), 
                
                # Option B: Center Crop (Focuses on center, might miss edge buildings)
                A.CenterCrop(height=img_size, width=img_size),

                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ], additional_targets={"image2": "image"})


    def __len__(self):
        return len(self.samples)

    def _load_tif(self, path):
        with rasterio.open(path) as src:
            img = src.read()  # (C, H, W)
        # Take first 3 bands (RGB) if more exist
        img = img[:3]
        img = np.transpose(img, (1, 2, 0))  # (H, W, C)
        return img.astype(np.uint8)

    def _build_mask(self, label_path, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        try:
            with open(label_path, 'r') as f:
                data = json.load(f)
            
            shapes = []
            # xBD JSONs usually have features -> xy -> wkt
            # Check structure based on your provided file
            features = data.get('features', {}).get('xy', [])
            
            for feat in features:
                subtype = feat['properties'].get('subtype', 'no-damage')
                cls_id = SUBTYPE_TO_CLASS.get(subtype, 0)
                if cls_id == 0: continue
                
                wkt_str = feat.get('wkt')
                if wkt_str:
                    try:
                        from shapely import wkt
                        geom = wkt.loads(wkt_str)
                        shapes.append((geom, cls_id))
                    except:
                        pass
            
            if shapes:
                mask = rasterize(shapes, out_shape=(h, w), fill=0, dtype=np.uint8)
                
        except Exception as e:
            print(f"Error parsing label {label_path}: {e}")
            
        return mask

    # def __getitem__(self, idx):
    #     pre_path, post_path, label_path = self.samples[idx]
        
    #     pre_img = self._load_tif(pre_path)
    #     post_img = self._load_tif(post_path)
    #     h, w = post_img.shape[:2]
        
    #     mask = self._build_mask(label_path, h, w)
        
    #     aug = self.transform(image=post_img, image2=pre_img, mask=mask)
    #     return aug['image2'], aug['image'], aug['mask'].long()
    def __getitem__(self, idx):
        pre_p, post_p, lbl_p = self.samples[idx]
        with rasterio.open(pre_p) as s: 
            # Read, take first 3 bands, transpose to HWC
            pre = s.read()[:3].transpose(1, 2, 0)
            # FORCE CAST TO UINT8 (or float32 / 255.0)
            if pre.dtype != np.uint8:
                # If 16-bit, normalize and cast. Or just cast if range is 0-255.
                # xBD is usually 0-255 range but stored weirdly sometimes.
                # Safer: Normalize to 0-1 float32 if you're unsure of range
                # but standard albumentations expects uint8 [0,255] or float [0,1].
                
                # Simple cast (assuming values are already 0-255)
                pre = pre.astype(np.uint8) 

        with rasterio.open(post_p) as s: 
            post = s.read()[:3].transpose(1, 2, 0)
            if post.dtype != np.uint8:
                post = post.astype(np.uint8)


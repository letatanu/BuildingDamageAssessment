import os
import argparse
import torch
import torch.nn as nn
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import DetrForObjectDetection, DetrImageProcessor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from PIL import Image
from tqdm import tqdm

# --- Constants ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_args():
    parser = argparse.ArgumentParser(description="Train RGB-D DETR for Building Detection")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4) # DETR is heavy, use small batch
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    return parser.parse_args()

def get_rgbd_detr(num_labels):
    """
    Loads DETR and modifies the first layer of the ResNet backbone 
    to accept 4 channels (RGB + Depth).
    """
    model_name = "facebook/detr-resnet-50"
    model = DetrForObjectDetection.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    
    # Access the first conv layer of the ResNet backbone
    # Structure: model.model.backbone.conv_encoder.model.embedder.embedder.convolution
    # (The path can vary slightly by version, we use the standard HF path)
    backbone = model.model.backbone.conv_encoder.model
    old_conv = backbone.embedder.embedder.convolution
    
    # Create new 4-channel Conv layer
    new_conv = nn.Conv2d(
        in_channels=4, 
        out_channels=old_conv.out_channels, 
        kernel_size=old_conv.kernel_size, 
        stride=old_conv.stride, 
        padding=old_conv.padding, 
        bias=False
    )
    
    # Initialize weights
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        # Avg of RGB for the depth channel
        new_conv.weight[:, 3] = torch.mean(old_conv.weight, dim=1)
        
    # Replace
    backbone.embedder.embedder.convolution = new_conv
    model.config.num_channels = 4
    
    return model

class RGBDBoxDataset(Dataset):
    def __init__(self, root_dir, split, processor):
        self.split_dir = os.path.join(root_dir, split)
        self.img_dir = os.path.join(self.split_dir, "images")
        self.mask_dir = os.path.join(self.split_dir, "labels")
        self.depth_dir = os.path.join(self.split_dir, "depth")
        self.processor = processor
        
        valid_exts = ('.png', '.jpg', '.jpeg', '.tif')
        self.images = sorted([f for f in os.listdir(self.img_dir) if f.endswith(valid_exts)])
        self.masks = sorted([f for f in os.listdir(self.mask_dir) if f.endswith(valid_exts)])
        self.depths = sorted([f for f in os.listdir(self.depth_dir) if f.endswith(valid_exts)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Load Data
        img = Image.open(os.path.join(self.img_dir, self.images[idx])).convert("RGB")
        depth = Image.open(os.path.join(self.depth_dir, self.depths[idx]))
        mask = np.array(Image.open(os.path.join(self.mask_dir, self.masks[idx])))

        # 2. Extract Boxes from Mask
        boxes = []
        labels = []
        
        # Identify object classes in this image (skip 0)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]

        for class_id in obj_ids:
            # Binary mask for this class
            binary_mask = (mask == class_id).astype(np.uint8)
            # Find blobs
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                # Filter tiny noise
                if w < 5 or h < 5: continue
                
                # Format: [x_min, y_min, x_max, y_max]
                boxes.append([x, y, x+w, y+h])
                labels.append(int(class_id) - 1) # Map 1-4 to 0-3 for model

        # 3. Prepare Inputs
        # Resize depth to match image if needed
        if depth.size != img.size:
            depth = depth.resize(img.size, Image.NEAREST)
            
        img_np = np.array(img)
        depth_np = np.array(depth)
        
        # Merge RGB + Depth -> 4 channels
        if len(depth_np.shape) == 2:
            depth_np = depth_np[:, :, np.newaxis]
        
        # Normalize Depth roughly 0-255 like RGB for processor
        # (Processor expects uint8 usually, or we handle float manually)
        # Here we stack first.
        rgbd_np = np.concatenate([img_np, depth_np], axis=2) # [H, W, 4]
        
        # 4. Use Processor
        # We trick the processor: passing 4-channel image might warn, but usually works 
        # if we handle the tensor conversion correctly. 
        # Actually, simpler: Pass RGB to processor to get 'pixel_values', 
        # then manually append normalized depth channel.
        
        encoding = self.processor(
            images=img,
            annotations={'image_id': idx, 'annotations': self._format_anns(boxes, labels)},
            return_tensors="pt"
        )
        
        # Get processed RGB tensor [1, 3, H, W]
        rgb_tensor = encoding['pixel_values'].squeeze(0)
        
        # Process Depth Manually to match RGB size
        # Processor resized RGB, we must resize Depth to match output tensor size
        target_h, target_w = rgb_tensor.shape[1], rgb_tensor.shape[2]
        
        # Resize depth using Torch
        depth_tensor = torch.tensor(np.array(depth).astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
        depth_resized = torch.nn.functional.interpolate(depth_tensor, size=(target_h, target_w), mode='nearest')
        depth_resized = depth_resized.squeeze(0) # [1, H, W]
        
        # Normalize depth with same mean/std logic or keep simple 0-1. 
        # Let's subtract 0.5 for simple centering
        depth_resized = (depth_resized - 0.5) / 0.5
        
        # Concatenate
        rgbd_tensor = torch.cat([rgb_tensor, depth_resized], dim=0) # [4, H, W]
        
        return {
            "pixel_values": rgbd_tensor, 
            "labels": encoding['labels'][0] # Dict with 'boxes' and 'class_labels'
        }

    def _format_anns(self, boxes, labels):
        anns = []
        for i, box in enumerate(boxes):
            anns.append({
                'id': i, 'image_id': 0, 'category_id': labels[i],
                'bbox': [box[0], box[1], box[2]-box[0], box[3]-box[1]], # xywh for processor
                'area': (box[2]-box[0])*(box[3]-box[1]), 'iscrowd': 0
            })
        return anns

def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # DETR processor pads images to max size in batch
    # We must replicate that logic for 4-channel tensors manually or use a simple stack 
    # if we force resize in processor.
    # For simplicity here: standard stack assumes processor resized to common size (e.g. 800x800)
    # If sizes vary, we need padding. Let's use a simple pad function.
    
    max_h = max(t.shape[1] for t in pixel_values)
    max_w = max(t.shape[2] for t in pixel_values)
    
    batch_tensors = []
    for t in pixel_values:
        pad_h = max_h - t.shape[1]
        pad_w = max_w - t.shape[2]
        # Pad with 0
        padded = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h))
        batch_tensors.append(padded)
        
    return {
        "pixel_values": torch.stack(batch_tensors),
        "labels": labels
    }

def main():
    args = get_args()
    
    # Processor (Handles resizing images to typically min_size=800, max_size=1333)
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    
    # Dataset
    train_dataset = RGBDBoxDataset(args.data_root, "train", processor)
    val_dataset = RGBDBoxDataset(args.data_root, "test", processor)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # Model (4 classes for buildings, DETR handles 'no_object' internally)
    model = get_rgbd_detr(num_labels=4)
    model.to(DEVICE)
    
    # Optimization
    # DETR needs different LRs for backbone and transformer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": args.lr_backbone},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    
    # Metric: mAP
    map_metric = MeanAveragePrecision().to(DEVICE)

    print("--- Starting RGB-D DETR Training ---")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            pixel_values = batch["pixel_values"].to(DEVICE)
            # Labels is a list of dicts, move inner tensors to device
            labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]
            
            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=labels)
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        map_metric.reset()
        
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(DEVICE)
                labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]
                
                outputs = model(pixel_values=pixel_values)
                
                # Convert logits to boxes for metric
                # DETR outputs [cx, cy, w, h] normalized
                target_sizes = torch.tensor([img.shape[1:] for img in pixel_values]).to(DEVICE)
                results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)
                
                # Update metric
                # results is list of dicts {'boxes':, 'scores':, 'labels':}
                # labels is list of dicts {'boxes':, 'class_labels':} -> Metric needs 'labels' key
                formatted_labels = []
                for l in labels:
                    formatted_labels.append({'boxes': l['boxes'], 'labels': l['class_labels']})
                    
                map_metric.update(results, formatted_labels)

        # Report
        metrics = map_metric.compute()
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val mAP_50: {metrics['map_50'].item():.4f}")
        
        # Save
        if (epoch+1) % 10 == 0:
            save_path = os.path.join(args.output_dir, f"detr_rgbd_ep{epoch+1}")
            model.save_pretrained(save_path)
            print(f"Saved to {save_path}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
train_mask2former_rgbd_distributed.py

Purpose:
    Multi-GPU Training for Mask2Former with 4-Channel Input (RGB + Depth).
    
    Updates:
    - Added mIoU metric to validation.
    - Suppressed warnings.
    - Saves best model based on mIoU.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Mask2FormerForUniversalSegmentation, 
    Mask2FormerConfig, 
    Mask2FormerImageProcessor
)
from transformers import logging as hf_logging
from PIL import Image
import rasterio
from tqdm import tqdm
from accelerate import Accelerator
import warnings

# --- 1. SUPPRESS WARNINGS ---
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error() # Silence HF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# --- CONFIGURATION ---
ID2LABEL = {
    0: "background", 
    1: "no_damage", 
    2: "minor_damage", 
    3: "major_damage", 
    4: "destroyed"
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
NUM_LABELS = len(ID2LABEL)

class CrossViewAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma      = nn.Parameter(torch.zeros(1))
        self.softmax    = nn.Softmax(dim=-1)
        
        # Define Scale Factor
        self.scale = (in_dim // 8) ** -0.5  # 1 / sqrt(dim)

    def forward(self, rgb_feat, depth_feat):
        B, C, H, W = rgb_feat.size()
        
        proj_query = self.query_conv(rgb_feat).view(B, -1, W * H).permute(0, 2, 1) 
        proj_key   = self.key_conv(depth_feat).view(B, -1, W * H)                 
        proj_value = self.value_conv(depth_feat).view(B, -1, W * H)               

        # --- FIX: Multiply by Scale ---
        energy = torch.bmm(proj_query, proj_key) * self.scale
        
        attention = self.softmax(energy)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        out = self.gamma * out + rgb_feat
        return out

# --- DATASET ---
class RGBD_Mask2FormerDataset(Dataset):
    def __init__(self, root_dir, processor, split="train", size=(512, 512)):
        self.root = os.path.join(root_dir, split)
        self.img_dir = os.path.join(self.root, "images")
        self.lbl_dir = os.path.join(self.root, "labels")
        self.dep_dir = os.path.join(self.root, "depth")
        self.processor = processor
        self.size = size
        self.files = sorted([f for f in os.listdir(self.img_dir) if f.endswith(".png")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        
        # RGB
        image = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        image = image.resize(self.size, Image.BILINEAR)
        rgb = np.array(image).astype(np.float32) / 255.0 
        
        # Depth
        dep_path = os.path.join(self.dep_dir, fname.replace(".png", ".tif"))
        if os.path.exists(dep_path):
            with rasterio.open(dep_path) as src:
                depth = src.read(1)
            d_img = Image.fromarray(depth).resize(self.size, Image.BILINEAR) 
            depth = np.array(d_img).astype(np.float32)
        else:
            depth = np.zeros(self.size[::-1], dtype=np.float32)

        depth = np.clip(depth, 0, 10.0) / 10.0
        depth = np.expand_dims(depth, axis=2) 
        
        # Stack RGBD
        rgbd = np.concatenate([rgb, depth], axis=2) 
        pixel_values = torch.tensor(rgbd).permute(2, 0, 1).float()
        
        # Labels
        label_img = Image.open(os.path.join(self.lbl_dir, fname)).resize(self.size, Image.NEAREST)
        label_arr = np.array(label_img)
        
        # Process for Model (Masks)
        inputs = self.processor(
            images=image, 
            segmentation_maps=label_arr, 
            task_inputs=["semantic"],
            return_tensors="pt"
        )
        
        return {
            "pixel_values": pixel_values,
            "mask_labels": inputs["mask_labels"][0],
            "class_labels": inputs["class_labels"][0],
            "labels": torch.tensor(label_arr).long() # Raw label for IoU calculation
        }

def rgbd_collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    mask_labels = [item["mask_labels"] for item in batch]
    class_labels = [item["class_labels"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch])
    return {
        "pixel_values": pixel_values, 
        "mask_labels": mask_labels, 
        "class_labels": class_labels,
        "labels": labels
    }

# --- MODEL SURGERY ---
def get_rgbd_mask2former(model_name):
    config = Mask2FormerConfig.from_pretrained(model_name)
    config.num_labels = NUM_LABELS
    config.id2label = ID2LABEL
    config.label2id = LABEL2ID
    
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_name, config=config, ignore_mismatched_sizes=True
    )
    
    # Modify Backbone Input Layer
    backbone = model.model.pixel_level_module.encoder
    old_proj = backbone.embeddings.patch_embeddings.projection
    
    new_proj = nn.Conv2d(
        in_channels=4, 
        out_channels=old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding
    )
    
    with torch.no_grad():
        new_proj.weight[:, :3, :, :] = old_proj.weight
        new_proj.weight[:, 3:, :, :] = torch.zeros_like(old_proj.weight[:, 0:1, :, :])
        new_proj.bias = old_proj.bias
        
    backbone.embeddings.patch_embeddings.projection = new_proj
    model.config.backbone_config.num_channels = 4
    
    return model

# --- METRIC UTILS ---
def update_confusion_matrix(preds, targets, num_classes):
    """Computes the confusion matrix for a batch."""
    # Filter invalid/ignore pixels (e.g. 255)
    mask = (targets >= 0) & (targets < num_classes)
    
    # bincount trick for fast confusion matrix
    return torch.bincount(
        num_classes * targets[mask] + preds[mask], 
        minlength=num_classes**2
    ).reshape(num_classes, num_classes)

# --- TRAINING LOOP ---
def train(args):
    accelerator = Accelerator()
    
    if accelerator.is_main_process:
        print(f"Training on {accelerator.num_processes} GPUs")
        print("Warnings are suppressed. Logs will be clean.")

    # Processor
    processor = Mask2FormerImageProcessor.from_pretrained(
        args.model_name, do_resize=False, do_normalize=False, ignore_index=255
    )
    
    # Datasets
    train_ds = RGBD_Mask2FormerDataset(args.data_root, processor, split="train", size=(args.img_size, args.img_size))
    val_split = "test" if os.path.exists(os.path.join(args.data_root, "test")) else "train"
    val_ds = RGBD_Mask2FormerDataset(args.data_root, processor, split=val_split, size=(args.img_size, args.img_size))
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, 
        collate_fn=rgbd_collate_fn, num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, 
        collate_fn=rgbd_collate_fn, num_workers=args.workers, pin_memory=True
    )
    
    model = get_rgbd_mask2former(args.model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Prepare
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    best_iou = 0.0
    
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        
        progress = tqdm(train_loader, desc="Train", disable=not accelerator.is_main_process)
        
        for batch in progress:
            optimizer.zero_grad()
            
            outputs = model(
                pixel_values=batch["pixel_values"], 
                mask_labels=batch["mask_labels"], 
                class_labels=batch["class_labels"]
            )
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            
            train_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
            
        # --- VALIDATE & IoU ---
        model.eval()
        val_loss = 0.0
        
        # Confusion Matrix for IoU (on device)
        # Size: 5x5
        confusion_matrix = torch.zeros((NUM_LABELS, NUM_LABELS), device=accelerator.device)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", disable=not accelerator.is_main_process):
                # 1. Forward Pass
                outputs = model(
                    pixel_values=batch["pixel_values"], 
                    mask_labels=batch["mask_labels"], 
                    class_labels=batch["class_labels"]
                )
                val_loss += outputs.loss.item()
                
                # 2. Get Semantic Map Predictions
                # Target size needed to resize logits back to 512x512
                target_sizes = [(args.img_size, args.img_size)] * len(batch["pixel_values"])
                
                pred_maps = processor.post_process_semantic_segmentation(
                    outputs, target_sizes=target_sizes
                )
                
                # Stack predictions to tensor (B, H, W)
                pred_tensor = torch.stack(pred_maps) 
                target_tensor = batch["labels"]
                
                # 3. Update Confusion Matrix
                confusion_matrix += update_confusion_matrix(pred_tensor, target_tensor, NUM_LABELS)

        # --- METRIC AGGREGATION ---
        
        # Sync Loss
        avg_val_loss = val_loss / len(val_loader)
        avg_val_loss = accelerator.gather(torch.tensor([avg_val_loss]).to(accelerator.device)).mean().item()
        
        # Sync Confusion Matrix (Sum across GPUs)
        confusion_matrix = accelerator.reduce(confusion_matrix, reduction="sum")
        
        if accelerator.is_main_process:
            # Calculate IoU
            # IoU = diag / (row_sum + col_sum - diag)
            intersection = torch.diag(confusion_matrix)
            union = confusion_matrix.sum(dim=1) + confusion_matrix.sum(dim=0) - intersection
            
            # Avoid division by zero
            iou_per_class = intersection / (union + 1e-6)
            mIoU = iou_per_class.mean().item()
            
            print(f"Val Loss: {avg_val_loss:.4f} | mIoU: {mIoU:.4f}")
            print(f"Class IoU: {[f'{x:.3f}' for x in iou_per_class.cpu().tolist()]}")
            
            # Save Best Model (based on mIoU)
            if mIoU > best_iou:
                best_iou = mIoU
                unwrapped_model = accelerator.unwrap_model(model)
                save_path = os.path.join(args.output_dir, "best_model.pth")
                torch.save(unwrapped_model.state_dict(), save_path)
                print(f"Saved New Best Model! (mIoU: {mIoU:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/mask2former_distributed")
    parser.add_argument("--model-name", type=str, default="facebook/mask2former-swin-tiny-cityscapes-semantic")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
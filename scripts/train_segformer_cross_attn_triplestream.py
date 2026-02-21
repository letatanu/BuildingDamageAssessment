#!/usr/bin/env python

"""
train_segformer_tristream_gated.py

Purpose:
Train a TRI-STREAM SegFormer with GATED CROSS-ATTENTION Fusion.

Architecture:
1. RGB Encoder (Stream A)   -> Extracts Texture
2. Depth Encoder (Stream B) -> Extracts Flood Intensity (1 Channel)
3. Mask Encoder (Stream C)  -> Extracts Building Geometry (1 Channel)
4. Gated Fusion             -> RGB + (alpha * Depth_Attn) + (beta * Mask_Attn)
5. Decoder                  -> Predicts Damage

Usage:
torchrun --nproc_per_node=4 scripts/train_segformer_tristream_gated.py \
--data_root data/crasar_water_osm_mask \
--batch_size 4
"""

import os
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp
from transformers import SegformerForSemanticSegmentation, SegformerConfig, SegformerModel
from tqdm import tqdm
from PIL import Image
import rasterio
import torchmetrics
from scipy.ndimage import label as nd_label
import torch.nn.functional as F
from rasterio.errors import NotGeoreferencedWarning
import warnings

# Suppress rasterio warnings
warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)

# --- 1. DATASET (RGB, Depth, Mask Separate) ---

class CRASARTriStreamDataset(Dataset):
    def __init__(self, root_dir, split="train", target_size=512):
        self.root = os.path.join(root_dir, split)
        self.img_dir = os.path.join(self.root, "images")
        self.lbl_dir = os.path.join(self.root, "labels")
        self.dep_dir = os.path.join(self.root, "depth")
        self.mask_dir = os.path.join(self.root, "ms_building_masks")
        
        self.files = sorted([f for f in os.listdir(self.img_dir) if f.endswith(".png")])
        self.target_size = target_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        
        # Load RGB
        img = np.array(Image.open(os.path.join(self.img_dir, fname)).convert("RGB"))
        
        # Load Label
        lbl = np.array(Image.open(os.path.join(self.lbl_dir, fname)))
        
        # Load Depth
        dep_path = os.path.join(self.dep_dir, fname.replace(".png", ".tif"))
        if os.path.exists(dep_path):
            with rasterio.open(dep_path) as src:
                depth = src.read(1)
                depth[depth == -100] = 0
                depth = np.nan_to_num(depth)
        else:
            depth = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
            
        # Load Mask
        mask_path = os.path.join(self.mask_dir, fname)
        if os.path.exists(mask_path):
            b_mask = np.array(Image.open(mask_path))
        else:
            b_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        # Resize if needed
        target_size = self.target_size
        if img.shape[0] != target_size:
            img = np.array(Image.fromarray(img).resize((target_size, target_size)))
            lbl = np.array(Image.fromarray(lbl).resize((target_size, target_size), Image.NEAREST))
            depth = np.array(Image.fromarray(depth).resize((target_size, target_size)))
            b_mask = np.array(Image.fromarray(b_mask).resize((target_size, target_size), Image.NEAREST))

        # Normalize
        img = img.astype(np.float32) / 255.0
        b_mask = (b_mask > 0).astype(np.float32) # Binary 0 or 1
        depth = np.clip(depth, 0, 100) / 100.0     # 0 to 1

        # Prepare Tensors
        # Stream 1: RGB (3, H, W)
        rgb_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        
        # Stream 2: Depth (1, H, W)
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).float()
        
        # Stream 3: Mask (1, H, W)
        mask_tensor = torch.from_numpy(b_mask).unsqueeze(0).float()

        lbl_tensor = torch.from_numpy(lbl).long()

        return {
            "rgb": rgb_tensor, 
            "depth": depth_tensor, 
            "mask": mask_tensor, 
            "labels": lbl_tensor
        }

# --- 2. GATED CROSS ATTENTION MODULE ---

class GatedCrossAttention(nn.Module):
    """
    Fuses Main Feature (RGB) with Context Feature (Depth or Mask).
    Uses a learnable Gate initialized to 0 to prevent noise injection at start.
    
    Q = RGB, K = Context, V = Context
    Output = Gate * Attention(Q, K, V)
    """
    def __init__(self, in_channels, heads=4, reduction_factor=1):
        super().__init__()
        self.reduction_factor = reduction_factor
        
        # Optional: downsample before attention to save memory
        if reduction_factor > 1:
            self.pool = nn.AvgPool2d(kernel_size=reduction_factor, stride=reduction_factor)
            self.upsample = nn.Upsample(scale_factor=reduction_factor, mode='bilinear', align_corners=False)
        
        self.norm_main = nn.LayerNorm(in_channels)
        self.norm_ctx = nn.LayerNorm(in_channels)
        
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=heads, batch_first=True)
        
        # Learnable Gate initialized to 0
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x_main, x_ctx):
        b, c, h, w = x_main.shape
        
        # Downsample if needed
        if self.reduction_factor > 1:
            x_main_red = self.pool(x_main)
            x_ctx_red = self.pool(x_ctx)
        else:
            x_main_red = x_main
            x_ctx_red = x_ctx
            
        # Flatten: (B, C, H, W) -> (B, H*W, C)
        q = x_main_red.flatten(2).transpose(1, 2)
        k = x_ctx_red.flatten(2).transpose(1, 2)
        v = k # Use context as both Key and Value
        
        # Norm
        q = self.norm_main(q)
        k = self.norm_ctx(k)
        
        # Attention
        attn_out, _ = self.attn(query=q, key=k, value=k)
        
        # Reshape back to (B, C, H, W)
        h_r, w_r = x_main_red.shape[-2:]
        attn_out = attn_out.transpose(1, 2).reshape(b, c, h_r, w_r)
        
        # Upsample if we downsampled
        if self.reduction_factor > 1:
            attn_out = self.upsample(attn_out)
            
        # Return Gated Attention (Main stream added later)
        return self.gate * attn_out

# --- 3. TRI-STREAM MODEL ---

class TriStreamSegFormer(nn.Module):
    def __init__(self, model_name="nvidia/mit-b0", num_classes=5):
        super().__init__()
        
        # -- Stream 1: RGB Encoder --
        self.rgb_model = SegformerModel.from_pretrained(model_name)
        
        # -- Stream 2: Depth Encoder (1 Channel) --
        self.depth_model = SegformerModel.from_pretrained(model_name)
        self._modify_first_layer(self.depth_model)
        
        # -- Stream 3: Mask Encoder (1 Channel) --
        self.mask_model = SegformerModel.from_pretrained(model_name)
        self._modify_first_layer(self.mask_model)

        # -- Fusion Layers --
        # SegFormer has 4 feature scales. We need fusion at each scale.
        dims = self.rgb_model.config.hidden_sizes # e.g. [32, 64, 160, 256]
        
        # Independent attention blocks for Depth and Mask
        self.fusion_depth = nn.ModuleList([
            GatedCrossAttention(dims[0], reduction_factor=2),
            GatedCrossAttention(dims[1], reduction_factor=2),
            GatedCrossAttention(dims[2], reduction_factor=4),
            GatedCrossAttention(dims[3], reduction_factor=4)
        ])
        
        self.fusion_mask = nn.ModuleList([
            GatedCrossAttention(dims[0], reduction_factor=2),
            GatedCrossAttention(dims[1], reduction_factor=2),
            GatedCrossAttention(dims[2], reduction_factor=4),
            GatedCrossAttention(dims[3], reduction_factor=4)
        ])
        
        # Feed Forward after fusion to mix features
        self.mix_ffns = nn.ModuleList([
             nn.Sequential(
                nn.Conv2d(dim, dim, 1),
                nn.GELU(),
                nn.Conv2d(dim, dim, 1)
             ) for dim in dims
        ])

        # -- Decoder Head --
        temp_full_model = SegformerForSemanticSegmentation.from_pretrained(model_name, num_labels=num_classes)
        self.decode_head = temp_full_model.decode_head

    def _modify_first_layer(self, model):
        """Changes 3-channel input layer to 1-channel, avg init."""
        old_layer = model.encoder.patch_embeddings[0].proj
        new_layer = nn.Conv2d(1, old_layer.out_channels, 
                              kernel_size=old_layer.kernel_size, 
                              stride=old_layer.stride, 
                              padding=old_layer.padding)
        
        # Initialize with average of RGB weights
        with torch.no_grad():
            avg_weight = torch.mean(old_layer.weight, dim=1, keepdim=True)
            new_layer.weight[:] = avg_weight
            new_layer.bias[:] = old_layer.bias
            
        model.encoder.patch_embeddings[0].proj = new_layer

    def forward(self, rgb, depth, mask, labels=None):
        # 1. Extract Features
        out_rgb = self.rgb_model(rgb, output_hidden_states=True).hidden_states
        out_depth = self.depth_model(depth, output_hidden_states=True).hidden_states
        out_mask = self.mask_model(mask, output_hidden_states=True).hidden_states
        
        # 2. Fuse Features at each scale
        fused_features = []
        for i in range(len(out_rgb)):
            f_rgb = out_rgb[i]
            f_depth = out_depth[i]
            f_mask = out_mask[i]
            
            # Gated Fusion: RGB + (alpha * Depth) + (beta * Mask)
            attn_d = self.fusion_depth[i](f_rgb, f_depth)
            attn_m = self.fusion_mask[i](f_rgb, f_mask)
            
            fused = f_rgb + attn_d + attn_m
            
            # Optional Mix
            fused = self.mix_ffns[i](fused)
            
            fused_features.append(fused)

        # 3. Decode
        logits = self.decode_head(fused_features)
        
        # 4. Upsample
        logits = F.interpolate(logits, size=rgb.shape[-2:], mode='bilinear', align_corners=False)

        loss = None
        if labels is not None:
            # Important: Background is 0, so ignore_index must NOT be 255 unless labels are 255
            # We assume standard 0-4 labels where 0 is background.
            loss_fct = nn.CrossEntropyLoss(ignore_index=255) 
            loss = loss_fct(logits, labels)
            
        return loss, logits

# --- 4. BUILDING LEVEL METRICS ---

def compute_building_confusion_matrix(preds_cpu, labels_cpu, num_classes=5):
    """
    Counts whole buildings.
    """
    batch_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    structure = np.ones((3,3), dtype=np.int8)

    for i in range(len(labels_cpu)):
        pred = preds_cpu[i]
        lbl = labels_cpu[i]
        
        # Building mask (everything except background 0)
        building_mask = (lbl > 0) 
        labeled_array, num_features = nd_label(building_mask, structure=structure)
        
        if num_features == 0: continue
        
        for feature_id in range(1, num_features + 1):
            mask = (labeled_array == feature_id)
            
            # Ground Truth
            true_cls = np.bincount(lbl[mask]).argmax()
            
            # Prediction
            pred_pixels = pred[mask]
            if len(pred_pixels) == 0: continue
            
            # Majority Vote
            pred_cls = np.bincount(pred_pixels, minlength=num_classes).argmax()
            
            batch_cm[true_cls, pred_cls] += 1
            
    return torch.tensor(batch_cm)

# --- 5. MAIN ---

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="runs/tristream_gated")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=6e-5)
    args = parser.parse_args()

    ddp_setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    device = torch.device(f"cuda:{local_rank}")

    if global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"--- Tri-Stream Gated Training | {args.output_dir} ---")

    # Dataset
    train_ds = CRASARTriStreamDataset(args.data_root, split="train")
    
    # Check if test split exists, else use train for validation (debugging)
    val_split = "test" if os.path.exists(os.path.join(args.data_root, "test")) else "train"
    val_ds = CRASARTriStreamDataset(args.data_root, split=val_split)

    train_sampler = DistributedSampler(train_ds)
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)

    # Model
    model = TriStreamSegFormer(num_classes=5).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Metrics
    # Note: ignore_index=0 for Pixel IoU (Buildings only)
    iou_per_class = torchmetrics.JaccardIndex(task="multiclass", num_classes=5, average='none', ignore_index=0).to(device)
    class_names = ["Backgrnd", "No Damag", "Minor", "Major", "Destroye"]

    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        
        if global_rank == 0:
            print(f"--- Epoch {epoch+1}/{args.epochs} ---")
            
        for batch in tqdm(train_loader, disable=(global_rank != 0)):
            rgb = batch["rgb"].to(device)
            depth = batch["depth"].to(device)
            mask = batch["mask"].to(device)
            lbl = batch["labels"].to(device)
            
            optimizer.zero_grad()
            loss, logits = model(rgb, depth, mask, lbl)
            loss.backward()
            optimizer.step()

        # --- Validation ---
        model.eval()
        iou_per_class.reset()
        local_bld_cm = torch.zeros((5, 5), dtype=torch.int64).to(device)

        with torch.no_grad():
            for batch in tqdm(val_loader, disable=(global_rank != 0), desc="Validating"):
                rgb = batch["rgb"].to(device)
                depth = batch["depth"].to(device)
                mask = batch["mask"].to(device)
                lbl = batch["labels"].to(device)
                
                _, logits = model(rgb, depth, mask)
                preds = torch.argmax(logits, dim=1)
                
                # Pixel IoU
                iou_per_class.update(preds, lbl)
                
                # Building CM
                curr_cm = compute_building_confusion_matrix(preds.cpu().numpy(), lbl.cpu().numpy())
                local_bld_cm += curr_cm.to(device)

        # Sync
        final_ious = iou_per_class.compute() # Vector of 5
        all_reduce(local_bld_cm, op=ReduceOp.SUM)

        if global_rank == 0:
            cm_np = local_bld_cm.cpu().numpy()
            row_sums = cm_np.sum(axis=1)
            row_sums[row_sums == 0] = 1
            b_recall = np.diag(cm_np) / row_sums

            print("="*60)
            print(f"VALIDATION REPORT (Tri-Stream Gated Epoch {epoch+1})")
            print("-" * 60)
            print(f"{'Class':<12} | {'Pixel IoU':<10} | {'Bld Recall':<10}")
            print("-" * 60)
            
            # Print from index 1 (skip background)
            for i in range(1, 5):
                print(f"{class_names[i]:<12} | {final_ious[i].item():.4f}     | {b_recall[i]:.4f}")
            
            print("-" * 60)
            print("BUILDING CONFUSION MATRIX")
            print("-" * 60)
            
            # Header
            header = "".join([f"{name:>10}" for name in class_names])
            print(f"{'True v':>10} {header}")
            
            for i, row in enumerate(class_names):
                row_str = "".join([f"{val:10d}" for val in cm_np[i]])
                print(f"{row:>10} {row_str}")
            print("="*60)

            # Save
            if (epoch + 1) % 5 == 0:
                torch.save(model.module.state_dict(), os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pth"))

    cleanup()

if __name__ == "__main__":
    main()

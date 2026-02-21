#!/usr/bin/env python
"""
train_segformer_cross_attn.py

Purpose:
    Train a DUAL-STREAM SegFormer with CROSS-ATTENTION Fusion.
    
    Architecture:
    1. RGB Encoder (Stream A) -> Extracts Texture
    2. Geo Encoder (Stream B) -> Extracts Structure (Depth + Mask)
    3. Cross-Attention Heads  -> Fuses streams at 4 scales
    4. Decoder                -> Predicts Damage

Usage:
    torchrun --nproc_per_node=4 scripts/train_segformer_cross_attn.py \
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
from tqdm import tqdm

# Suppress rasterio warnings
warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)
# --- 1. DATASET (RGB + Depth + Mask) ---

class CRASARFusedDataset(Dataset):
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
                depth[depth < -100] = 0
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
        b_mask = (b_mask > 0).astype(np.float32)
        depth = np.clip(depth, 0, 100) / 100.0
        
        # Prepare Tensors
        # Stream 1: RGB (3, H, W)
        rgb_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        
        # Stream 2: Geometry (Depth + Mask) -> (2, H, W)
        geo_stack = np.stack([depth, b_mask], axis=0) # Channel First
        geo_tensor = torch.from_numpy(geo_stack).float()
        
        lbl_tensor = torch.from_numpy(lbl).long()
        
        return {"rgb": rgb_tensor, "geo": geo_tensor, "labels": lbl_tensor}

# --- 2. CROSS ATTENTION MODULE ---

class CrossAttentionFusion(nn.Module):
    """
    Fuses two feature maps (RGB and Geo) using Cross Attention.
    Q = RGB, K = Geo, V = Geo
    """
    def __init__(self, in_channels, heads=4, reduction_factor=4):
        super().__init__()
        self.reduction_factor = reduction_factor
        
        # Optional: downsample before attention
        if reduction_factor > 1:
            self.pool = nn.AvgPool2d(kernel_size=reduction_factor)
            self.upsample = nn.Upsample(scale_factor=reduction_factor, mode='bilinear', align_corners=False)
        
        self.norm_rgb = nn.LayerNorm(in_channels)
        self.norm_geo = nn.LayerNorm(in_channels)
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.GELU(),
            nn.Linear(in_channels * 4, in_channels)
        )
        self.norm_ffn = nn.LayerNorm(in_channels)

    def forward(self, x_rgb, x_geo):
        b, c, h, w = x_rgb.shape
        
        # Downsample if needed
        if self.reduction_factor > 1:
            x_rgb_reduced = self.pool(x_rgb)
            x_geo_reduced = self.pool(x_geo)
            _, _, h_r, w_r = x_rgb_reduced.shape
        else:
            x_rgb_reduced = x_rgb
            x_geo_reduced = x_geo
            h_r, w_r = h, w
        
        # Flatten: (B, C, H, W) -> (B, H*W, C)
        rgb_flat = x_rgb_reduced.flatten(2).transpose(1, 2)
        geo_flat = x_geo_reduced.flatten(2).transpose(1, 2)
        
        # Norm
        rgb_norm = self.norm_rgb(rgb_flat)
        geo_norm = self.norm_geo(geo_flat)
        
        # Cross Attention
        attn_out, _ = self.attn(query=rgb_norm, key=geo_norm, value=geo_norm)
        
        # Residual Connection
        fused = rgb_flat + attn_out
        
        # Feed Forward + Residual
        fused = fused + self.ffn(self.norm_ffn(fused))
        
        # Reshape back
        fused = fused.transpose(1, 2).reshape(b, c, h_r, w_r)
        
        # Upsample if we downsampled
        if self.reduction_factor > 1:
            fused = self.upsample(fused)
        
        return fused


# --- 3. DUAL STREAM MODEL ---

class DualStreamSegFormer(nn.Module):
    def __init__(self, model_name="nvidia/mit-b0", num_classes=5):
        super().__init__()
        
        # -- Stream 1: RGB Encoder --
        self.rgb_model = SegformerModel.from_pretrained(model_name)
        
        # -- Stream 2: Geometry Encoder (Depth+Mask) --
        # We initialize a second encoder and modify input to 2 channels
        self.geo_model = SegformerModel.from_pretrained(model_name)
        
        # Modify first layer for 2 channels
        old_layer = self.geo_model.encoder.patch_embeddings[0].proj
        new_layer = nn.Conv2d(2, old_layer.out_channels, kernel_size=7, stride=4, padding=3)
        # Initialize with average of RGB weights
        with torch.no_grad():
            new_layer.weight[:, :2] = old_layer.weight[:, :2] # Copy R,G weights roughly
        self.geo_model.encoder.patch_embeddings[0].proj = new_layer
        
        # -- Fusion Layers --
        # SegFormer has 4 feature scales. We need 4 fusion blocks.
        # hidden_sizes for b0: [32, 64, 160, 256]
        # hidden_sizes for b2: [64, 128, 320, 512]
        dims = self.rgb_model.config.hidden_sizes
        self.fusion_blocks = nn.ModuleList([
            CrossAttentionFusion(dims[0], reduction_factor=2),
            CrossAttentionFusion(dims[1], reduction_factor=2),
            CrossAttentionFusion(dims[2], reduction_factor=4),
            CrossAttentionFusion(dims[3], reduction_factor=4)
        ])

        
        # -- Decoder Head --
        # We reuse the logic from SegFormerForSemanticSegmentation but applied to fused features
        # Easier trick: Use a dummy SegFormerForSegmentation to hold the decoder
        temp_full_model = SegformerForSemanticSegmentation.from_pretrained(model_name, num_labels=num_classes)
        self.decode_head = temp_full_model.decode_head
        
        # Resize decoder head to match num_classes if pretrained didn't match
        # (Handled by from_pretrained params usually, but good to be safe)
        
    def forward(self, rgb, geo, labels=None):
        # 1. Extract Features
        # output_hidden_states=True guarantees we get the list of 4 scales
        out_rgb = self.rgb_model(rgb, output_hidden_states=True)
        out_geo = self.geo_model(geo, output_hidden_states=True)
        
        feats_rgb = out_rgb.hidden_states # Tuple of 4 tensors
        feats_geo = out_geo.hidden_states
        
        # 2. Fuse Features at each scale
        fused_features = []
        for i, block in enumerate(self.fusion_blocks):
            f_rgb = feats_rgb[i]
            f_geo = feats_geo[i]
            fused = block(f_rgb, f_geo)
            fused_features.append(fused)
            
        # 3. Decode
        logits = self.decode_head(fused_features)
        
        # 4. Resize to Input Size (4x upsample)
        logits = F.interpolate(logits, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=255)
            loss = loss_fct(logits, labels)
            
        return {"loss": loss, "logits": logits}

# --- 4. BUILDING LEVEL CONFUSION MATRIX ---
def compute_building_confusion_matrix(preds_cpu, labels_cpu, num_classes=5):
    # (Same robust object-counting logic as previous script)
    batch_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    structure = np.ones((3,3), dtype=np.int8)
    for i in range(len(labels_cpu)):
        pred, lbl = preds_cpu[i], labels_cpu[i]
        building_mask = (lbl > 0)
        labeled_array, num_features = nd_label(building_mask, structure=structure)
        if num_features == 0: continue
        for feature_id in range(1, num_features + 1):
            mask = (labeled_array == feature_id)
            true_cls = np.bincount(lbl[mask]).argmax()
            pred_pixels = pred[mask]
            if len(pred_pixels) == 0: continue
            pred_cls = np.bincount(pred_pixels, minlength=num_classes).argmax()
            batch_cm[true_cls, pred_cls] += 1
    return torch.tensor(batch_cm)

# --- 5. MAIN ---

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="runs")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ddp_setup()
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{rank}")
    
    # Dataset
    train_ds = CRASARFusedDataset(args.data_root, split="train")
    val_split = "test" if os.path.exists(os.path.join(args.data_root, "test")) else "train"
    val_ds = CRASARFusedDataset(args.data_root, split=val_split)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=DistributedSampler(train_ds), num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=DistributedSampler(val_ds, shuffle=False), num_workers=4)
    
    # Model (Dual Stream)
    model = DualStreamSegFormer(num_classes=5).to(device)
    model = DDP(model, device_ids=[rank]) # Unused params possible in decode head
    optim = torch.optim.AdamW(model.parameters(), lr=0.00006)
    
    # Metrics
    # IoU per class (average='none' returns a list)
    iou_per_class = torchmetrics.JaccardIndex(task="multiclass", num_classes=5, average='none', ignore_index=255).to(device)
    
    class_names = ["Backgrnd", "No Damag", "Minor", "Major", "Destroye"]
    
    for epoch in range(args.epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch)
        if rank == 0: print(f"--- Epoch {epoch+1} ---")

        # Train Loop with tqdm
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", disable=(rank != 0))
        epoch_loss = 0.0
        
        for batch in pbar:
            rgb = batch["rgb"].to(device)
            geo = batch["geo"].to(device)
            lbl = batch["labels"].to(device)
            
            optim.zero_grad()
            out = model(rgb, geo, lbl)
            loss = out["loss"]
            loss.backward()
            optim.step()
            
            epoch_loss += loss.item()
            
            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        if rank == 0:
            print(f"Epoch {epoch+1} Avg Loss: {epoch_loss/len(train_loader):.4f}\n")

        # Validation Loop
        model.eval()
        iou_per_class.reset()
        local_bld_cm = torch.zeros((5,5), dtype=torch.int64).to(device)
        
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch["rgb"].to(device)
                geo = batch["geo"].to(device)
                lbl = batch["labels"].to(device)
                
                logits = model(rgb, geo)["logits"]
                preds = torch.argmax(logits, dim=1)
                
                # Update Metrics
                iou_per_class.update(preds, lbl)
                
                # Update Building Matrix
                cm = compute_building_confusion_matrix(preds.cpu().numpy(), lbl.cpu().numpy())
                local_bld_cm += cm.to(device)

        # Sync and Print
        final_ious = iou_per_class.compute() # Vector [5]
        all_reduce(local_bld_cm, op=ReduceOp.SUM)
        
        if rank == 0:
            cm_np = local_bld_cm.cpu().numpy()
            
            # Calculate Building Recall
            row_sums = cm_np.sum(axis=1)
            row_sums[row_sums == 0] = 1
            b_recall = np.diag(cm_np) / row_sums

            print("\n" + "="*60)
            print(f"VALIDATION REPORT (Dual Stream + Cross Attn)")
            print("="*60)
            
            # 1. Per-Class Pixel IoU
            print(f"{'Class':<15} | {'Pixel IoU':<10} | {'Bld Recall':<10}")
            print("-" * 60)
            for i in range(1, 5): # Skip background
                p_iou = final_ious[i].item()
                b_rec = b_recall[i]
                print(f"{class_names[i]:<15} | {p_iou:.4f}     | {b_rec:.4f}")
            
            print("-" * 60)
            print("BUILDING CONFUSION MATRIX")
            print("-" * 60)
            header = "".join([f"{name:>10}" for name in class_names])
            print(f"{'True v':<10} {header}")
            for i, row in enumerate(class_names):
                row_str = "".join([f"{val:>10d}" for val in cm_np[i]])
                print(f"{row[:8]:<10} {row_str}")
            print("="*60 + "\n")
            
            if (epoch+1) % 5 == 0:
                # torch.save(model.module.state_dict(), f"runs/dual_stream_checkpoint_{epoch+1}.pth")
                torch.save(model.module.state_dict(), f"{args.output_dir}/dual_stream_checkpoint_{epoch+1}.pth")

    destroy_process_group()

if __name__ == "__main__":
    main()
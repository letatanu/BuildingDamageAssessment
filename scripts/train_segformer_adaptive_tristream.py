#!/usr/bin/env python
"""
train_segformer_adaptive_tristream.py

Purpose:
Train a TRI-STREAM SegFormer with ADAPTIVE GATED CROSS-ATTENTION Fusion.

Architecture:
1. RGB Encoder (Stream A) -> Extracts Texture
2. Depth Encoder (Stream B) -> Extracts Flood Intensity (1 Channel)
3. Mask Encoder (Stream C) -> Extracts Building Geometry (1 Channel)
4. Adaptive Gated Fusion -> RGB + (alpha * Depth_Attn) + (beta * Mask_Attn)
   where alpha < beta (prioritizing mask over depth)
5. Decoder -> Predicts Damage

Usage:
torchrun --nproc_per_node=4 train_segformer_adaptive_tristream.py \
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
from transformers import SegformerForSemanticSegmentation, SegformerModel
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
        b_mask = (b_mask > 0).astype(np.float32)  # Binary 0 or 1
        depth = np.clip(depth, 0, 100) / 100.0  # 0 to 1
        
        # Prepare Tensors
        rgb_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(b_mask).unsqueeze(0).float()
        lbl_tensor = torch.from_numpy(lbl).long()
        
        return {
            "rgb": rgb_tensor,
            "depth": depth_tensor,
            "mask": mask_tensor,
            "labels": lbl_tensor
        }


# --- 2. ADAPTIVE GATED CROSS ATTENTION MODULE ---
class AdaptiveGatedCrossAttention(nn.Module):
    """
    Fuses Main Feature (RGB) with Context Feature (Depth or Mask).
    Uses context-aware gating that adapts based on feature quality.
    """
    def __init__(self, in_channels, heads=4, reduction_factor=1, init_gate_value=0.0):
        super().__init__()
        self.reduction_factor = reduction_factor
        
        # Optional: downsample before attention to save memory
        if reduction_factor > 1:
            self.pool = nn.AvgPool2d(kernel_size=reduction_factor, stride=reduction_factor)
            self.upsample = nn.Upsample(scale_factor=reduction_factor, mode='bilinear', align_corners=False)
        
        self.norm_main = nn.LayerNorm(in_channels)
        self.norm_ctx = nn.LayerNorm(in_channels)
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=heads, batch_first=True)
        
        # Context-aware gating network
        self.gate_network = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # Learnable base gate with custom initialization
        self.base_gate = nn.Parameter(torch.tensor([init_gate_value]))

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
        v = k
        
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
        
        # Compute adaptive gate based on both features
        concat_features = torch.cat([x_main, x_ctx], dim=1)
        adaptive_gate = self.gate_network(concat_features)
        
        # Combine base gate (learnable global importance) with adaptive gate (instance-specific)
        final_gate = torch.sigmoid(self.base_gate) * adaptive_gate
        
        return final_gate * attn_out


# --- 3. HIERARCHICAL GATED FUSION ---
class HierarchicalGatedFusion(nn.Module):
    """
    Fuses RGB with gated Depth and Mask attention outputs.
    Uses learnable importance weights to prioritize mask over depth.
    """
    def __init__(self, in_channels):
        super().__init__()
        
        # Confidence estimators for each modality
        self.depth_confidence = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 1, 1),
            nn.Softplus()
        )
        
        self.mask_confidence = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 1, 1),
            nn.Softplus()
        )
        
        # Learnable importance weights (initialized to favor mask over depth)
        # depth_importance = 0.3, mask_importance = 0.7
        self.importance_weights = nn.Parameter(torch.tensor([0.3, 0.7]))
    
    def forward(self, f_rgb, f_depth, f_mask, attn_d, attn_m):
        # Estimate confidence for each modality
        conf_d = self.depth_confidence(f_depth)
        conf_m = self.mask_confidence(f_mask)
        
        # Combine confidence with learned importance using softmax
        weighted_conf = torch.stack([
            conf_d * torch.abs(self.importance_weights[0]),
            conf_m * torch.abs(self.importance_weights[1])
        ], dim=0)
        
        # Normalize weights
        weights = torch.softmax(weighted_conf, dim=0)
        
        # Weighted fusion
        fused = f_rgb + weights[0] * attn_d + weights[1] * attn_m
        return fused


# --- 4. TRI-STREAM MODEL WITH ADAPTIVE GATING ---
class AdaptiveTriStreamSegFormer(nn.Module):
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
        dims = self.rgb_model.config.hidden_sizes  # e.g. [32, 64, 160, 256]
        
        # Depth attention (start with lower importance: init_gate_value=-1.0)
        self.fusion_depth = nn.ModuleList([
            AdaptiveGatedCrossAttention(dims[0], reduction_factor=2, init_gate_value=-1.0),
            AdaptiveGatedCrossAttention(dims[1], reduction_factor=2, init_gate_value=-1.0),
            AdaptiveGatedCrossAttention(dims[2], reduction_factor=4, init_gate_value=-1.0),
            AdaptiveGatedCrossAttention(dims[3], reduction_factor=4, init_gate_value=-1.0)
        ])
        
        # Mask attention (start with higher importance: init_gate_value=0.5)
        self.fusion_mask = nn.ModuleList([
            AdaptiveGatedCrossAttention(dims[0], reduction_factor=2, init_gate_value=0.5),
            AdaptiveGatedCrossAttention(dims[1], reduction_factor=2, init_gate_value=0.5),
            AdaptiveGatedCrossAttention(dims[2], reduction_factor=4, init_gate_value=0.5),
            AdaptiveGatedCrossAttention(dims[3], reduction_factor=4, init_gate_value=0.5)
        ])
        
        # Hierarchical fusion modules
        self.hierarchical_fusion = nn.ModuleList([
            HierarchicalGatedFusion(dim) for dim in dims
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
        temp_full_model = SegformerForSemanticSegmentation.from_pretrained(
            model_name, num_labels=num_classes
        )
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
            
            # Adaptive Gated Attention
            attn_d = self.fusion_depth[i](f_rgb, f_depth)
            attn_m = self.fusion_mask[i](f_rgb, f_mask)
            
            # Hierarchical Fusion with learned priorities
            fused = self.hierarchical_fusion[i](f_rgb, f_depth, f_mask, attn_d, attn_m)
            
            # Mix features
            fused = self.mix_ffns[i](fused)
            fused_features.append(fused)
        
        # 3. Decode
        logits = self.decode_head(fused_features)
        
        # 4. Upsample
        logits = F.interpolate(logits, size=rgb.shape[-2:], mode='bilinear', align_corners=False)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=255)
            loss = loss_fct(logits, labels)
        
        return loss, logits


# --- 5. BUILDING LEVEL METRICS ---
def compute_building_confusion_matrix(preds_cpu, labels_cpu, num_classes=5):
    """Counts whole buildings."""
    batch_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    structure = np.ones((3, 3), dtype=np.int8)
    
    for i in range(len(labels_cpu)):
        pred = preds_cpu[i]
        lbl = labels_cpu[i]
        
        # Building mask (everything except background 0)
        building_mask = (lbl > 0)
        labeled_array, num_features = nd_label(building_mask, structure=structure)
        
        if num_features == 0:
            continue
        
        for feature_id in range(1, num_features + 1):
            mask = (labeled_array == feature_id)
            
            # Ground Truth
            true_cls = np.bincount(lbl[mask]).argmax()
            
            # Prediction
            pred_pixels = pred[mask]
            if len(pred_pixels) == 0:
                continue
            
            # Majority Vote
            pred_cls = np.bincount(pred_pixels, minlength=num_classes).argmax()
            batch_cm[true_cls, pred_cls] += 1
    
    return torch.tensor(batch_cm)


# --- 6. MAIN ---
def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup():
    destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="runs/adaptive_tristream")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--model_name", type=str, default="nvidia/mit-b0")
    args = parser.parse_args()
    
    ddp_setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    device = torch.device(f"cuda:{local_rank}")
    
    if global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"--- Adaptive Tri-Stream Training | {args.output_dir} ---")
    
    # Dataset
    train_ds = CRASARTriStreamDataset(args.data_root, split="train")
    val_split = "test" if os.path.exists(os.path.join(args.data_root, "test")) else "train"
    val_ds = CRASARTriStreamDataset(args.data_root, split=val_split)
    
    train_sampler = DistributedSampler(train_ds)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, 
                            sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, 
                          sampler=val_sampler, num_workers=4, pin_memory=True)
    
    # Model
    model = AdaptiveTriStreamSegFormer(model_name=args.model_name, num_classes=5).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Metrics
    iou_per_class = torchmetrics.JaccardIndex(
        task="multiclass", num_classes=5, average='none', ignore_index=0
    ).to(device)
    
    class_names = ["Backgrnd", "No Damag", "Minor", "Major", "Destroye"]
    
    best_miou = 0.0
    
    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        
        if global_rank == 0:
            print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        epoch_loss = 0.0
        for batch in tqdm(train_loader, disable=(global_rank != 0)):
            rgb = batch["rgb"].to(device)
            depth = batch["depth"].to(device)
            mask = batch["mask"].to(device)
            lbl = batch["labels"].to(device)
            
            optimizer.zero_grad()
            loss, logits = model(rgb, depth, mask, lbl)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        
        if global_rank == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        
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
        
        # Sync metrics
        final_ious = iou_per_class.compute()
        all_reduce(local_bld_cm, op=ReduceOp.SUM)
        
        if global_rank == 0:
            cm_np = local_bld_cm.cpu().numpy()
            row_sums = cm_np.sum(axis=1)
            row_sums[row_sums == 0] = 1
            b_recall = np.diag(cm_np) / row_sums
            
            # Calculate mean IoU (excluding background)
            mean_iou = final_ious[1:].mean().item()
            
            print("=" * 60)
            print(f"VALIDATION REPORT (Adaptive Tri-Stream Epoch {epoch+1})")
            print("-" * 60)
            print(f"PIXEL-LEVEL mIoU (Buildings Only): {mean_iou:.4f}")
            print("-" * 60)
            print(f"{'Class':<12} | {'Pixel IoU':<10} | {'Bld Recall':<10}")
            print("-" * 60)
            
            for i in range(1, 5):
                print(f"{class_names[i]:<12} | {final_ious[i].item():.4f}     | {b_recall[i]:.4f}")
            
            print("-" * 60)
            print("BUILDING CONFUSION MATRIX")
            print("-" * 60)
            header = "".join([f"{name:>10}" for name in class_names])
            print(f"{'True v':>10} {header}")
            for i, row in enumerate(class_names):
                row_str = "".join([f"{val:10d}" for val in cm_np[i]])
                print(f"{row:>10} {row_str}")
            print("=" * 60)
            
            # Save best model
            if mean_iou > best_miou:
                best_miou = mean_iou
                torch.save(model.module.state_dict(), 
                         os.path.join(args.output_dir, "best_model.pth"))
                print(f"Saved best model with mIoU: {best_miou:.4f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'miou': mean_iou,
                }, os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pth"))
    
    cleanup()


if __name__ == "__main__":
    main()

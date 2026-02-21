'''
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 scripts/train_segformer_rgbd.py   \
    --data_root data/crasar_water   --output_dir runs/segformer_crasar_rgbd_final --batch_size 4
'''


import os
import argparse
import sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from transformers import SegformerForSemanticSegmentation, SegformerModel, SegformerConfig
from transformers.modeling_outputs import SemanticSegmenterOutput
from torchmetrics import JaccardIndex
from tqdm import tqdm
from PIL import Image
import albumentations as A
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class PixelContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, max_samples=1024):
        super().__init__()
        self.temperature = temperature
        self.max_samples = max_samples # Limit pixels to save memory

    def _sample_pixels(self, features, labels):
        """
        Randomly samples pixels from the features map for Background (0) and Buildings (1-4).
        Returns:
            anchors: [N, C] (Building pixels)
            negatives: [N, C] (Background pixels)
        """
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(-1, C) # [B*H*W, C]
        labels = labels.reshape(-1)                            # [B*H*W]

        # Identify indices
        bg_indices = (labels == 0).nonzero(as_tuple=True)[0]
        # Treat all classes 1-4 as "Building" for this specific contrast
        building_indices = (labels > 0).nonzero(as_tuple=True)[0]

        # If image has no buildings or no background, skip
        if len(building_indices) == 0 or len(bg_indices) == 0:
            return None, None

        # Downsample to save memory
        n_samples = min(len(building_indices), len(bg_indices), self.max_samples)
        
        # Random choice without replacement
        perm_build = torch.randperm(len(building_indices))[:n_samples]
        perm_bg = torch.randperm(len(bg_indices))[:n_samples]

        sampled_build = features[building_indices[perm_build]]
        sampled_bg = features[bg_indices[perm_bg]]

        return sampled_build, sampled_bg

    def forward(self, features, labels):
        """
        features: [Batch, Channels, H, W] (Output from Encoder or Decoder)
        labels: [Batch, H, W]
        """
        # Normalize features so dot product = cosine similarity
        features = F.normalize(features, p=2, dim=1)
        
        anchors, negatives = self._sample_pixels(features, labels)
        
        if anchors is None:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # --- Contrastive Logic ---
        # We want Anchors (Buildings) to be far from Negatives (Background).
        # Simple definition: Maximize distance (minimize similarity)
        
        # Calculate Similarity matrix [N, N]
        # We want to minimize the similarity between any Building and any Background
        similarity = torch.matmul(anchors, negatives.T) # [N, N]
        
        # We want these similarities to be LOW (negative or near 0).
        # We can treat this as a binary classification problem per pair, 
        # or simply penalize high similarity.
        
        # InfoNCE style:
        # logits = similarity / self.temperature
        # We want to minimize these logits. 
        # Since we strictly want to push them apart, we can simply take the mean exp:
        loss = torch.exp(similarity / self.temperature).mean()
        
        return loss


# --- Constants ---
# Metric: Ignore 0 to see mIoU of only buildings
IGNORE_INDEX_METRIC = 0 
# Loss: Keep 0 (background) so the model learns boundaries
IGNORE_INDEX_LOSS = -100 

import cv2

class InstanceMetrics:
    def __init__(self, num_classes=5, overlap_threshold=0.1):
        """
        overlap_threshold: Percentage (0.0 to 1.0) of GT pixels that must be 
                           covered by prediction to count as "detected".
                           0.1 means only 10% overlap is needed.
        """
        self.num_classes = num_classes
        self.overlap_threshold = overlap_threshold
        self.reset()

    def reset(self):
        self.total_gt_buildings = 0
        self.detected_buildings = 0
        self.correctly_classified_buildings = 0
        
        # Track per-class confusion (optional, but helpful)
        # matches[gt_class] = [correct_count, total_count]
        self.class_stats = {i: {'total': 0, 'detected': 0, 'correct': 0} for i in range(1, self.num_classes)}

    def update(self, preds, targets):
        """
        preds: [Batch, H, W] (Integer Class Indices)
        targets: [Batch, H, W] (Integer Class Indices)
        """
        # Convert to numpy for faster connected components processing
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()

        for i in range(len(preds_np)):
            self._process_single_image(preds_np[i], targets_np[i])

    def _process_single_image(self, pred, target):
        # 1. Find all Ground Truth instances (blobs)
        # We process all building classes (1-4) combined to find instances
        # or process them individually. Here we iterate unique classes present.
        
        present_classes = np.unique(target)
        present_classes = present_classes[present_classes != 0] # Ignore background

        for cls_id in present_classes:
            # Create binary mask for this class
            gt_mask = (target == cls_id).astype(np.uint8)
            
            # Find connected components (individual buildings)
            num_labels, labels_im = cv2.connectedComponents(gt_mask, connectivity=8)

            for label_idx in range(1, num_labels): # Skip 0 (background)
                # Extract Single GT Building Mask
                instance_mask = (labels_im == label_idx)
                instance_area = np.sum(instance_mask)
                
                self.total_gt_buildings += 1
                self.class_stats[cls_id]['total'] += 1

                # 2. Check overlap with Prediction
                # Look at the Prediction map ONLY where the GT building is
                pred_area = pred[instance_mask]
                
                # Check for ANY non-background prediction in this area
                # We count how many pixels are NOT class 0
                valid_pred_pixels = np.count_nonzero(pred_area)
                
                # Calculate Overlap Ratio
                overlap_ratio = valid_pred_pixels / instance_area
                
                if overlap_ratio >= self.overlap_threshold:
                    self.detected_buildings += 1
                    self.class_stats[cls_id]['detected'] += 1
                    
                    # 3. Check Classification
                    # Find the most frequent class predicted in this area
                    # (excluding 0 if possible, or just taking mode)
                    if valid_pred_pixels > 0:
                        # Get counts of predicted classes in this region
                        pred_classes, counts = np.unique(pred_area[pred_area != 0], return_counts=True)
                        if len(counts) > 0:
                            majority_class = pred_classes[np.argmax(counts)]
                            
                            if majority_class == cls_id:
                                self.correctly_classified_buildings += 1
                                self.class_stats[cls_id]['correct'] += 1

    def compute(self):
        # Avoid division by zero
        if self.total_gt_buildings == 0:
            return 0.0, 0.0

        detection_recall = self.detected_buildings / self.total_gt_buildings
        
        # Accuracy: Of the buildings we detected, how many were right?
        if self.detected_buildings == 0:
            classification_acc = 0.0
        else:
            classification_acc = self.correctly_classified_buildings / self.detected_buildings
            
        return detection_recall, classification_acc
    
    def print_report(self):
        recall, acc = self.compute()
        print(f"\n--- Instance-Level Metrics ---")
        print(f"Total Buildings (GT): {self.total_gt_buildings}")
        print(f"Detection Recall:     {recall:.4f} (Found {self.detected_buildings} buildings)")
        print(f"Classification Acc:   {acc:.4f} (Correct Type for {self.correctly_classified_buildings} buildings)")
        print("-" * 30)
        print(f"{'Class':<10} | {'Recall':<10} | {'Type Acc':<10}")
        for cls_id in self.class_stats:
            stats = self.class_stats[cls_id]
            if stats['total'] > 0:
                cls_recall = stats['detected'] / stats['total']
            else:
                cls_recall = 0.0
            
            if stats['detected'] > 0:
                cls_acc = stats['correct'] / stats['detected']
            else:
                cls_acc = 0.0
            print(f"Type {cls_id:<5} | {cls_recall:.4f}     | {cls_acc:.4f}")

# ==========================================
# 1. Model Architecture (Defined Locally)
# ==========================================

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
    
    
class RGBDSegformerFusion(nn.Module):
    def __init__(self, model_name, num_classes, dropout_prob=0.1, drop_path_rate=0.1):
        super().__init__()
        
        # Load Config and Inject Dropout
        config = SegformerConfig.from_pretrained(model_name)
        config.hidden_dropout_prob = dropout_prob
        config.attention_probs_dropout_prob = dropout_prob
        config.drop_path_rate = drop_path_rate
        config.num_labels = num_classes
        
        # 1. RGB Stream (Standard Encoder)
        self.base = SegformerForSemanticSegmentation.from_pretrained(
            model_name, 
            config=config,
            ignore_mismatched_sizes=True,
            use_safetensors=True
        )
        self.rgb_encoder = self.base.segformer
        
        # 2. Depth Stream (Separate Encoder, same architecture)
        self.depth_encoder = SegformerModel.from_pretrained(
            model_name, 
            config=config,
            use_safetensors=True
        )
        
        # 3. Cross Attention Fusion Modules (One per stage)
        hidden_sizes = config.hidden_sizes
        self.fusion_layers = nn.ModuleList([
            CrossViewAttention(dim) for dim in hidden_sizes
        ])
        
        # 4. Decoder
        self.decode_head = self.base.decode_head

    def forward(self, rgb_images, depth_images, labels=None):
        # A. Encode RGB
        rgb_outputs = self.rgb_encoder(rgb_images, output_hidden_states=True)
        rgb_features = rgb_outputs.hidden_states 
        
        # B. Encode Depth (Repeat 1 channel -> 3 channels)
        depth_input = depth_images.repeat(1, 3, 1, 1)
        depth_outputs = self.depth_encoder(depth_input, output_hidden_states=True)
        depth_features = depth_outputs.hidden_states

        # C. Fusion
        fused_features = []
        for i, (rgb_f, depth_f) in enumerate(zip(rgb_features, depth_features)):
            fused = self.fusion_layers[i](rgb_f, depth_f)
            fused_features.append(fused)

        # # D. Decode
        # logits = self.decode_head(fused_features)
        
        # # Upsample
        # upsampled_logits = nn.functional.interpolate(
        #     logits, 
        #     size=rgb_images.shape[-2:], 
        #     mode="bilinear", 
        #     align_corners=False
        # )

        # loss = None
        # if labels is not None:
        #     loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX_LOSS)(upsampled_logits, labels)

        # # Return object compatible with HF
        # from transformers.modeling_outputs import SemanticSegmenterOutput
        # return SemanticSegmenterOutput(
        #     loss=loss,
        #     logits=upsampled_logits,
        #     hidden_states=None,
        #     attentions=None
        # )
        
        feature_for_loss = fused_features[-1] 
        
        logits = self.decode_head(fused_features)
        
        # Upsample Logits
        upsampled_logits = nn.functional.interpolate(logits, size=rgb_images.shape[-2:], mode="bilinear", align_corners=False)
        
        # Also Upsample Features to match label size for pixel sampling
        upsampled_features = nn.functional.interpolate(feature_for_loss, size=rgb_images.shape[-2:], mode="bilinear", align_corners=False)

        loss = None
        if labels is not None:
            # ce_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX_LOSS)(upsampled_logits, labels)
            
            
            # We calculate contrastive loss outside or inside. 
            # Let's return the necessary components.
            # loss = ce_loss # Placeholder, we add contrastive in the loop
            loss = FocalLoss(ignore_index=IGNORE_INDEX_LOSS)(upsampled_logits, labels)

        # We stick the features in 'hidden_states' so we can access them in the loop
        return SemanticSegmenterOutput(
            loss=loss, 
            logits=upsampled_logits, 
            hidden_states=upsampled_features
        )

# ==========================================
# 2. Transforms & Dataset
# ==========================================

def get_geometric_transforms():
    """Applied to RGB, Depth, and Mask identically."""
    return A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.5),
    ], additional_targets={'depth': 'image'}) 

def get_pixel_transforms():
    """Applied ONLY to RGB."""
    return A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(p=1),
            A.HueSaturationValue(p=1),
            A.RGBShift(p=1)
        ], p=0.3),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
    ])

class RGBDBuildingDataset(Dataset):
    def __init__(self, data_root, split):
        self.split_dir = os.path.join(data_root, split)
        self.image_dir = os.path.join(self.split_dir, "images")
        self.depth_dir = os.path.join(self.split_dir, "depth")
        self.mask_dir = os.path.join(self.split_dir, "labels")
        
        valid_exts = ('.png', '.jpg', '.jpeg', '.tif')
        self.images = sorted([f for f in os.listdir(self.image_dir) if f.endswith(valid_exts)])
        self.masks = sorted([f for f in os.listdir(self.mask_dir) if f.endswith(valid_exts)])
        self.depths = sorted([f for f in os.listdir(self.depth_dir) if f.endswith(valid_exts)])

        # Validation
        assert len(self.images) == len(self.masks) == len(self.depths), \
            f"Mismatch in {split}: Imgs={len(self.images)}, Masks={len(self.masks)}, Depth={len(self.depths)}"

        # Transforms Setup
        if split == "train":
            self.geo_transform = get_geometric_transforms()
            self.pixel_transform = get_pixel_transforms()
        else:
            # Test/Val: Only resize
            self.geo_transform = A.Compose([A.Resize(512, 512)], additional_targets={'depth': 'image'})
            self.pixel_transform = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load Data
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        depth_path = os.path.join(self.depth_dir, self.depths[idx])

        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        depth = np.array(Image.open(depth_path)) # Keep as 2D array first

        # 1. Apply Geometric Transforms (RGB, Depth, Mask)
        if self.geo_transform:
            augmented = self.geo_transform(image=img, mask=mask, depth=depth)
            img = augmented['image']
            mask = augmented['mask']
            depth = augmented['depth']

        # 2. Apply Pixel Transforms (RGB Only)
        if self.pixel_transform:
            augmented_px = self.pixel_transform(image=img)
            img = augmented_px['image']

        # 3. Normalization & Tensor Conversion
        # RGB
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        img = torch.tensor(img).permute(2, 0, 1).float() # [3, H, W]

        # Depth
        depth = depth.astype(np.float32)
        if depth.max() > 0:
            depth = depth / 255.0 # Normalize 8-bit depth to 0-1
        depth = torch.tensor(depth).unsqueeze(0).float() # [1, H, W]

        # Mask
        mask = torch.tensor(mask).long()

        return {
            "rgb": img,
            "depth": depth,
            "labels": mask
        }

# ==========================================
# 3. Main Training Script
# ==========================================

def setup_distributed():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_distributed():
    destroy_process_group()

def get_args():
    parser = argparse.ArgumentParser(description="Train RGB-D SegFormer Fusion")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--model_name", type=str, default="nvidia/mit-b0")
    parser.add_argument("--num_workers", type=int, default=4)
    # Classes: 0 (Back) + 4 Buildings = 5 Total
    parser.add_argument("--num_classes", type=int, default=5)
    return parser.parse_args()

def main():
    args = get_args()
    
    # --- Distributed Setup ---
    is_distributed = "LOCAL_RANK" in os.environ
    if is_distributed:
        setup_distributed()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if local_rank == 0:
        print(f"--- RGB-D Fusion Training ---")
        print(f"Data Root: {args.data_root}")
        print(f"Augmentation: Enabled (Split Geo/Pixel)")
        print(f"Regularization: Drop={args.dropout}, WD={args.weight_decay}")

    # --- Datasets ---
    train_dataset = RGBDBuildingDataset(args.data_root, "train")
    val_dataset = RGBDBuildingDataset(args.data_root, "test")

    if is_distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), 
        sampler=train_sampler, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        sampler=val_sampler, num_workers=args.num_workers, pin_memory=True
    )

    # --- Model ---
    model = RGBDSegformerFusion(
        model_name=args.model_name,
        num_classes=args.num_classes,
        dropout_prob=args.dropout,
        drop_path_rate=0.1
    )
    model.to(device)
    
    if is_distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank])

    # --- Optimization ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Metrics
    # Average="none" returns IoU for each class separately
    iou_metric_per_class = JaccardIndex(
        task="multiclass", 
        num_classes=args.num_classes, 
        average="none"
    ).to(device)

    # --- Training Loop ---
    scaler = torch.amp.GradScaler("cuda")
    best_iou = 0.0

    for epoch in range(args.epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)
            
        model.train()
        train_loss = 0.0
        
        if local_rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        else:
            pbar = train_loader

        for batch in pbar:
            rgb = batch["rgb"].to(device)
            depth = batch["depth"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = model(rgb_images=rgb, depth_images=depth, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            
            # --- FIX: Unscale and Clip Gradients ---
            scaler.unscale_(optimizer) 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            
            if local_rank == 0 and isinstance(pbar, tqdm):
                pbar.set_postfix({"loss": loss.item()})

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        
        # Initialize Metrics
        iou_metric_per_class.reset()
        instance_metric = InstanceMetrics(num_classes=args.num_classes, overlap_threshold=0.1) # <--- NEW
        
        if local_rank == 0:
            print("Running Validation...")
            
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch["rgb"].to(device)
                depth = batch["depth"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast("cuda"):
                    outputs = model(rgb_images=rgb, depth_images=depth, labels=labels)
                    val_loss += outputs.loss.item()
                    logits = outputs.logits

                preds = torch.argmax(logits, dim=1)
                
                # Update standard IoU
                iou_metric_per_class.update(preds, labels)
                
                # Update New Instance Metric (Pass raw tensors)
                instance_metric.update(preds, labels)

        # --- Metrics Calculation ---
        # Returns tensor [IoU_0, IoU_1, IoU_2, IoU_3, IoU_4]
        per_class_iou = iou_metric_per_class.compute()
        
        # Calculate Mean IoU for BUILDINGS only (indices 1-4)
        # Class 0 is Background
        building_iou_tensor = per_class_iou[1:]
        mean_building_iou = building_iou_tensor.mean().item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        if local_rank == 0:
            print(f"\n--- Epoch {epoch+1} Summary ---")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss:   {avg_val_loss:.4f}")
            print(f"  Mean Building IoU: {mean_building_iou:.4f}")
            
            print("  Per-Class IoU:")
            class_names = ["Background", "Build_Type1", "Build_Type2", "Build_Type3", "Build_Type4"]
            for i, score in enumerate(per_class_iou):
                c_name = class_names[i] if i < len(class_names) else f"Class {i}"
                print(f"    {c_name}: {score:.4f}")

            # Save Best Model Manually
            if mean_building_iou > best_iou:
                best_iou = mean_building_iou
                save_path = os.path.join(args.output_dir, "best_model")
                os.makedirs(save_path, exist_ok=True)
                
                # Unwrap model
                model_to_save = model.module if hasattr(model, "module") else model
                
                # Save Weights
                torch.save(model_to_save.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
                
                # Save Config Info
                with open(os.path.join(save_path, "config.txt"), "w") as f:
                    f.write(f"Model: {args.model_name}\n")
                    f.write(f"Classes: {args.num_classes}\n")
                    f.write(f"Best IoU: {best_iou:.4f}\n")
                    
                print(f"  --> Saved New Best Model to {save_path}")
            
            # Print New Metrics
            instance_metric.print_report() 

    if is_distributed:
        cleanup_distributed()

if __name__ == "__main__":
    main()
    
    
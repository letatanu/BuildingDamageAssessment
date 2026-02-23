#!/usr/bin/env python3
"""
Train SegFormer on FloodNet-formatted datasets.
Supports standard RGB (3-channel) or RGB+DEM (4-channel) training.
"""
import os
import glob
import math
import json
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)
import albumentations as A

# ---------------------------------------------------------
# 1. Dataset Definition (FloodNet Format)
# ---------------------------------------------------------
class FloodNetSegmentationDataset(Dataset):
    """
    Expects FloodNet directory structure:
        root_dir/
            <split>-org-img/    (e.g., 0.png)
            <split>-label-img/  (e.g., 0_lab.png)
            <split>-dem/        (e.g., 0_dem.tif) - Optional
    """
    def __init__(self, root_dir, image_processor, augment=False, use_dem=False):
        self.root_dir = root_dir
        self.image_processor = image_processor
        self.augment = augment
        self.use_dem = use_dem
        
        # Automatically detect the folders based on suffix
        try:
            self.images_dir = glob.glob(os.path.join(root_dir, "*-org-img"))[0]
            self.labels_dir = glob.glob(os.path.join(root_dir, "*-label-img"))[0]
            if self.use_dem:
                self.dem_dir = glob.glob(os.path.join(root_dir, "*-dem"))[0]
        except IndexError:
            raise ValueError(f"Could not find standard FloodNet folders (*-org-img, *-label-img) in {root_dir}")
        
        # Find all images
        self.image_paths = sorted(glob.glob(os.path.join(self.images_dir, "*.*")))
        self.samples = []
        
        for img_path in self.image_paths:
            basename = os.path.basename(img_path)
            name_no_ext = os.path.splitext(basename)[0]
            
            # FloodNet label naming convention: {name}_lab.png
            lbl_path = os.path.join(self.labels_dir, f"{name_no_ext}_lab.png")
            
            if self.use_dem:
                dem_path = os.path.join(self.dem_dir, f"{name_no_ext}_dem.tif")
                if os.path.exists(lbl_path) and os.path.exists(dem_path):
                    self.samples.append((img_path, lbl_path, dem_path))
            else:
                if os.path.exists(lbl_path):
                    self.samples.append((img_path, lbl_path, None))
                
        print(f"Found {len(self.samples)} valid samples in {root_dir} (Use DEM: {self.use_dem})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path, dem_path = self.samples[idx]
        
        image = Image.open(img_path).convert("RGB")
        label = Image.open(lbl_path) 
        
        image_np = np.array(image)
        label_np = np.array(label)
        
        dem_np = None
        if self.use_dem:
            import rasterio
            with rasterio.open(dem_path) as src:
                dem_np = src.read(1) # Read first band

        # Apply augmentations
        if self.augment:
            if self.use_dem:
                # Spatial augmentations must apply to RGB, Mask, and DEM equally
                transform = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3)
                ], additional_targets={'dem': 'mask'})
                augmented = transform(image=image_np, mask=label_np, dem=dem_np)
                image_np = augmented['image']
                label_np = augmented['mask']
                dem_np = augmented['dem']
            else:
                transform = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3)
                ])
                augmented = transform(image=image_np, mask=label_np)
                image_np = augmented['image']
                label_np = augmented['mask']

        # Process RGB with HuggingFace processor (Normalizes & converts to Tensor)
        encoded = self.image_processor(image_np, segmentation_maps=label_np, return_tensors="pt")
        pixel_values = encoded["pixel_values"].squeeze(0) # Shape: [3, H, W]
        labels = encoded["labels"].squeeze(0).long()

        if self.use_dem:
            # Normalize DEM (simple Min-Max or standard scaling)
            # Clip extreme outliers then scale 0-1
            dem_np = np.clip(dem_np, -10, 50) 
            dem_min, dem_max = dem_np.min(), dem_np.max()
            if dem_max > dem_min:
                dem_norm = (dem_np - dem_min) / (dem_max - dem_min)
            else:
                dem_norm = np.zeros_like(dem_np)
            
            dem_tensor = torch.from_numpy(dem_norm).unsqueeze(0).float() # Shape: [1, H, W]
            
            # Concatenate RGB and DEM to create a 4-channel tensor
            pixel_values = torch.cat([pixel_values, dem_tensor], dim=0) # Shape: [4, H, W]
        
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

# ---------------------------------------------------------
# 2. Modify Model for 4 Channels (RGB + DEM)
# ---------------------------------------------------------
def adapt_model_for_4channels(model):
    """
    Modifies the first convolutional layer of SegFormer to accept 4 input channels instead of 3.
    """
    old_conv = model.segformer.encoder.patch_embeddings[0].proj
    
    # Create new conv layer with 4 input channels
    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None)
    )
    
    # Copy pre-trained weights for the first 3 channels (RGB)
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = old_conv.weight
        # Initialize the 4th channel (DEM) weights as the mean of the RGB weights
        new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
        if old_conv.bias is not None:
            new_conv.bias = old_conv.bias
            
    # Replace the layer in the model
    model.segformer.encoder.patch_embeddings[0].proj = new_conv
    print("Successfully adapted model to accept 4-channel (RGB + DEM) input.")
    return model

# ---------------------------------------------------------
# 3. Metric Computation (mIoU)
# ---------------------------------------------------------
def compute_metrics(eval_pred, num_classes=5, ignore_index=255):
    with torch.no_grad():
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
            
        logits_tensor = torch.from_numpy(logits)
        labels_tensor = torch.from_numpy(labels)
        
        logits_tensor = torch.nn.functional.interpolate(
            logits_tensor,
            size=labels_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        
        preds = logits_tensor.argmax(dim=1).numpy()
        labels = labels_tensor.numpy()

        valid_mask = (labels != ignore_index)
        if not np.any(valid_mask):
            return {"mIoU": 0.0}
            
        preds = preds[valid_mask]
        labels = labels[valid_mask]

        bincount = np.bincount(num_classes * labels + preds, minlength=num_classes**2)
        confusion_matrix = bincount.reshape(num_classes, num_classes)
        
        tp = np.diag(confusion_matrix)
        fp = confusion_matrix.sum(axis=0) - tp
        fn = confusion_matrix.sum(axis=1) - tp
        
        union = tp + fp + fn
        ious = np.divide(tp, union, out=np.zeros_like(tp, dtype=float), where=union!=0)
        
        present_classes = union > 0
        miou = np.mean(ious[present_classes]) if np.any(present_classes) else 0.0

        metrics = {"mIoU": miou}
        for i in range(num_classes):
            metrics[f"IoU_class_{i}"] = ious[i]

        return metrics

# ---------------------------------------------------------
# 4. Main Training Loop
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True, help="Path to FloodNet train split dir")
    parser.add_argument("--test_data", type=str, required=True, help="Path to FloodNet test/val split dir")
    parser.add_argument("--output_dir", type=str, default="./segformer_output")
    parser.add_argument("--model_name", type=str, default="nvidia/mit-b2")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--use_dem", action="store_true", help="If passed, will use DEM as 4th channel")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # 1. Initialize Processor
    image_processor = SegformerImageProcessor(
        do_resize=False,
        do_normalize=True,
        reduce_labels=False
    )

    # 2. Build Datasets
    print(f"Loading training dataset from {args.train_data}...")
    train_ds = FloodNetSegmentationDataset(args.train_data, image_processor, augment=True, use_dem=args.use_dem)
    
    print(f"Loading validation dataset from {args.test_data}...")
    val_ds = FloodNetSegmentationDataset(args.test_data, image_processor, augment=False, use_dem=args.use_dem)

    id2label = {0: "Background", 1: "No Damage", 2: "Minor Damage", 3: "Major Damage", 4: "Destroyed"}
    label2id = {v: k for k, v in id2label.items()}

    # 3. Initialize Model
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model_name,
        num_labels=args.num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # 3.5 Adjust Model for DEM if needed
    if args.use_dem:
        model = adapt_model_for_4channels(model)

    def collate_fn(batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
        labels = torch.stack([b["labels"] for b in batch], dim=0)
        return {"pixel_values": pixel_values, "labels": labels}

    # 4. Training Arguments
    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    save_steps = max(steps_per_epoch, 100)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="mIoU",
        greater_is_better=True,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        fp16=torch.cuda.is_available(), 
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, num_classes=args.num_classes),
    )

    # 6. Train!
    print("Starting training...")
    trainer.train()

    # 7. Evaluate and Save
    metrics = trainer.evaluate()
    print("Final Evaluation Metrics:")
    print(json.dumps(metrics, indent=2))

    trainer.save_model(args.output_dir)
    image_processor.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()

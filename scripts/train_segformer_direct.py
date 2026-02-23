#!/usr/bin/env python3
import os
import glob
import math
import json
import argparse
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ---------------------------------------------------------
# 1. Dataset Definition
# ---------------------------------------------------------
class DirectSegmentationDataset(Dataset):
    """
    Expects a directory structure like:
        root_dir/
            images/ (RGB .png or .jpg)
            labels/ (Grayscale mask .png)
    """
    def __init__(self, root_dir, image_processor, augment=False):
        self.root_dir = root_dir
        self.image_processor = image_processor
        self.augment = augment
        
        self.images_dir = os.path.join(root_dir, "images")
        self.labels_dir = os.path.join(root_dir, "labels")
        
        # Find all images
        self.image_paths = sorted(glob.glob(os.path.join(self.images_dir, "*.*")))
        self.samples = []
        
        for img_path in self.image_paths:
            basename = os.path.basename(img_path)
            # Match image extension to label extension (assuming labels are .png)
            name_no_ext = os.path.splitext(basename)[0]
            lbl_path = os.path.join(self.labels_dir, f"{name_no_ext}.png")
            
            if os.path.exists(lbl_path):
                self.samples.append((img_path, lbl_path))
                
        print(f"Found {len(self.samples)} valid image-label pairs in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        
        image = Image.open(img_path).convert("RGB")
        label = Image.open(lbl_path) # Mask, usually 1-channel L or P

        # Convert to numpy
        image_np = np.array(image)
        label_np = np.array(label)

        # Apply basic augmentations if training
        if self.augment:
            import albumentations as A
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3)
            ])
            augmented = transform(image=image_np, mask=label_np)
            image_np = augmented['image']
            label_np = augmented['mask']

        # Process with HuggingFace processor (Normalizes & converts to Tensor)
        # We set do_resize=False because tiles are already 512x512
        encoded = self.image_processor(image_np, segmentation_maps=label_np, return_tensors="pt")
        
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),  # Remove batch dim
            "labels": encoded["labels"].squeeze(0).long()
        }

# ---------------------------------------------------------
# 2. Metric Computation (mIoU)
# ---------------------------------------------------------
def compute_metrics(eval_pred, num_classes=5, ignore_index=255):
    with torch.no_grad():
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
            
        logits_tensor = torch.from_numpy(logits)
        labels_tensor = torch.from_numpy(labels)
        
        # SegFormer outputs logits at 1/4 the original resolution. 
        # We must upsample them to match the label size before argmax.
        logits_tensor = torch.nn.functional.interpolate(
            logits_tensor,
            size=labels_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        
        preds = logits_tensor.argmax(dim=1).numpy()
        labels = labels_tensor.numpy()

        # Compute confusion matrix
        valid_mask = (labels != ignore_index)
        
        # Ensure we only compute for valid pixels
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
        
        # Avoid division by zero
        ious = np.divide(tp, union, out=np.zeros_like(tp, dtype=float), where=union!=0)
        
        # Calculate mean IoU across classes that are actually present
        present_classes = union > 0
        miou = np.mean(ious[present_classes]) if np.any(present_classes) else 0.0

        metrics = {"mIoU": miou}
        for i in range(num_classes):
            metrics[f"IoU_class_{i}"] = ious[i]

        return metrics

# ---------------------------------------------------------
# 3. Main Training Loop
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data dir (must contain images/ and labels/)")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test/val data dir (must contain images/ and labels/)")
    parser.add_argument("--output_dir", type=str, default="./segformer_output")
    parser.add_argument("--model_name", type=str, default="nvidia/mit-b2")
    parser.add_argument("--num_classes", type=int, default=5) # 0=Bg, 1=No, 2=Minor, 3=Major, 4=Dest
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # 1. Initialize Processor
    image_processor = SegformerImageProcessor(
        do_resize=False,        # Assuming your tiles are already sized
        do_normalize=True,      # Standard ImageNet normalization
        reduce_labels=False     # We mapped our labels explicitly (0-4)
    )

    # 2. Build Datasets
    print("Loading training dataset...")
    train_ds = DirectSegmentationDataset(args.train_data, image_processor, augment=True)
    
    print("Loading validation dataset...")
    val_ds = DirectSegmentationDataset(args.test_data, image_processor, augment=False)

    # Label Mappings (For HuggingFace config)
    id2label = {0: "Background/Unclassified", 1: "No Damage", 2: "Minor Damage", 3: "Major Damage", 4: "Destroyed"}
    label2id = {v: k for k, v in id2label.items()}

    # 3. Initialize Model
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model_name,
        num_labels=args.num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True, # Essential if fine-tuning from Cityscapes (19 classes) to 5 classes
    )

    # 4. Collate Function
    def collate_fn(batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
        labels = torch.stack([b["labels"] for b in batch], dim=0)
        return {"pixel_values": pixel_values, "labels": labels}

    # 5. Training Arguments
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
        fp16=torch.cuda.is_available(), # Enable mixed precision if GPU available
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, num_classes=args.num_classes),
    )

    # 7. Train!
    print("Starting training...")
    trainer.train()

    # 8. Evaluate and Save
    metrics = trainer.evaluate()
    print("Final Evaluation Metrics:")
    print(json.dumps(metrics, indent=2))

    trainer.save_model(args.output_dir)
    image_processor.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()

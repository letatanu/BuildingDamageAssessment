import os
import argparse
from models.model_fusion import RGBDSegformerFusion
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from torchmetrics import JaccardIndex
from tqdm import tqdm
from PIL import Image
import numpy as np

# --- Constants ---
IGNORE_INDEX_METRIC = 0 
IGNORE_INDEX_LOSS = -100 

def setup_distributed():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_distributed():
    destroy_process_group()

def get_rgbd_segformer(model_name, num_classes):
    """
    Loads a SegFormer model and modifies the input layer to accept 4 channels (RGB+D).
    """
    # 1. Load standard RGB model
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    # 2. Modify Configuration
    model.config.num_channels = 4
    
    # 3. Access the first Patch Embedding layer (MiT-b0 to b5 structure)
    # The structure is usually: model.segformer.encoder.patch_embeddings[0].proj
    old_proj = model.segformer.encoder.patch_embeddings[0].proj
    
    # 4. Create new Conv2d layer with 4 input channels
    new_proj = nn.Conv2d(
        in_channels=4,
        out_channels=old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding
    )
    
    # 5. Initialize Weights
    # Copy RGB weights
    with torch.no_grad():
        new_proj.weight[:, :3] = old_proj.weight
        # Initialize Depth weights as average of RGB (better than random)
        new_proj.weight[:, 3] = torch.mean(old_proj.weight, dim=1)
        # Copy bias
        new_proj.bias = old_proj.bias

    # 6. Replace the layer
    model.segformer.encoder.patch_embeddings[0].proj = new_proj
    
    return model

class RGBDBuildingDataset(Dataset):
    def __init__(self, data_root, split, processor):
        """
        Expects structure:
        data_root/
          split/ (train or test)
            images/
            depth/
            labels/
        """
        self.split_dir = os.path.join(data_root, split)
        self.image_dir = os.path.join(self.split_dir, "images")
        self.depth_dir = os.path.join(self.split_dir, "depth")
        self.mask_dir = os.path.join(self.split_dir, "labels")
        self.processor = processor
        
        # Filter for valid extensions
        valid_exts = ('.png', '.jpg', '.jpeg', '.tif')
        self.images = sorted([f for f in os.listdir(self.image_dir) if f.endswith(valid_exts)])
        self.masks = sorted([f for f in os.listdir(self.mask_dir) if f.endswith(valid_exts)])
        self.depths = sorted([f for f in os.listdir(self.depth_dir) if f.endswith(valid_exts)])

        # Validation
        assert len(self.images) == len(self.masks) == len(self.depths), \
            f"Mismatch counts in {split}: Imgs={len(self.images)}, Masks={len(self.masks)}, Depth={len(self.depths)}"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Paths
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        depth_path = os.path.join(self.depth_dir, self.depths[idx])

        # Load Data
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path) # Int indices 0-4
        depth = Image.open(depth_path) # Likely grayscale (L) or 16-bit (I)

        # 1. Process RGB and Mask using HuggingFace Processor
        # We process RGB first to get the correct resizing and ImageNet normalization
        inputs = self.processor(
            images=image, 
            segmentation_maps=mask, 
            return_tensors="pt"
        )
        
        pixel_values_rgb = inputs["pixel_values"].squeeze(0) # [3, H, W]
        labels = inputs["labels"].squeeze(0).long()          # [H, W]

        # 2. Process Depth
        # We must manually resize depth to match the processed RGB output size
        target_h, target_w = pixel_values_rgb.shape[-2:]
        depth = depth.resize((target_w, target_h), resample=Image.NEAREST)
        
        # Convert to Tensor
        depth_tensor = torch.tensor(np.array(depth), dtype=torch.float32)
        
        # Normalize Depth
        # Adjust this depending on your depth data.
        # If 0-255 uint8: divide by 255. 
        # If real world meters: perhaps divide by max depth or standard deviation.
        # Here we assume simple 0-1 scaling for stability.
        if depth_tensor.max() > 0:
            depth_tensor = depth_tensor / 255.0 # Assuming 8-bit depth
            # If 16-bit (0-65535), use: depth_tensor = depth_tensor / 65535.0
        
        # Ensure depth is [1, H, W]
        if depth_tensor.ndim == 2:
            depth_tensor = depth_tensor.unsqueeze(0)
            
        # 3. Concatenate [RGB (3) + Depth (1)] -> [4, H, W]
        pixel_values_rgbd = torch.cat([pixel_values_rgb, depth_tensor], dim=0)

        return {
            "pixel_values": pixel_values_rgbd,
            "labels": labels
        }

def get_args():
    parser = argparse.ArgumentParser(description="Train RGB-D SegFormer")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--model-name", type=str, default="nvidia/segformer-b2-finetuned-ade-512-512")
    parser.add_argument("--num-classes", type=int, default=5)
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
        print(f"--- RGB-D SegFormer Training ---")
        print(f"Data Root: {args.data_root}")
        print(f"Classes: {args.num_classes} (Ignoring Class 0 in mIoU metric)")

    # --- Processor ---
    # We use this primarily for resizing and RGB normalization
    processor = SegformerImageProcessor.from_pretrained(
        args.model_name, 
        do_reduce_labels=False, 
        size={"height": 512, "width": 512}
    )

    # --- Datasets ---
    # Assumes data_root/train and data_root/test exist
    train_dataset = RGBDBuildingDataset(args.data_root, "train", processor)
    val_dataset = RGBDBuildingDataset(args.data_root, "test", processor)

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
    # Use our custom function to load RGB+D model
    # model = get_rgbd_segformer(args.model_name, args.num_classes)
    # New
    model = RGBDSegformerFusion(args.model_name, args.num_classes)
    model.to(device)
    
    if is_distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank])

    # --- Optimization ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Metric: IoU
    iou_metric = JaccardIndex(
        task="multiclass", 
        num_classes=args.num_classes, 
        ignore_index=IGNORE_INDEX_METRIC
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
            # 1. Get combined 4-channel input
            combined_input = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            # 2. Split into RGB (3 channels) and Depth (1 channel)
            rgb_imgs = combined_input[:, :3, :, :]
            depth_imgs = combined_input[:, 3:, :, :]

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                # 3. Pass separated inputs
                outputs = model(rgb_images=rgb_imgs, depth_images=depth_imgs, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            
            if local_rank == 0 and isinstance(pbar, tqdm):
                pbar.set_postfix({"loss": loss.item()})

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        iou_metric.reset()
        
        if local_rank == 0:
            print("Running Validation on Test Set...")
            
        with torch.no_grad():
            for batch in val_loader:
                # 1. Get combined 4-channel input
                combined_input = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                
                # 2. Split into RGB (3 channels) and Depth (1 channel)
                rgb_imgs = combined_input[:, :3, :, :]
                depth_imgs = combined_input[:, 3:, :, :]

                with torch.amp.autocast("cuda"):
                    # 3. Pass separated inputs to the Fusion Model
                    outputs = model(rgb_images=rgb_imgs, depth_images=depth_imgs, labels=labels)
                    loss = outputs.loss
                    val_loss += loss.item()
                    
                    logits = outputs.logits # Already upsampled in the model class

                preds = torch.argmax(logits, dim=1)
                iou_metric.update(preds, labels)

        # Sync and Compute Metrics
        final_iou = iou_metric.compute().item()
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        if local_rank == 0:
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss:   {avg_val_loss:.4f}")
            print(f"  Building mIoU (Classes 1-4): {final_iou:.4f}")

            # Save Best
            if final_iou > best_iou:
                best_iou = final_iou
                save_path = os.path.join(args.output_dir, "best_model")
                os.makedirs(save_path, exist_ok=True)
                
                # Unwrap model from DDP
                model_to_save = model.module if hasattr(model, "module") else model
                
                # --- FIX: Save State Dict Manually ---
                torch.save(model_to_save.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
                
                # Save processor (this still works because it is a standard HF object)
                processor.save_pretrained(save_path)
                
                # Save config so you remember hyperparameters
                with open(os.path.join(save_path, "config.txt"), "w") as f:
                    f.write(f"Model: {args.model_name}\nClasses: {args.num_classes}")
                    
                print(f"  --> Saved new best model to {save_path}")
                
    if is_distributed:
        cleanup_distributed()

if __name__ == "__main__":
    main()
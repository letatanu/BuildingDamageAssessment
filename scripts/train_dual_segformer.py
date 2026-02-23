'''
python scripts/train_dual_segformer.py \
    --config_file datasets/configs/segformer_xBD.py \
    --train_dir data/xBD_tiled \
    --val_dirs data/xBD_tiled data/CRASAR-tiles
'''

import os, sys, json, math, random
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from transformers import Trainer, set_seed
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix

# Import FloodNet dataset loaders and Models
from nh_datasets.floodnet import FloodNetSegDataset
from models.dual_siamese_segformer import DualHeadSiameseSegFormer
from transformers import SegformerImageProcessor

from scripts.utils import (
    setup_devices_autodetect,
    safe_training_args,
    compute_mIoU,
    choose_resume_checkpoint,
    parse_args,
)

# ---------------------------------------------------------
# 0) Dataset Wrappers & Auto-Detect Factory
# ---------------------------------------------------------
class PairedXBDFloodNetDataset(Dataset):
    """ Custom Paired loader that does not require pre-labels and synchronizes augmentations """
    def __init__(self, root_dir, split_base, image_processor, num_classes=5, augment=False, image_size=512):
        self.root = Path(root_dir)
        self.split_base = split_base
        self.ip = image_processor
        self.augment = augment
        self.image_size = image_size
        self.num_classes = num_classes

        self.pre_img_dir = self.root / f"{split_base}_pre" / f"{split_base}_pre-org-img"
        self.post_img_dir = self.root / f"{split_base}" / f"{split_base}-org-img"
        self.post_lbl_dir = self.root / f"{split_base}" / f"{split_base}-label-img"

        if not self.pre_img_dir.is_dir(): raise FileNotFoundError(f"Missing pre image dir: {self.pre_img_dir}")
        if not self.post_img_dir.is_dir(): raise FileNotFoundError(f"Missing post image dir: {self.post_img_dir}")
        if not self.post_lbl_dir.is_dir(): raise FileNotFoundError(f"Missing post label dir: {self.post_lbl_dir}")

        IMG_EXTS = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"}
        self.samples = []

        for fname in os.listdir(self.post_img_dir):
            stem, ext = os.path.splitext(fname)
            if ext not in IMG_EXTS:
                continue
                
            post_img_p = self.post_img_dir / fname
            
            # 1. Match Post Label
            post_lbl_p = None
            for ext2 in IMG_EXTS:
                cand = self.post_lbl_dir / f"{stem}_lab{ext2}"
                if cand.exists():
                    post_lbl_p = cand
                    break
            if not post_lbl_p:
                continue
                
            # 2. Match Pre Image
            pre_stem = stem.replace("post_disaster", "pre_disaster")
            pre_img_p = None
            for ext3 in IMG_EXTS:
                cand = self.pre_img_dir / f"{pre_stem}{ext3}"
                if cand.exists():
                    pre_img_p = cand
                    break
            
            if pre_img_p and pre_img_p.exists():
                self.samples.append((pre_img_p, post_img_p, post_lbl_p))

        if len(self.samples) == 0:
            raise RuntimeError(f"No paired matching samples found between {self.pre_img_dir} and {self.post_img_dir}")
            
        self.rng = random.Random(1337)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pre_img_p, post_img_p, post_lbl_p = self.samples[idx]

        pre_img = Image.open(pre_img_p).convert("RGB")
        post_img = Image.open(post_img_p).convert("RGB")
        lab = Image.open(post_lbl_p)
        if lab.mode != "L":
            lab = lab.convert("L")

        # Synchronously mirror Pre, Post, and Label
        if self.augment and self.rng.random() < 0.5:
            pre_img = pre_img.transpose(Image.FLIP_LEFT_RIGHT)
            post_img = post_img.transpose(Image.FLIP_LEFT_RIGHT)
            lab = lab.transpose(Image.FLIP_LEFT_RIGHT)

        pre_img = pre_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        post_img = post_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        lab = lab.resize((self.image_size, self.image_size), Image.NEAREST)

        lab_np = np.array(lab, dtype=np.int64)

        encoded_pre = self.ip(images=pre_img, return_tensors="pt")
        encoded_post = self.ip(images=post_img, return_tensors="pt")

        # FORCE CONTIGUOUS CLONES TO PREVENT CUDA MISALIGNED ADDRESS CRASH
        return {
            "pixel_values_pre": encoded_pre["pixel_values"].squeeze(0).clone().contiguous(),
            "pixel_values_post": encoded_post["pixel_values"].squeeze(0).clone().contiguous(),
            "labels": torch.from_numpy(lab_np).clone().contiguous(),
            "id": post_img_p.stem
        }

class SingleImageFloodNetDataset(Dataset):
    """ Wraps single-image datasets (e.g., CRASAR) providing a blank pre-disaster tensor """
    def __init__(self, root_dir, split, image_processor, num_classes=5, augment=False):
        self.dataset = FloodNetSegDataset(
            root=root_dir, split=split, image_processor=image_processor, augment=augment, num_classes=num_classes
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        pixel_values_post = item["pixel_values"]
        pixel_values_pre = torch.zeros_like(pixel_values_post)

        # FORCE CONTIGUOUS CLONES
        return {
            "pixel_values_pre": pixel_values_pre.clone().contiguous(),
            "pixel_values_post": pixel_values_post.clone().contiguous(),
            "labels": item["labels"].clone().contiguous(),
            "id": item["id"]
        }

def create_dataset(root_dir, split_base, image_processor, num_classes=5, augment=False):
    pre_dir = os.path.join(root_dir, f"{split_base}_pre")
    post_dir = os.path.join(root_dir, f"{split_base}")
    
    if os.path.isdir(pre_dir) and os.path.isdir(post_dir):
        print(f"[*] Detected Paired Dataset at {root_dir} (Split base: {split_base})")
        return PairedXBDFloodNetDataset(root_dir, split_base, image_processor, num_classes, augment)
    else:
        print(f"[*] Detected Single-Image Dataset at {root_dir} (Split: {split_base})")
        return SingleImageFloodNetDataset(root_dir, split_base, image_processor, num_classes, augment)

# ---------------------------------------------------------
# 1) Custom Model Wrapper for Hugging Face Trainer
# ---------------------------------------------------------
class XBDDualSiameseModel(nn.Module):
    def __init__(self, backbone="nvidia/mit-b2", num_damage_classes=5):
        super().__init__()
        self.model = DualHeadSiameseSegFormer(backbone=backbone, num_damage_classes=num_damage_classes)
        self.loss_loc = nn.CrossEntropyLoss()
        self.loss_cls = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, pixel_values_pre, pixel_values_post, labels=None, **kwargs):
        logits_loc, logits_cls = self.model(pixel_values_pre, pixel_values_post)
        loss = None
        if labels is not None:
            labels_loc = (labels > 0).long()
            loss_loc = self.loss_loc(logits_loc, labels_loc)
            loss_cls = self.loss_cls(logits_cls, labels)
            loss = loss_loc + loss_cls

        return {"loss": loss, "logits_loc": logits_loc, "logits_cls": logits_cls}

# ---------------------------------------------------------
# 2) Custom Collate & Metrics Function
# ---------------------------------------------------------
def collate_fn(batch):
    # EXTRA CONTIGUOUS SAFEGUARDS ON BATCHING
    return {
        "pixel_values_pre": torch.stack([b["pixel_values_pre"] for b in batch], dim=0).contiguous(),
        "pixel_values_post": torch.stack([b["pixel_values_post"] for b in batch], dim=0).contiguous(),
        "labels": torch.stack([b["labels"] for b in batch], dim=0).contiguous()
    }

def metrics_fn(eval_pred):
    logits_cls = eval_pred.predictions[1] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
    labels = eval_pred.label_ids
    
    metrics = compute_mIoU((logits_cls, labels), num_classes=5, ignore_index=0)
    
    preds = np.argmax(logits_cls, axis=1).flatten()
    targets = labels.flatten()
    
    valid_mask = (targets != 0)
    preds_valid = preds[valid_mask]
    targets_valid = targets[valid_mask]
    
    cm = confusion_matrix(targets_valid, preds_valid, labels=range(5))
    
    print("\n" + "="*50)
    print("VALIDATION CONFUSION MATRIX (Pixels)")
    print("Rows = Ground Truth | Columns = Predictions")
    print(f"{'':>12} | {'Bg (0)':>10} | {'No Dam (1)':>10} | {'Minor (2)':>10} | {'Major (3)':>10} | {'Destr (4)':>10}")
    print("-" * 75)
    class_names = ["Bg (0)", "No Dam (1)", "Minor (2)", "Major (3)", "Destr (4)"]
    
    for i in range(1, 5):  
        print(f"{class_names[i]:>12} | {cm[i][0]:10d} | {cm[i][1]:10d} | {cm[i][2]:10d} | {cm[i][3]:10d} | {cm[i][4]:10d}")
    print("="*50 + "\n")
    
    return metrics

# ---------------------------------------------------------
# 3) Training Loop Setup
# ---------------------------------------------------------
def train(args, custom_args, ddp_kwargs):
    processor = SegformerImageProcessor.from_pretrained(args.model_name)

    print(f"\n--- Loading Training Data ---")
    train_dir = custom_args["train_dir"] or args.data_root
    train_ds = create_dataset(
        root_dir=train_dir, split_base=args.train_split, 
        image_processor=processor, augment=True, num_classes=5
    )

    print(f"\n--- Loading Validation Data ---")
    eval_datasets = {}
    val_dirs = custom_args["val_dirs"] if custom_args["val_dirs"] else [args.data_root]
    
    for v_dir in val_dirs:
        ds_name = os.path.basename(os.path.normpath(v_dir))
        eval_datasets[ds_name] = create_dataset(
            root_dir=v_dir, split_base=args.val_split, 
            image_processor=processor, augment=False, num_classes=5
        )

    model = XBDDualSiameseModel(backbone=args.model_name, num_damage_classes=5)

    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * max(1, torch.cuda.device_count())))
    save_steps = max(steps_per_epoch, 100)

    first_val_name = list(eval_datasets.keys())[0]
    best_metric_name = f"{first_val_name}_mIoU"

    training_args = safe_training_args(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_steps=save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=best_metric_name, 
        greater_is_better=True,
        overwrite_output_dir=args.overwrite_output_dir,
        remove_unused_columns=False, 
        **ddp_kwargs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_datasets, 
        data_collator=collate_fn,
        compute_metrics=metrics_fn,
    )

    print("\n--- Starting Training ---")
    resume_from = choose_resume_checkpoint(args.resume, args.output_dir)
    trainer.train(resume_from_checkpoint=resume_from)
    
    metrics = trainer.evaluate()
    print("Final Multi-Dataset Evaluation Metrics:", metrics)

    torch.save(model.model.state_dict(), os.path.join(args.output_dir, "dual_siamese_best.pth"))
    print("Training complete. Best checkpoint saved at:", trainer.state.best_model_checkpoint)


def extract_custom_args():
    custom_args = {"train_dir": None, "val_dirs": []}
    
    if "--train_dir" in sys.argv:
        idx = sys.argv.index("--train_dir")
        custom_args["train_dir"] = sys.argv[idx + 1]
        sys.argv.pop(idx)
        sys.argv.pop(idx)
        
    if "--val_dirs" in sys.argv:
        idx = sys.argv.index("--val_dirs")
        sys.argv.pop(idx)
        while idx < len(sys.argv) and not sys.argv[idx].startswith("--"):
            custom_args["val_dirs"].append(sys.argv[idx])
            sys.argv.pop(idx)
            
    return custom_args

def main():
    custom_args = extract_custom_args()
    args = parse_args()

    mode, local_rank, world_size = setup_devices_autodetect()
    ddp_kwargs = {}
    if world_size > 1:
        ddp_kwargs.update(dict(ddp_find_unused_parameters=False, ddp_backend="nccl"))

    set_seed(args.seed)
    train(args, custom_args, ddp_kwargs)

if __name__ == "__main__":
    main()

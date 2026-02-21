
# train_attention_unet.py
# Train an Attention U-Net for semantic segmentation using the same dataset/config flow as train_segformer.py
# Works with nh_datasets.loader.build_dataset_from_py() datasets that return:
#   {"pixel_values": FloatTensor[3,H,W], "labels": LongTensor[H,W], "id": str}
#
# Usage (example):
#   python train_attention_unet.py --config_file path/to/your_dataset_config.py --output_dir runs/attnunet
#
# Notes:
# - Uses Hugging Face Trainer for convenience (DDP, checkpoints, logging).
# - Saves/loads like a HF model via PreTrainedModel/PretrainedConfig.
# - Evaluation computes mIoU with your project's utils.compute_mIoU.
#
import os, json, math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    Trainer,
    set_seed,
    SegformerImageProcessor
)

from nh_datasets.loader import build_dataset_from_py
from scripts.utils import (
    setup_devices_autodetect,
    safe_training_args,
    compute_mIoU,
    choose_resume_checkpoint,
    parse_args,
    discover_best_model_dir,
    take_first_n,
    ddp_barrier_safe
)

# ------------------------------------------
# Minimal image processor (reuse SegFormer one if available)
# ------------------------------------------
DefaultImageProcessor = SegformerImageProcessor

class AttnUNetConfig(PretrainedConfig):
    model_type = "attn-unet"

    def __init__(
        self,
        num_labels: int = 10,
        in_channels: int = 3,
        base_channels: int = 64,
        depth: int = 4,
        use_deconv: bool = False,
        ignore_index: int = 255,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.depth = depth
        self.use_deconv = use_deconv
        self.ignore_index = ignore_index

# ------------------------------------------
# Attention U-Net building blocks
# ------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """
    Classic Attention U-Net gate (Oktay et al., 2018).
    Filters encoder skip 'x' with a gating signal 'g' from the decoder.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating from decoder (coarser scale), x: skip from encoder (same HxW after upsample)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)           # [B,1,H,W]
        return x * psi                 # element-wise attention


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, use_deconv=False):
        super().__init__()
        if use_deconv:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            up_out = in_ch // 2
        else:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            up_out = in_ch

        self.att = AttentionGate(F_g=up_out, F_l=skip_ch, F_int=max(out_ch // 2, 1))
        self.conv = DoubleConv(up_out + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Ensure same spatial size
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.att(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ------------------------------------------
# Config & Model wrappers (HF-style)
# ------------------------------------------

class AttnUNetBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_labels: int = 10,
        base_channels: int = 64,
        depth: int = 4,
        use_deconv: bool = False,
    ):
        super().__init__()
        assert depth in (3, 4, 5), "depth should be 3, 4 or 5"
        ch = base_channels

        # Encoder
        self.enc1 = DoubleConv(in_channels, ch)        # H
        self.pool1 = nn.MaxPool2d(2)                   # H/2
        self.enc2 = DoubleConv(ch, ch * 2)             # H/2
        self.pool2 = nn.MaxPool2d(2)                   # H/4
        self.enc3 = DoubleConv(ch * 2, ch * 4)         # H/4
        self.pool3 = nn.MaxPool2d(2)                   # H/8

        if depth >= 4:
            self.enc4 = DoubleConv(ch * 4, ch * 8)     # H/8
            self.pool4 = nn.MaxPool2d(2)               # H/16
        else:
            self.enc4 = None

        if depth == 5:
            self.enc5 = DoubleConv(ch * 8, ch * 16)    # H/16
            self.pool5 = nn.MaxPool2d(2)               # H/32
        else:
            self.enc5 = None

        # Bottleneck
        bott_in = {3: ch * 4, 4: ch * 8, 5: ch * 16}[depth]
        self.bottleneck = DoubleConv(bott_in, bott_in * 2)

        # Decoder
        if depth == 5:
            self.up5 = UpBlock(bott_in * 2, ch * 16, ch * 16, use_deconv)
            up4_in = ch * 16
        else:
            self.up5 = None
            up4_in = bott_in * 2

        if depth >= 4:
            self.up4 = UpBlock(up4_in, ch * 8, ch * 8, use_deconv)
            up3_in = ch * 8
        else:
            self.up4 = None
            up3_in = bott_in * 2

        self.up3 = UpBlock(up3_in, ch * 4, ch * 4, use_deconv)
        self.up2 = UpBlock(ch * 4, ch * 2, ch * 2, use_deconv)
        self.up1 = UpBlock(ch * 2, ch, ch, use_deconv)

        # Head
        self.head = nn.Conv2d(ch, num_labels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        x = self.pool1(e1)

        e2 = self.enc2(x)
        x = self.pool2(e2)

        e3 = self.enc3(x)
        x = self.pool3(e3)

        if self.enc4 is not None:
            e4 = self.enc4(x)
            x = self.pool4(e4)
        else:
            e4 = None

        if self.enc5 is not None:
            e5 = self.enc5(x)
            x = self.pool5(e5)
        else:
            e5 = None

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        if self.up5 is not None:
            x = self.up5(x, e5)
        if self.up4 is not None:
            x = self.up4(x, e4)
        x = self.up3(x, e3)
        x = self.up2(x, e2)
        x = self.up1(x, e1)

        logits = self.head(x)
        return logits


class AttnUNetForSemanticSegmentation(PreTrainedModel):
    config_class = AttnUNetConfig
    def __init__(self, config: AttnUNetConfig):
        super().__init__(config)
        self.model = AttnUNetBackbone(
            in_channels=config.in_channels,
            num_labels=config.num_labels,
            base_channels=config.base_channels,
            depth=config.depth,
            use_deconv=config.use_deconv,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        logits = self.model(pixel_values)  # [B,C,H,W]
        loss = None
        if labels is not None:
            if logits.shape[-2:] != labels.shape[-2:]:
                logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logits, labels.long(), ignore_index=self.config.ignore_index)
        return {"loss": loss, "logits": logits}


# ------------------------------------------
# Evaluate-only (test set)
# ------------------------------------------
@torch.no_grad()
def evaluate_only(args, ddp_kwargs):
    IGNORE = args.ignore_index
    ddp_barrier_safe()

    # Choose model dir to load
    model_dir = args.eval_from or discover_best_model_dir(args.output_dir)
    print(f"[evaluate] loading model from: {model_dir}")

    # Image processor
    try:
        image_processor = DefaultImageProcessor.from_pretrained(model_dir)
        image_processor.do_resize = False
        image_processor.do_normalize = True
    except Exception:
        image_processor = DefaultImageProcessor(do_resize=False, do_normalize=True, reduce_labels=False)


    # Dataset
    eval_ds_full = build_dataset_from_py(
        args.config_file, split=args.test_split, augment=False, image_processor=image_processor
    )
    eval_ds = take_first_n(eval_ds_full, args.eval_limit)

    # Model
    model = AttnUNetForSemanticSegmentation.from_pretrained(
        model_dir,
        num_labels=args.num_classes,
        id2label=eval_ds_full.id2label,
        label2id=eval_ds_full.label2id,
        ignore_mismatched_sizes=True,
    )

    # Collate
    def collate_fn(batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
        labels = torch.stack([b["labels"] for b in batch], dim=0)
        return {"pixel_values": pixel_values, "labels": labels}

    def metrics_fn(eval_pred):
        return compute_mIoU(eval_pred, num_classes=args.num_classes, ignore_index=IGNORE)

    training_args = safe_training_args(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.eval_batch_size,
        fp16=args.fp16,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        **ddp_kwargs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_ds,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=metrics_fn,
    )

    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=2))
    # Dump predictions as PNGs
    if args.save_preds_dir:
        os.makedirs(args.save_preds_dir, exist_ok=True)
        # Run a forward pass to get logits (HF evaluate already did; we’ll re-run to also get ids cleanly)
        from torch.utils.data import DataLoader
        dl = DataLoader(eval_ds, batch_size=args.eval_batch_size, shuffle=False,
                        num_workers=2, collate_fn=collate_fn)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        with torch.no_grad():
            idx_base = 0
            for batch in dl:
                pv = batch["pixel_values"].to(device)
                logits = model(pixel_values=pv).logits  # [B,C,h,w]
                # upsample to label size
                H, W = batch["labels"].shape[-2], batch["labels"].shape[-1]
                if logits.shape[-2:] != (H, W):
                    logits = torch.nn.functional.interpolate(
                        logits, size=(H, W), mode="bilinear", align_corners=False
                    )
                preds = logits.argmax(1).cpu().numpy().astype(np.uint8)

                # Save as grayscale index PNGs
                for i in range(preds.shape[0]):
                    # get the original id if available (Subset wraps items as indices)
                    sample = eval_ds_full[idx_base + i] if isinstance(eval_ds, torch.utils.data.Subset) else eval_ds[idx_base + i]
                    im_id = sample["id"] if isinstance(sample, dict) else f"img_{idx_base+i:06d}"
                    Image.fromarray(preds[i], mode="L").save(os.path.join(args.save_preds_dir, f"{im_id}_pred.png"))
                idx_base += preds.shape[0]

        print(f"[evaluate] Saved predictions to: {args.save_preds_dir}")


# ------------------------------------------
# Train
# ------------------------------------------
def train(args, ddp_kwargs):
    # Image processor
    image_processor = DefaultImageProcessor(do_resize=False, 
                                            do_normalize=True, 
                                            reduce_labels=False)

    # Datasets
    train_ds = build_dataset_from_py(
        args.config_file, 
        split=args.train_split, 
        augment=True, 
        image_processor=image_processor
    )
    val_ds = build_dataset_from_py(
        args.config_file, 
        split=args.val_split, 
        augment=False, 
        image_processor=image_processor
    )

    id2label = train_ds.id2label
    label2id = train_ds.label2id

    cfg = AttnUNetConfig(
        num_labels=args.num_classes,
        in_channels=3,
        base_channels=getattr(args, "base_channels", 64),
        depth=getattr(args, "depth", 4),
        use_deconv=getattr(args, "use_deconv", False),
        ignore_index=args.ignore_index
    )
    # Model
    model = AttnUNetForSemanticSegmentation(cfg)

    # Collate
    def collate_fn(batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
        labels = torch.stack([b["labels"] for b in batch], dim=0)
        return {"pixel_values": pixel_values, "labels": labels}

    # Steps & saving cadence
    steps_per_epoch = max(1, math.ceil(len(train_ds) / (args.batch_size * max(1, torch.cuda.device_count()))))
    save_steps = max(steps_per_epoch, 100)

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
        metric_for_best_model="mIoU",
        greater_is_better=True,
        overwrite_output_dir=args.overwrite_output_dir,
        remove_unused_columns=False,  # critical: we manually collate
        **ddp_kwargs,
    )

    def metrics_fn(eval_pred):
        return compute_mIoU(eval_pred, num_classes=args.num_classes, ignore_index=args.ignore_index)

    resume_from = choose_resume_checkpoint(args.resume, args.output_dir)
    print(f"[resume] Using checkpoint: {resume_from}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=metrics_fn,
    )

    trainer.train(resume_from_checkpoint=resume_from)
    metrics = trainer.evaluate()
    print(metrics)

    # Save
    trainer.save_model(args.output_dir)
    with open(os.path.join(args.output_dir, "id2label.json"), "w") as f:
        json.dump(id2label, f, indent=2)
    with open(os.path.join(args.output_dir, "label2id.json"), "w") as f:
        json.dump(label2id, f, indent=2)

    print("Training complete. Best checkpoint in:", getattr(trainer.state, "best_model_checkpoint", None))


# ------------------------------------------
# Main
# ------------------------------------------
def main():
    args = parse_args()
    # Optional extra args (with defaults) for U-Net
    if not hasattr(args, "base_channels"):
        args.base_channels = 64
    if not hasattr(args, "depth"):
        args.depth = 4
    if not hasattr(args, "use_deconv"):
        args.use_deconv = False

    mode, local_rank, world_size = setup_devices_autodetect()
    ddp_kwargs = {}
    if world_size > 1:
        ddp_kwargs.update(dict(
            ddp_find_unused_parameters=False,
            ddp_backend="nccl",
        ))
    set_seed(args.seed)

    if args.evaluate:
        evaluate_only(args, ddp_kwargs)
        return
    train(args, ddp_kwargs)


if __name__ == "__main__":
    main()

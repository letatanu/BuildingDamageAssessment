"""
Train SegFormer on one dataset (xBD or CRASAR) and evaluate/test on the other.

Usage examples:

  # Train on xBD, evaluate on CRASAR at every epoch:
  python -m scripts.train_segformer_xbd_crasar \
      --train_config  nh_datasets/configs/segformer_xBD.py \
      --test_configs  nh_datasets/configs/segformer_crarsar.py \
      --output_dir    runs/segformer_xbd_to_crasar

  # Reverse: train on CRASAR, evaluate on xBD:
  python -m scripts.train_segformer_xbd_crasar \
      --train_config  nh_datasets/configs/segformer_crarsar.py \
      --test_configs  nh_datasets/configs/segformer_xBD.py \
      --output_dir    runs/segformer_crasar_to_xbd

  # Train on xBD, evaluate on BOTH CRASAR and xBD-test:
 python -m scripts.train_segformer_xbd_crasar \
      --train_config  nh_datasets/configs/segformer_xBD.py \
      --test_configs  nh_datasets/configs/segformer_crarsar.py nh_datasets/configs/segformer_xBD.py \
      --output_dir    runs/segformer_xbd_to_both

  # Evaluation only (skip training, load best checkpoint):
  python scripts/train_segformer_xbd_crasar.py \
      --train_config  nh_datasets/configs/segformer_xBD.py \
      --test_configs  nh_datasets/configs/segformer_crarsar.py \
      --output_dir    runs/segformer_xbd_to_crasar \
      --evaluate True
"""

import os
import sys
import json
import math
import runpy
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    Trainer,
    set_seed,
)

from nh_datasets.loader import build_dataset_from_py
from scripts.utils import (
    setup_devices_autodetect,
    safe_training_args,
    compute_mIoU,
    choose_resume_checkpoint,
    discover_best_model_dir,
    take_first_n,
    ddp_barrier_safe,
)


# ---------------------------------------------------------------------------
# CLI: parse the extra --train_config / --test_configs arguments BEFORE
#      the standard parse_args() that reads from the config file.
# ---------------------------------------------------------------------------

def parse_cross_args():
    """
    Lightweight pre-parser that extracts the cross-dataset arguments.
    All values can also be overridden per-flag on the CLI.
    """
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--train_config",  required=True,
                   help="Path to the *training* dataset config (.py).")
    p.add_argument("--test_configs",  nargs="+", required=True,
                   help="Paths to one or more *test* dataset configs (.py).")
    p.add_argument("--output_dir",    type=str, default=None,
                   help="Override output directory (also overrides config).")
    # Standard overrides (forwarded to the config reader)
    p.add_argument("--model_name",    type=str, default=None)
    p.add_argument("--batch_size",    type=int, default=None)
    p.add_argument("--eval_batch_size", type=int, default=None)
    p.add_argument("--epochs",        type=int, default=None)
    p.add_argument("--lr",            type=float, default=None)
    p.add_argument("--weight_decay",  type=float, default=None)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--fp16",          type=lambda x: x.lower() == "true",
                   default=None)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--resume",        type=str, default=None)
    p.add_argument("--overwrite_output_dir",
                   type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--evaluate",
                   type=lambda x: x.lower() == "true", default=False,
                   help="If True, skip training and only evaluate.")
    p.add_argument("--eval_from",     type=str, default=None,
                   help="Explicit checkpoint/model dir for eval-only mode.")
    p.add_argument("--eval_limit",    type=int, default=None,
                   help="Cap the number of test samples per dataset.")
    p.add_argument("--save_preds_dir", type=str, default=None,
                   help="If set, dump per-image prediction PNGs here.")
    p.add_argument("--train_split",   type=str, default=None)
    p.add_argument("--val_split",     type=str, default=None)
    p.add_argument("--test_split",    type=str, default=None)
    p.add_argument("--ignore_index",  type=int, default=None)
    p.add_argument("--num_classes",   type=int, default=None)

    return p.parse_args()


def _load_cfg(config_path: str) -> dict:
    """Run a dataset config .py and return its namespace as a dict."""
    return runpy.run_path(config_path)


def _resolve(user_val, cfg_val, default):
    """Return the first non-None value in priority: CLI > config > default."""
    if user_val is not None:
        return user_val
    if cfg_val is not None:
        return cfg_val
    return default


# ---------------------------------------------------------------------------
# Collate helper
# ---------------------------------------------------------------------------

def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
    labels       = torch.stack([b["labels"]       for b in batch], dim=0)
    return {"pixel_values": pixel_values, "labels": labels}


# ---------------------------------------------------------------------------
# Build a *named* eval-dataset dict from a list of config paths
# ---------------------------------------------------------------------------

def build_eval_datasets(test_config_paths, test_split, image_processor,
                        eval_limit=None):
    """
    Returns an OrderedDict  { dataset_name: Dataset }  so the Trainer can
    evaluate on multiple held-out sets simultaneously (same as
    train_dual_segformer.py's val_dirs pattern).
    """
    eval_datasets = {}
    for cfg_path in test_config_paths:
        cfg     = _load_cfg(cfg_path)
        ds_name = cfg.get("DATASET_NAME", Path(cfg_path).stem)
        split   = cfg.get("test_split", test_split)
        ds_full = build_dataset_from_py(
            cfg_path, split=split, augment=False,
            image_processor=image_processor)
        ds = take_first_n(ds_full, eval_limit)
        eval_datasets[ds_name] = ds
        print(f"  [{ds_name}] test split='{split}', "
              f"samples={len(ds_full)}"
              + (f" (capped at {eval_limit})" if eval_limit else ""))
    return eval_datasets


# ---------------------------------------------------------------------------
# 1) Evaluation only
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_only(cross, train_cfg):
    ddp_barrier_safe()

    num_classes  = _resolve(cross.num_classes,  train_cfg.get("num_classes"),  5)
    ignore_index = _resolve(cross.ignore_index, train_cfg.get("ignore_index"), 255)
    output_dir   = _resolve(cross.output_dir,   train_cfg.get("output_dir"),
                            "runs/segformer_cross")
    test_split   = _resolve(cross.test_split,   train_cfg.get("test_split"),   "test")
    eval_limit   = cross.eval_limit

    model_dir = cross.eval_from or discover_best_model_dir(output_dir)
    print(f"[evaluate] Loading model from: {model_dir}")

    try:
        image_processor = SegformerImageProcessor.from_pretrained(model_dir)
        image_processor.do_resize    = False
        image_processor.do_normalize = True
    except Exception:
        image_processor = SegformerImageProcessor(
            do_resize=False, do_normalize=True, reduce_labels=False)

    eval_datasets = build_eval_datasets(
        cross.test_configs, test_split, image_processor, eval_limit)

    # id2label / label2id from the first test config
    first_cfg = _load_cfg(cross.test_configs[0])
    ds_tmp    = build_dataset_from_py(
        cross.test_configs[0],
        split=first_cfg.get("test_split", test_split),
        augment=False,
        image_processor=image_processor,
    )
    id2label = ds_tmp.id2label
    label2id = ds_tmp.label2id

    model = SegformerForSemanticSegmentation.from_pretrained(
        model_dir,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    def metrics_fn(eval_pred):
        return compute_mIoU(eval_pred, num_classes=num_classes,
                            ignore_index=ignore_index)

    _, _, world_size = setup_devices_autodetect()
    ddp_kwargs = {}
    if world_size > 1:
        ddp_kwargs = dict(ddp_find_unused_parameters=False, ddp_backend="nccl")

    training_args = safe_training_args(
        output_dir=output_dir,
        per_device_eval_batch_size=cross.eval_batch_size or 2,
        fp16=cross.fp16 if cross.fp16 is not None else False,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        **ddp_kwargs,
    )

    all_metrics = {}
    for ds_name, eval_ds in eval_datasets.items():
        print(f"\n[evaluate] Running on: {ds_name}")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=None,
            eval_dataset=eval_ds,
            processing_class=image_processor,
            data_collator=collate_fn,
            compute_metrics=metrics_fn,
        )
        m = trainer.evaluate()
        all_metrics[ds_name] = m
        print(json.dumps({ds_name: m}, indent=2))

    # ---- optional: dump prediction PNGs (mirrors train_segformer.py) ----
    if cross.save_preds_dir:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval().to(device)
        for ds_name, eval_ds in eval_datasets.items():
            out_dir = os.path.join(cross.save_preds_dir, ds_name)
            os.makedirs(out_dir, exist_ok=True)
            dl = DataLoader(eval_ds, batch_size=cross.eval_batch_size or 2,
                            shuffle=False, num_workers=2,
                            collate_fn=collate_fn)
            idx_base = 0
            with torch.no_grad():
                for batch in dl:
                    pv     = batch["pixel_values"].to(device)
                    logits = model(pixel_values=pv).logits
                    H, W   = (batch["labels"].shape[-2],
                              batch["labels"].shape[-1])
                    if logits.shape[-2:] != (H, W):
                        logits = torch.nn.functional.interpolate(
                            logits, size=(H, W), mode="bilinear",
                            align_corners=False)
                    preds = logits.argmax(1).cpu().numpy().astype(np.uint8)
                    for i in range(preds.shape[0]):
                        im_id = f"img_{idx_base+i:06d}"
                        Image.fromarray(preds[i], mode="L").save(
                            os.path.join(out_dir, f"{im_id}_pred.png"))
                    idx_base += preds.shape[0]
            print(f"[evaluate] Saved predictions '{ds_name}' → {out_dir}")

    print("\n[evaluate] All metrics:")
    print(json.dumps(all_metrics, indent=2))


# ---------------------------------------------------------------------------
# 2) Training
# ---------------------------------------------------------------------------

def train(cross, train_cfg, ddp_kwargs):
    # ---- Resolve hyper-params (CLI > config > sensible default) ----
    model_name   = _resolve(cross.model_name,   train_cfg.get("model_name"),
                            "nvidia/segformer-b2-finetuned-ade-512-512")
    num_classes  = _resolve(cross.num_classes,  train_cfg.get("num_classes"),  5)
    ignore_index = _resolve(cross.ignore_index, train_cfg.get("ignore_index"), 255)
    batch_size   = _resolve(cross.batch_size,   train_cfg.get("batch_size"),   2)
    eval_bs      = _resolve(cross.eval_batch_size, None,                       2)
    lr           = _resolve(cross.lr,           train_cfg.get("lr"),           6e-5)
    wd           = _resolve(cross.weight_decay, train_cfg.get("weight_decay"), 0.01)
    epochs       = _resolve(cross.epochs,       train_cfg.get("num_epochs"),   100)
    fp16         = _resolve(cross.fp16,         train_cfg.get("fp16"),         False)
    save_limit   = cross.save_total_limit
    output_dir   = _resolve(cross.output_dir,   train_cfg.get("output_dir"),
                            "runs/segformer_cross")
    train_split  = _resolve(cross.train_split,  train_cfg.get("train_split"),  "train")
    val_split    = _resolve(cross.val_split,     train_cfg.get("val_split"),   "val")
    test_split   = _resolve(cross.test_split,    train_cfg.get("test_split"),  "test")
    eval_limit   = cross.eval_limit

    image_processor = SegformerImageProcessor(
        do_resize=False, do_normalize=True, reduce_labels=False)

    # ---- Training dataset (from train_config) ----
    print(f"\n--- Loading TRAIN data from: {cross.train_config}"
          f"  (split='{train_split}') ---")
    train_ds = build_dataset_from_py(
        cross.train_config, split=train_split, augment=True,
        image_processor=image_processor)
    print(f"  Train samples: {len(train_ds)}")

    # ---- Home-domain val split (same config, val split) ----
    print(f"\n--- Loading VAL data from: {cross.train_config}"
          f"  (split='{val_split}') ---")
    eval_datasets = {}
    try:
        val_ds_home = build_dataset_from_py(
            cross.train_config, split=val_split, augment=False,
            image_processor=image_processor)
        home_val_name = (train_cfg.get("DATASET_NAME",
                                       Path(cross.train_config).stem) + "_val")
        eval_datasets[home_val_name] = take_first_n(val_ds_home, eval_limit)
        print(f"  Home-val samples: {len(val_ds_home)}")
    except Exception as exc:
        print(f"  [warn] Could not load home val split ({exc}); skipping.")

    # ---- Cross-domain test datasets ----
    print(f"\n--- Loading TEST data (cross-domain eval) ---")
    cross_eval = build_eval_datasets(
        cross.test_configs, test_split, image_processor, eval_limit)
    eval_datasets.update(cross_eval)
    print(f"\n  eval_datasets keys: {list(eval_datasets.keys())}")

    # ---- Model ----
    id2label = train_ds.id2label
    label2id = train_ds.label2id

    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # ---- Training args ----
    steps_per_epoch = math.ceil(
        len(train_ds) / (batch_size * max(1, torch.cuda.device_count())))
    save_steps = max(steps_per_epoch, 100)

    # Best-model metric is driven by the FIRST eval dataset (cross-domain)
    first_test_name = list(eval_datasets.keys())[0]
    best_metric     = f"{first_test_name}_mIoU"

    training_args = safe_training_args(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_bs,
        learning_rate=lr,
        weight_decay=wd,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_steps=save_steps,
        save_total_limit=save_limit,
        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        greater_is_better=True,
        overwrite_output_dir=cross.overwrite_output_dir,
        remove_unused_columns=False,
        fp16=fp16,
        **ddp_kwargs,
    )

    def metrics_fn(eval_pred):
        return compute_mIoU(eval_pred, num_classes=num_classes,
                            ignore_index=ignore_index)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_datasets,     # dict → multi-dataset eval
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=metrics_fn,
    )

    resume_from = choose_resume_checkpoint(cross.resume, output_dir)
    print(f"[resume] Using checkpoint: {resume_from}")
    trainer.train(resume_from_checkpoint=resume_from)

    metrics = trainer.evaluate()
    print("\nFinal evaluation metrics:")
    print(json.dumps(metrics, indent=2))

    # ---- Save ----
    trainer.save_model(output_dir)
    image_processor.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "id2label.json"), "w") as f:
        json.dump(id2label, f, indent=2)
    with open(os.path.join(output_dir, "label2id.json"), "w") as f:
        json.dump(label2id, f, indent=2)

    print("Training complete. Best checkpoint in:",
          trainer.state.best_model_checkpoint)


# ---------------------------------------------------------------------------
# 3) Main
# ---------------------------------------------------------------------------

def main():
    cross     = parse_cross_args()
    train_cfg = _load_cfg(cross.train_config)

    set_seed(cross.seed)
    mode, local_rank, world_size = setup_devices_autodetect()

    ddp_kwargs = {}
    if world_size > 1:
        ddp_kwargs = dict(ddp_find_unused_parameters=False, ddp_backend="nccl")

    if cross.evaluate:
        evaluate_only(cross, train_cfg)
    else:
        train(cross, train_cfg, ddp_kwargs)


if __name__ == "__main__":
    main()

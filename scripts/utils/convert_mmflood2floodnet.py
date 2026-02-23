import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image


def normalize_band(band: np.ndarray, low=2, high=98) -> np.ndarray:
    """
    Percentile-based normalization to [0, 255] uint8.
    """
    vmin = np.percentile(band, low)
    vmax = np.percentile(band, high)
    band = np.clip((band - vmin) / (vmax - vmin + 1e-6), 0, 1)
    return (band * 255).astype(np.uint8)


def load_subset_mapping(activations_json: Path):
    """
    Returns a dict: EMSR code (e.g. 'EMSR548') -> 'train'/'val'/'test'
    based on activations.json ('subset' field).
    """
    data = json.loads(activations_json.read_text())
    mapping = {}
    for emsr_code, info in data.items():
        subset = info.get("subset", "train")  # default to train if missing
        # Normalize names a bit
        subset = subset.lower()
        if subset == "validation":
            subset = "val"
        mapping[emsr_code] = subset
    return mapping


def convert_mmflood_to_floodnet(
    mmflood_root: str,
    out_root: str,
    use_split_from_json: bool = True,
):
    mmflood_root = Path(mmflood_root)
    out_root = Path(out_root)
    acts_dir = mmflood_root / "activations"
    act_json = mmflood_root / "activations.json"

    if not acts_dir.is_dir():
        raise FileNotFoundError(f"{acts_dir} not found")
    if not act_json.is_file():
        raise FileNotFoundError(f"{act_json} not found")

    subset_map = load_subset_mapping(act_json)
    print(f"Loaded subset mapping for {len(subset_map)} EMSR activations.")

    # Prepare output dirs
    for split in ["train", "val", "test"]:
        (out_root / split / f"{split}-org-img").mkdir(parents=True, exist_ok=True)
        (out_root / split / f"{split}-label-img").mkdir(parents=True, exist_ok=True)

    # Walk through each EMSR region
    emsr_regions = sorted([d for d in acts_dir.iterdir() if d.is_dir()])
    print(f"Found {len(emsr_regions)} EMSR regions under {acts_dir}")

    n_tiles = 0
    for region_dir in emsr_regions:
        region_name = region_dir.name                 # e.g. 'EMSR548-0'
        emsr_code = region_name.split("-")[0]         # 'EMSR548'

        if use_split_from_json:
            split = subset_map.get(emsr_code, "train")
        else:
            # if you want your own random split, you could ignore activations.json here
            split = "train"

        if split not in ("train", "val", "test"):
            print(f"  [WARN] Unknown subset '{split}' for {emsr_code}, forcing to 'train'")
            split = "train"

        img_out_dir = out_root / split / f"{split}-org-img"
        lab_out_dir = out_root / split / f"{split}-label-img"

        s1_dir = region_dir / "s1_raw"
        mask_dir = region_dir / "mask"

        if not s1_dir.is_dir() or not mask_dir.is_dir():
            print(f"  [SKIP] Missing s1_raw or mask in {region_dir}")
            continue

        s1_files = sorted(s1_dir.glob("*.tif"))
        if not s1_files:
            print(f"  [SKIP] No s1_raw tiles in {s1_dir}")
            continue

        print(f"[{region_name}] -> split={split}, {len(s1_files)} tiles")

        for tif_path in s1_files:
            stem = tif_path.stem    # e.g. 'EMSR548-0-0'
            mask_path = mask_dir / f"{stem}.tif"
            if not mask_path.is_file():
                print(f"    [SKIP] Missing mask for {stem}")
                continue

            # --- Load SAR VV/VH and build pseudo-RGB ---
            with rasterio.open(tif_path) as src:
                arr = src.read()   # [C, H, W], typically C=2 (VV, VH)
            if arr.ndim != 3 or arr.shape[0] < 2:
                print(f"    [SKIP] Unexpected SAR shape {arr.shape} in {tif_path}")
                continue

            vv = normalize_band(arr[0])
            vh = normalize_band(arr[1])

            # Simple composite: [VV, VH, VV]
            rgb = np.stack([vv, vh, vv], axis=-1)  # [H, W, 3]
            rgb_img = Image.fromarray(rgb, mode="RGB")

            # --- Load mask, keep original uint8 labels ---
            with rasterio.open(mask_path) as msrc:
                mask = msrc.read(1).astype(np.uint8)

            # Optional: ensure binary [0=non-flood,1=flood]; comment out if your masks are already 0/1.
            # mask = (mask > 0).astype(np.uint8)

            lab_img = Image.fromarray(mask, mode="L")

            # --- Save in FloodNet-like structure ---
            out_img_path = img_out_dir / f"{stem}.jpg"
            out_lab_path = lab_out_dir / f"{stem}_lab.png"

            rgb_img.save(out_img_path, quality=95)
            lab_img.save(out_lab_path)

            n_tiles += 1

    print(f"Done. Wrote {n_tiles} tiles into {out_root}")


if __name__ == "__main__":
    """
    Example usage:

    python mmflood_to_floodnet_format.py
    """
    # EDIT THESE PATHS to match your setup:
    mmflood_root = "/data/nhl224/code/semantic_2D/data/mmflood"  # folder containing activations/ and activations.json
    out_root = "/data/nhl224/code/semantic_2D/data/converted_mmflood"

    convert_mmflood_to_floodnet(mmflood_root, out_root)

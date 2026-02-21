#!/bin/bash
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate vltseg 
cd /working/VLTSeg

python /working/VLTSeg/tools/train.py \
  --config /working/datasets/floodnet/floonet.yaml \
  --pretrained /working/VLTSeg/pretrained/vltseg_base.pth \
  --output_dir runs/floodnet_vltseg_base \
  --lr 2e-5 \
  --epochs 50 \
  --batch_size 8 \
  --precision bf16

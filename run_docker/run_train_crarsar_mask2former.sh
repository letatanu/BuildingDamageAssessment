#!/usr/bin/env bash
set -e

DEVICES="6,7" 
NPROC=$(( $(tr -cd ',' <<<"$DEVICES" | wc -c) + 1 ))
OMP_NUM_THREADS=2

## --------------------------------------------------------- ##
DOCKER_IMAGE="semseg_2d:latest"

docker run --rm -it\
  -v /dev/shm:/dev/shm \
  --gpus "\"device=${DEVICES}\"" \
  -w /working \
  -v /data/nhl224/code/semantic_2D/FloodSemSeg/:/working \
  -v /data/nhl224/code/semantic_2D/data:/data \
  -e OMP_NUM_THREADS=${OMP_NUM_THREADS} \
  "${DOCKER_IMAGE}" \
  bash -lc "
        set -euo pipefail
        export OMP_NUM_THREADS=16
        # Activate conda (adjust if your image uses a different prefix)
        if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
        . /opt/conda/etc/profile.d/conda.sh
        elif [ -f \$HOME/miniconda3/etc/profile.d/conda.sh ]; then
        . \$HOME/miniconda3/etc/profile.d/conda.sh
        else
        echo 'conda.sh not found in image' >&2; exit 1
        fi
        conda activate semseg
        torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC}  \
        -m train_mask2former.py   --config_file /working/nh_datasets/configs/mask2former_crarsar.py \
        "
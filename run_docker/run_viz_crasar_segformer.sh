#!/usr/bin/env bash
set -e


## --------------------------------------------------------- ##
DOCKER_IMAGE="letatanu/semseg_2d:latest"

docker run --rm -it -v /dev/shm:/dev/shm \
  --gpus "all" \
  -w /working \
  -v /data/nhl224/code/semantic_2D/FloodSemSeg/:/working \
  -v /data/nhl224/code/semantic_2D/data:/data \
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
        python  viz_segformer.py \
        --gt_folder /data/CRASAR-tiles/test/test-label-img/ \
        --folder /data/CRASAR-tiles/test/test-org-img/ \
        --model /working/runs/segformer_crarsar/checkpoint-3246   \
        --outdir runs/segformer_crarsar/viz/ --no_show \
        --ext .png
        "
#!/usr/bin/env bash
set -e


## --------------------------------------------------------- ##
DOCKER_IMAGE="letatanu/semseg_2d:latest"

docker run --rm -it -v /dev/shm:/dev/shm \
  --gpus "all" \
  -w /working \
  -v /data/nhl224/code/semantic_2D/FloodSemSeg/:/working \
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
        python  \
        viz_mask2former.py --model /working/runs/mask2former_mmflood/checkpoint-312 \
        --gt_folder data/converted_mmflood/test/test-label-img/ \
        --image data/converted_mmflood/test/test-org-img/EMSR520-0-7.jpg \
        --outdir runs/mask2former_mmflood/viz/
        "
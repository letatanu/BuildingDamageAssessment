#!/usr/bin/env bash
set -e

DEVICES="2,3,4,5" 
NPROC=$(( $(tr -cd ',' <<<"$DEVICES" | wc -c) + 1 ))


## --------------------------------------------------------- ##
DOCKER_IMAGE="semseg_2d:latest"
TRAIN_DATA="data/CRASAR-FloodNet/train"
TEST_DATA="data/xBD_tiles/test"
OUTPUT_DIR="runs/test"
BATCH_SIZE=8
EPOCHS=50
docker run --rm -ti\
  -v /dev/shm:/dev/shm \
  --gpus "\"device=${DEVICES}\"" \
  -w /working \
  -v /data/nhl224/code/semantic_2D/FloodSemSeg/:/working \
  -v /data/nhl224/code/semantic_2D/data:/data \
  "${DOCKER_IMAGE}" \
  # bash -lc "
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
        python scripts/train_segformer_seperate_ds.py     \
        --train_data ${TRAIN_DATA}  --test_data ${TEST_DATA}    \  
        --output_dir ${OUTPUT_DIR}     --batch_size ${BATCH_SIZE}     \
        --epochs ${EPOCHS}
        "
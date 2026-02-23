#!/usr/bin/env bash
set -e

DEVICES="3,4" # change this to specify which GPUs to use (e.g., "0,1,2,3" for 4 GPUs)
NPROC=$(( $(tr -cd ',' <<<"$DEVICES" | wc -c) + 1 ))


## --------------------------------------------------------- ##
DOCKER_IMAGE="letatanu/semseg_2d:latest"

docker run --rm -ti\
  -v /dev/shm:/dev/shm \
  --gpus "\"device=${DEVICES}\"" \
  -w /working \
  -v /data/nhl224/code/semantic_2D/FloodSemSeg/:/working \
  "${DOCKER_IMAGE}"
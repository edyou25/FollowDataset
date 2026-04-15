#!/usr/bin/env bash
set -euo pipefail

# Run this script on the Docker host, not inside the container.
#
# Example:
#   CONTAINER_NAME=Javis \
#   NEW_IMAGE=followdataset:gpu-gl \
#   NEW_CONTAINER=javis-gpu \
#   WORKSPACE_DIR=/home/yyf/IROS2026 \
#   bash tools/restart_with_nvidia_graphics.sh

CONTAINER_NAME="${CONTAINER_NAME:-Javis}"
NEW_IMAGE="${NEW_IMAGE:-followdataset:gpu-gl}"
NEW_CONTAINER="${NEW_CONTAINER:-${CONTAINER_NAME}-gpu}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/home/yyf/IROS2026}"
DISPLAY_VALUE="${DISPLAY_VALUE:-${DISPLAY:-:1}}"

echo "[1/4] Allowing local Docker containers to access X11"
xhost +local:docker

echo "[2/4] Committing current container ${CONTAINER_NAME} -> ${NEW_IMAGE}"
docker commit "${CONTAINER_NAME}" "${NEW_IMAGE}"

echo "[3/4] Removing old replacement container if it exists"
docker rm -f "${NEW_CONTAINER}" >/dev/null 2>&1 || true

echo "[4/4] Starting ${NEW_CONTAINER} with NVIDIA graphics enabled"
docker run -it --rm \
  --name "${NEW_CONTAINER}" \
  --gpus all \
  --env DISPLAY="${DISPLAY_VALUE}" \
  --env QT_X11_NO_MITSHM=1 \
  --env NVIDIA_VISIBLE_DEVICES=all \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env LIBGL_ALWAYS_INDIRECT=0 \
  --env __GLX_VENDOR_LIBRARY_NAME=nvidia \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --volume "${WORKSPACE_DIR}:${WORKSPACE_DIR}" \
  --network host \
  --ipc host \
  --privileged \
  --workdir "${WORKSPACE_DIR}/FollowDataset" \
  "${NEW_IMAGE}" \
  bash


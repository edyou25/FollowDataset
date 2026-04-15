#!/usr/bin/env bash
set -euo pipefail

# Run this script on the Docker host, not inside the container.
#
# Default flow:
# 1. Commit the current container to ORIGINAL_IMAGE_REPO:NEW_TAG
# 2. Remove the old container
# 3. Start a new container with the same name, but with NVIDIA graphics enabled
#
# Example:
#   CONTAINER_NAME=ros \
#   ORIGINAL_IMAGE_REPO=nvidia/cuda \
#   ORIGINAL_IMAGE_TAG=12.8.0-devel-ubuntu20.04 \
#   NEW_TAG=12.8.0-devel-ubuntu20.04-gfx \
#   WORKSPACE_DIR=/home/yyf/IROS2026 \
#   bash tools/restart_with_nvidia_graphics.sh

CONTAINER_NAME="${CONTAINER_NAME:-ros}"
ORIGINAL_IMAGE_REPO="${ORIGINAL_IMAGE_REPO:-nvidia/cuda}"
ORIGINAL_IMAGE_TAG="${ORIGINAL_IMAGE_TAG:-12.8.0-devel-ubuntu20.04}"
NEW_TAG="${NEW_TAG:-12.8.0-devel-ubuntu20.04-gfx}"
TARGET_IMAGE="${ORIGINAL_IMAGE_REPO}:${NEW_TAG}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/home/yyf/IROS2026}"
DISPLAY_VALUE="${DISPLAY_VALUE:-${DISPLAY:-:1}}"

echo "[0/5] Checking container ${CONTAINER_NAME}"
docker inspect "${CONTAINER_NAME}" >/dev/null

echo "[1/5] Allowing local Docker containers to access X11"
xhost +local:docker

echo "[2/5] Committing ${CONTAINER_NAME} -> ${TARGET_IMAGE}"
docker commit "${CONTAINER_NAME}" "${TARGET_IMAGE}"

echo "[3/5] Removing old container ${CONTAINER_NAME}"
docker rm -f "${CONTAINER_NAME}"

echo "[4/5] Starting persistent container ${CONTAINER_NAME} with NVIDIA graphics enabled"
docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --env DISPLAY="${DISPLAY_VALUE}" \
  --env QT_X11_NO_MITSHM=1 \
  --env NVIDIA_VISIBLE_DEVICES=all \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env LIBGL_ALWAYS_INDIRECT=0 \
  --env __GLX_VENDOR_LIBRARY_NAME=nvidia \
  --env XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-root}" \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --volume "${WORKSPACE_DIR}:${WORKSPACE_DIR}" \
  --network host \
  --ipc host \
  --privileged \
  --workdir "${WORKSPACE_DIR}/FollowDataset" \
  "${TARGET_IMAGE}" \
  tail -f /dev/null

echo "[5/5] New container started from ${TARGET_IMAGE}"
echo
echo "Enter it any time with:"
echo "  docker exec -it ${CONTAINER_NAME} bash"

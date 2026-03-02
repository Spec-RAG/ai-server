#!/usr/bin/env bash
set -euo pipefail

APP_IMAGE="ai-server:latest"
ARCHIVE="ai-server-images.tar"

docker build --platform linux/amd64 -t "${APP_IMAGE}" -f Dockerfile .
docker pull --platform linux/amd64 redis:7-alpine
docker save -o "${ARCHIVE}" "${APP_IMAGE}" redis:7-alpine

echo "[ok] created ${ARCHIVE}"
ls -lh "${ARCHIVE}"

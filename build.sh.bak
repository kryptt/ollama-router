#!/bin/bash
set -euo pipefail

REGISTRY="${REGISTRY:-ghcr.io}"
OWNER=kryptt
APP=ollama-router
VERSION=$(grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)"/\1/')

# Push target. Namespaced under $OWNER to mirror the GHCR path
# (ghcr.io/$OWNER/$APP).
IMG=$REGISTRY/$OWNER/$APP:$VERSION

if docker manifest inspect "$IMG" &>/dev/null; then
  echo "ERROR: $IMG already exists in registry."
  echo "Bump version in Cargo.toml before building."
  exit 1
fi

docker buildx build . -t "$IMG"
docker push "$IMG"

echo "Pushed $IMG"
echo "Update fleet manifest: image: $IMG"

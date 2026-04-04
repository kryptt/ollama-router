#!/bin/bash
set -euo pipefail

REGISTRY=registry.hr-home.xyz
APP=ollama-router
VERSION=$(grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)"/\1/')

IMG=$REGISTRY/$APP:$VERSION

if docker manifest inspect "$IMG" &>/dev/null; then
  echo "ERROR: $IMG already exists in registry."
  echo "Bump version in Cargo.toml before building."
  exit 1
fi

docker buildx build . -t "$IMG"
docker push "$IMG"

echo "Pushed $IMG"
echo "Update fleet manifest: image: $IMG"

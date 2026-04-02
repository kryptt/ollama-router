#!/bin/bash
set -euo pipefail

# Run unit tests inside the Docker build environment.
# Uses the "test" stage defined in the Dockerfile.
docker buildx build --target test .

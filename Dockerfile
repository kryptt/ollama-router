# syntax=docker/dockerfile:1
FROM rust:1-alpine AS builder

# TARGETPLATFORM is set automatically by BuildKit (e.g. linux/amd64,
# linux/arm64). We declare it as an ARG so the cache-mount `id`s below
# can interpolate it — keeping per-arch cargo target directories
# separate. Without per-arch scoping a parallel amd64+arm64 release
# build (the GH Actions release.yml uses both) would either trample
# each other's compilation artifacts or serialise via `sharing=locked`
# and lose the parallelism.
ARG TARGETPLATFORM

RUN apk add --no-cache musl-dev

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY tests/ tests/

# Resolve the native musl target triple so the same Dockerfile builds on
# amd64 and arm64 hosts (e.g. under QEMU emulation in GH Actions buildx).
# rustc prints lines like "host: x86_64-unknown-linux-musl"; grab that.
RUN rustc -vV | awk '/^host:/ {print $2}' > /tmp/target-triple && \
    cat /tmp/target-triple

# Three persistent BuildKit cache mounts each stage:
#
#   cargo-registry  — index + downloaded crate tarballs (platform-agnostic;
#                     same content for amd64 and arm64, so id is unscoped
#                     and sharing=shared lets both arches read concurrently)
#   cargo-git       — git-dependency checkouts (same shape)
#   ollama-router-target-${TARGETPLATFORM}
#                   — compiled artifacts. Per-arch because the .rlib/.so
#                     bits in target/<triple>/release differ per arch.
#                     sharing=locked because cargo holds a build-lock on
#                     the target dir; concurrent writers on the same id
#                     would corrupt each other.
#
# With these mounts in place + `cache-to: type=gha,mode=max` in the
# release workflow, an incremental commit only recompiles the changed
# crate (ours) rather than re-pulling and recompiling all ~100 deps.
# Saves ~25s on a local amd64 build and ~4-8x that on the QEMU-emulated
# arm64 build in CI.

FROM builder AS test
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry,sharing=shared \
    --mount=type=cache,target=/usr/local/cargo/git,id=cargo-git,sharing=shared \
    --mount=type=cache,target=/build/target,id=ollama-router-target-${TARGETPLATFORM},sharing=locked \
    TARGET="$(cat /tmp/target-triple)" && \
    cargo test --target "$TARGET" -- --test-threads=1

FROM builder AS release
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry,sharing=shared \
    --mount=type=cache,target=/usr/local/cargo/git,id=cargo-git,sharing=shared \
    --mount=type=cache,target=/build/target,id=ollama-router-target-${TARGETPLATFORM},sharing=locked \
    TARGET="$(cat /tmp/target-triple)" && \
    cargo build --release --target "$TARGET" && \
    cp "target/$TARGET/release/ollama-router" /ollama-router
RUN mkdir -p /empty-tmp

FROM scratch

COPY --from=release /empty-tmp /tmp
COPY --from=release /ollama-router /ollama-router

EXPOSE 11434 9090

ENTRYPOINT ["/ollama-router"]

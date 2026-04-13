# syntax=docker/dockerfile:1
FROM rust:1-alpine AS builder

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

FROM builder AS test
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry \
    --mount=type=cache,target=/usr/local/cargo/git,id=cargo-git \
    TARGET="$(cat /tmp/target-triple)" && \
    cargo test --target "$TARGET" -- --test-threads=1

FROM builder AS release
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry \
    --mount=type=cache,target=/usr/local/cargo/git,id=cargo-git \
    TARGET="$(cat /tmp/target-triple)" && \
    cargo build --release --target "$TARGET" && \
    cp "target/$TARGET/release/ollama-router" /ollama-router
RUN mkdir -p /empty-tmp

FROM scratch

COPY --from=release /empty-tmp /tmp
COPY --from=release /ollama-router /ollama-router

EXPOSE 11434 9090

ENTRYPOINT ["/ollama-router"]

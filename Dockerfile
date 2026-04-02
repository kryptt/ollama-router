# syntax=docker/dockerfile:1
FROM rust:1-alpine AS builder

RUN apk add --no-cache musl-dev

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src/ src/

# --- Test stage: docker buildx build --target test . ---
FROM builder AS test
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry \
    --mount=type=cache,target=/usr/local/cargo/git,id=cargo-git \
    cargo test

# --- Release build ---
FROM builder AS release
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry \
    --mount=type=cache,target=/usr/local/cargo/git,id=cargo-git \
    cargo build --release --target x86_64-unknown-linux-musl

# --- Runtime ---
FROM scratch

COPY --from=release /build/target/x86_64-unknown-linux-musl/release/ollama-router /ollama-router

EXPOSE 11434 9090

ENTRYPOINT ["/ollama-router"]

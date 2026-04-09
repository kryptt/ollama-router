# syntax=docker/dockerfile:1
FROM rust:1-alpine AS builder

RUN apk add --no-cache musl-dev

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY tests/ tests/

FROM builder AS test
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry \
    --mount=type=cache,target=/usr/local/cargo/git,id=cargo-git \
    cargo test --target x86_64-unknown-linux-musl -- --test-threads=1

FROM builder AS release
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry \
    --mount=type=cache,target=/usr/local/cargo/git,id=cargo-git \
    cargo build --release --target x86_64-unknown-linux-musl
RUN mkdir -p /empty-tmp

FROM scratch

COPY --from=release /empty-tmp /tmp
COPY --from=release /build/target/x86_64-unknown-linux-musl/release/ollama-router /ollama-router

EXPOSE 11434 9090

ENTRYPOINT ["/ollama-router"]

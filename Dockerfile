FROM rust:1-alpine AS builder

RUN apk add --no-cache musl-dev

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src/ src/

RUN cargo build --release --target x86_64-unknown-linux-musl

FROM scratch

COPY --from=builder /build/target/x86_64-unknown-linux-musl/release/ollama-router /ollama-router

EXPOSE 11434 9090

ENTRYPOINT ["/ollama-router"]

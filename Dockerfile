# Build stage
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS builder

WORKDIR /app

# Enable bytecode compilation and copy/link mode
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install dependencies
COPY pyproject.toml .
RUN uv sync --no-dev --no-install-project

# Copy source and install package
COPY src ./src
RUN uv sync --no-dev

# Runtime stage
FROM python:3.14-slim-bookworm

WORKDIR /app

# Install runtime dependencies for file parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Make sure we use the venv binaries
ENV PATH="/app/.venv/bin:$PATH"

# Create data directory for local storage
RUN mkdir -p /data/documents

# Copy source for editable-style imports (not strictly needed because package is installed in venv)
COPY src ./src

EXPOSE 8000

CMD ["raggit", "--help"]

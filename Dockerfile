# =============================================================================
# Multi-stage build for WaterFlow GPU workflows (CUDA 12.6 runtime)
# =============================================================================

FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    git \
    build-essential \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
RUN uv sync --frozen --no-install-project

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Install the project itself
RUN uv sync --frozen

# Pre-download ESM3 model to bake it into the image.
# The generate script loads esm3-open without authentication, so no HF token is needed.
ENV HF_HOME=/app/.cache/huggingface
RUN . .venv/bin/activate && python -c "\
from esm.models.esm3 import ESM3; \
ESM3.from_pretrained('esm3-open'); \
print('ESM3 model downloaded successfully')"

# Compile Python bytecode for faster startup
RUN . .venv/bin/activate && python -m compileall -q src/ scripts/

# Verify core GPU and preprocessing imports work
RUN . .venv/bin/activate && python -c "\
import torch; \
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}'); \
from torch_scatter import scatter_add; print('torch-scatter OK'); \
from torch_cluster import radius_graph; print('torch-cluster OK'); \
import pymol2; print('pymol2 OK')"

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# Minimal image with only runtime dependencies
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

WORKDIR /app

# Copy virtual environment and source from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/scripts /app/scripts
COPY --from=builder /app/.cache/huggingface /app/.cache/huggingface

# Copy entrypoint script
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create mount points for data volumes
RUN mkdir -p /data/pdb /data/cache /data/checkpoints /data/outputs /data/logs /data/splits

# Environment variables
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# CUDA configuration for H100 GPUs
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="9.0"

# HuggingFace cache location (pre-downloaded ESM3 model)
ENV HF_HOME=/app/.cache/huggingface

# Default data paths (can be overridden via docker run -e)
ENV WATERFLOW_PDB_DIR=/data/pdb
ENV WATERFLOW_CACHE_DIR=/data/cache
ENV WATERFLOW_CHECKPOINT_DIR=/data/checkpoints
ENV WATERFLOW_OUTPUT_DIR=/data/outputs
ENV WATERFLOW_LOG_DIR=/data/logs
ENV WATERFLOW_SPLITS_DIR=/data/splits

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--help"]

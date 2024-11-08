# Use multi-stage build for smaller final image
FROM ubuntu:23.10 AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    # Add environment variables for better PyTorch performance
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    CUDA_LAUNCH_BLOCKING=1

# Install system dependencies and security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3.11-dev \
    python3-dev \
    python3.11-venv \
    python3-pip \
    wget \
    git \
    poppler-utils \
    # Add dependencies for PDF and image processing
    libpoppler-dev \
    libpoppler-cpp-dev \
    pkg-config \
    # Add dependencies for PyTorch
    libgl1 \
    libglib2.0-0 \
    # Cleanup apt cache
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Final stage
FROM ubuntu:23.10

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    # Add environment variables for better PyTorch performance
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    CUDA_LAUNCH_BLOCKING=1 \
    # Add environment variables for memory management
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create non-root user
RUN groupadd -r datafoguser && useradd -r -g datafoguser -s /sbin/nologin -d /home/datafoguser datafoguser

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    poppler-utils \
    libpoppler-cpp-dev \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create and set permissions for required directories
RUN mkdir -p /tmp /home/datafoguser/tmp /home/datafoguser/app && \
    chown -R datafoguser:datafoguser /home/datafoguser && \
    chmod 755 /home/datafoguser && \
    chmod 1777 /tmp /home/datafoguser/tmp

# Copy application code
COPY --chown=datafoguser:datafoguser app /home/datafoguser/app

# Set working directory
WORKDIR /home/datafoguser/app

# Switch to non-root user
USER datafoguser

# Expose port
EXPOSE 8000

# Set entrypoint script
COPY --chmod=755 docker-entrypoint.sh /usr/local/bin/
ENTRYPOINT ["docker-entrypoint.sh"]
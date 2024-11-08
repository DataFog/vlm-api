#!/bin/bash
set -e

# Activate virtual environment
. /opt/venv/bin/activate

# Start the FastAPI application with memory optimizations
exec uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 2 \
    --timeout 300 \
    --limit-max-requests 1000
# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/root/.job_evaluator

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy configuration and project files
COPY pyproject.toml .
COPY README.md .

# Install dependencies in a cache-friendly way
RUN uv venv && uv pip install --no-cache -r pyproject.toml

# Copy source code
COPY src/ src/

# Install project
RUN uv pip install -e .

# Create the data directory
RUN mkdir -p /root/.job_evaluator

# Set command to run the MCP server
ENTRYPOINT ["uv", "run", "job-scan-mcp"]

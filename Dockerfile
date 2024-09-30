# Use an official Python runtime as a parent image
FROM python:3.9-slim

RUN pip install torch==2.4.0+cpu \
    torchvision==0.19.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu \
    && rm -rf /root/.cache/pip

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy pyproject.toml and uv.lock
COPY pyproject.toml uv.lock ./

# Create a virtual environment and install dependencies
RUN uv venv && \
    . .venv/bin/activate && \
    uv pip install .

# Copy the rest of the application code
COPY src ./src

# Declare volumes
VOLUME ["/app/samples", "/app/predictions", "/app/data"]

# Set the entrypoint to use the virtual environment
ENTRYPOINT ["/app/.venv/bin/python"]
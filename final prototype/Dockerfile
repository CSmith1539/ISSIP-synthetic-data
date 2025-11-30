# Use a minimal Python base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies required for building Python packages and running TF
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        wget \
        curl \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libffi-dev \
        libssl-dev \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy your actual project code
COPY . .

# Set default command
CMD ["python", "run.py"]

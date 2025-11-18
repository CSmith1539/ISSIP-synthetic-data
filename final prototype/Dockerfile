
# Use a minimal Python base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your actual project code
COPY . .

# Default command (change to your real entry point)
CMD ["python", "run.py"]

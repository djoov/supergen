FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /app

# Install system dependencies (required for building some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Note: we are installing standard dependencies. Make sure requirements.txt is accurate.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the API port if running the FastAPI server
EXPOSE 8000

# By default, run the CLI. 
# You can override this in docker-compose.yml to run the FastAPI server instead (uvicorn main:app --host 0.0.0.0 --port 8000)
CMD ["python", "cli.py"]

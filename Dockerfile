# Use slim Python image to reduce size
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make startup script executable (in case chmod was lost)
RUN chmod +x start.sh

# Cloud Run uses PORT env var (default 8080)
ENV PORT=8080

# Cloud Run only exposes ONE port — Streamlit takes it
EXPOSE 8080

# Run the startup script (launches both backend + UI)
CMD ["./start.sh"]
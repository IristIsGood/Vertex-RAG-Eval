#!/bin/bash
set -e

# Start FastAPI backend in the background on port 8000
echo "🚀 Starting FastAPI backend on internal port 8000..."
uvicorn app:app --host 0.0.0.0 --port 8000 &

# Wait a moment for backend to be ready
sleep 5

# Start Streamlit in the foreground on PORT (default 8080 for Cloud Run)
echo "🎨 Starting Streamlit UI on port ${PORT:-8080}..."
streamlit run ui.py \
  --server.port "${PORT:-8080}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
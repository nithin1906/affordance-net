# backend/Dockerfile

# 1. Use Python 3.12 to match your local environment
FROM python:3.12-slim

# 2. Install system dependencies required for OpenCV/YOLO
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Set working directory
WORKDIR /app

# 4. Install Python dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# 5. Copy the application code
COPY backend /app/backend
COPY backend_models /app/backend_models

# 6. Copy the base YOLO model if it exists in root
COPY yolov8n.pt /app/

# 7. Expose the port
EXPOSE 8000

# 8. Run the server
CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8000"]
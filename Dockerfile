FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if any needed for building wheels)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code (backend and frontend)
COPY backend ./backend
COPY frontend ./frontend

# Expose the FastAPI port
EXPOSE 8001

# Environment variables (can be overridden at runtime)
ENV HOST=0.0.0.0
ENV PORT=8001

# Run the application
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]

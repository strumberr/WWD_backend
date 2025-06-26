# Use a minimal base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

RUN python setup.py

# Install only necessary system dependencies
RUN apt-get update && apt-get install -y git && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy only essential files
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Now copy only the required source files
COPY main.py inference.py self_play.py test.py model/ /app/

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Use a minimal base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install only necessary system dependencies
RUN apt-get update && apt-get install -y git && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy setup.py first so we can run it
COPY setup.py .

# Run setup.py to download model into model/
RUN python setup.py

# Now copy the rest of your source files
COPY main.py inference.py self_play.py test.py /app/

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

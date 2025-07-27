FROM python:3.10-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential gcc wget git \
    && rm -rf /var/lib/apt/lists/*

# Preinstall only requirements
COPY requirement.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirement.txt

# Copy only necessary code
COPY app.py ./app.py
COPY . .
# Avoid copying static if not needed at build time

CMD ["python", "app.py"]


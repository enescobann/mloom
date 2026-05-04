# Use a slim version of Python to keep the image size small
FROM python:3.12.3-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (useful if you use PostgreSQL later)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the project files
COPY pyproject.toml .
COPY mloom/ ./mloom/

RUN pip install --no-cache-dir .[server]

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the app using uvicorn
CMD ["uvicorn", "mloom.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
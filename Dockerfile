# Use the official slim Python image (≈150 MB)
FROM python:3.11-slim

# Prevent Python from writing .pyc files & buffer issues
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy only requirements first (leveraging Docker cache)
COPY requirements.txt .

# Install dependencies without caching wheels
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application
COPY . .

# Expose the port Railway will use
EXPOSE  $PORT

# Use Gunicorn to serve; adjust run:app if your entrypoint differs
CMD ["gunicorn", "run:app", "--bind", "0.0.0.0:$PORT"]

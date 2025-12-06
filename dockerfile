# 1. Use a lightweight Python base image
FROM python:3.11-slim

# 2. Environment settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. Set working directory
WORKDIR /app

# 4. Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# 5. Copy dependency files first (leverages Docker cache)
COPY requirements.txt pyproject.toml ./

# 6. Install Python dependencies
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 7. Copy the full project
COPY . .
COPY data/embeddings /app/data/embeddings


# 8. Expose Streamlit port
EXPOSE 8501

# 9. Start the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

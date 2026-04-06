# 1. Use a Python base image (3.10 is stable for most ML)
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /code

# 3. Install system dependencies (needed for Git LFS and building some libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy only requirements first (this makes rebuilding faster)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your files (app.py, best_model.pth, input.txt, etc.)
COPY . .

# 6. Hugging Face Spaces run as a non-root user (ID 1000)
# This ensures your app has permission to read the model files
RUN chmod -R 777 /code

# 7. Tell Hugging Face which port to look at
EXPOSE 7860

# 8. Start Streamlit
# --server.address=0.0.0.0 is MANDATORY for Docker
# --server.port=7860 is MANDATORY for Hugging Face
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
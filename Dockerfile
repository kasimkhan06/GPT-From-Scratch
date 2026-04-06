FROM python:3.10-slim

WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all folders into the container
COPY . .

# Set permissions for Hugging Face user
RUN chmod -R 777 /code

# Expose HF Port
EXPOSE 7860

# Run Streamlit from the root, pointing to the file in src/
# Note: We keep the working directory at /code so that 
# relative paths like "models/best_model.pth" still work.
CMD ["streamlit", "run", "src/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
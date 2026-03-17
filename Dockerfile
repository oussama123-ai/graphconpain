FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y \
    git wget curl libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 \
    libxext6 ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# PyTorch Geometric
RUN pip install torch-geometric torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

# Copy source
COPY . .

# Default command
CMD ["python", "scripts/inference.py", "--help"]

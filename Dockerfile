FROM nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu22.04

ENV PYTHON_VERSION=3.12

# Install Python 3.12 and system tools
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    tmux \
    git \
    vim \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Install PyTorch with CUDA 12.8 support
# Use torch's default index to get CUDA-compatible wheels
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128

WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
    numpy==2.2.6 \
    pandas==2.3.3 \
    scikit-learn==1.7.2 \
    xgboost==3.2.0 \
    shap==0.49.1 \
    imbalanced-learn==0.14.1 \
    matplotlib==3.10.8 \
    tqdm==4.67.3 \
    pyyaml==6.0.3 \
    joblib==1.5.3 \
    pytest==9.0.2

# Install project package
COPY . .
RUN pip install --no-cache-dir -e .

CMD ["bash"]

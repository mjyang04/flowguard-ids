FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Install system tools
RUN apt-get update && apt-get install -y \
    tmux \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

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

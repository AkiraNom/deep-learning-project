# ベースとなるDockerイメージを指定
FROM python:3.12-slim

# 必要なパッケージをインストール
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /scripts

# Copy requirements file
COPY requirements.txt .

# Install Python packages
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    rm -rf ~/.cache/pip

# execute jupyterlab as a default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--LabApp.token=''", "--notebook-dir=/scripts"]

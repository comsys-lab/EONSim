FROM ubuntu:24.04

# Set noninteractive installation
ENV DEBIAN_FRONTEND=noninteractive

# Add user creation arguments
ARG USER_ID
ARG GROUP_ID
ARG USERNAME=appuser

# Create a non-root user with the same UID/GID as the host user
RUN groupadd -g ${GROUP_ID} ${USERNAME} && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USERNAME}

# Install required packages
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    python3-pybind11 \
    pybind11-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and install Python 3.12.3
WORKDIR /tmp
RUN wget https://www.python.org/ftp/python/3.12.3/Python-3.12.3.tgz \
    && tar xzf Python-3.12.3.tgz \
    && cd Python-3.12.3 \
    && ./configure --enable-optimizations \
    && make -j $(nproc) \
    && make install \
    && cd .. \
    && rm -rf Python-3.12.3 Python-3.12.3.tgz

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install required Python packages
RUN pip3 install \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    tqdm \
    h5py \
    faiss-cpu \
    wikipedia-api \
    sentence-transformers \
    transformers \
    datasets \
    torch \
    huggingface-hub \    
    pybind11 \
    # Additional libraries for Working set
    scikit-learn \
    scipy \
    plotly \
    ipykernel \
    jupyter \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    isort

RUN pip3 install \
    colorama

# Change ownership of necessary directories to the new user
RUN mkdir -p /workspace && chown -R ${USERNAME}:${USERNAME} /workspace

# # Create directories for data and results
# RUN mkdir -p /app/logs /app/data /app/results /app/bigann /app/deep1b

# # Set working directory
WORKDIR /workspace

# Optional: Copy requirements.txt if you want to manage dependencies separately
# COPY requirements.txt .
# RUN pip3 install -r requirements.txt

# Switch to the non-root user
USER ${USERNAME}

ENTRYPOINT ["/bin/bash"]
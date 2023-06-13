# Base CUDA devel image
FROM nvidia/cuda:11.7.0-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace/scrolls

# Apt-get installs
RUN \
    apt update && \
    apt-get -y install git unzip wget tmux curl libgl1 libglib2.0-0

# Install Miniconda
RUN \
    wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh && \
    bash Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b && \
    rm Miniconda3-py310_23.1.0-1-Linux-x86_64.sh

# Add paths to PATH and PYTHONPATH
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Create conda env
RUN conda init bash
RUN \
    conda create \
    --name scrolls \
    python=3.10 \
    gcc_linux-64 gxx_linux-64 libgcc cudatoolkit=11.7 \
    -c conda-forge -y
RUN /root/miniconda3/envs/scrolls/bin/python -m pip install --upgrade pip

# Install torch & other related deps: 
# - not in requirements.txt because order of install matters
# - torch first due to some dependencies importing torch during install
RUN \
    /root/miniconda3/envs/scrolls/bin/pip install \
    --default-timeout=1000 \
    --extra-index-url https://download.pytorch.org/whl/cu117 \
    torch==2.0.1+cu117 \
    torchvision==0.15.2+cu117

# Install requirements
COPY requirements.txt .
RUN /root/miniconda3/envs/scrolls/bin/pip install \
    --default-timeout=1000 \
    -r requirements.txt 

# Install libvips: need to be done after pip install pyvips 
# (which is in requirements.txt) due to some weird bug
RUN \
    apt-get -y install libvips libvips-dev libvips-tools g++-9

# Add repo folder to PYTHONPATH
ENV PYTHONPATH="/workspace/scrolls:${PYTHONPATH}"
ARG PYTHONPATH="/workspace/scrolls:${PYTHONPATH}"
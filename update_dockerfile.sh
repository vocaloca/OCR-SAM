#!/bin/bash

# Script to add font installation commands to an existing Dockerfile

echo "=== Adding Asian font support to Dockerfile ==="

# Font installation commands for Dockerfile
FONT_INSTALL_COMMANDS='

# Install fonts for multilingual support
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-noto-cjk \
        fonts-noto-cjk-extra \
        fonts-noto \
        fonts-nanum \
        fonts-wqy-zenhei \
        fonts-wqy-microhei \
        fonts-arphic-ukai \
        fonts-arphic-uming \
        fonts-thai-tlwg \
        fonts-dejavu-core \
        fontconfig \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -fv \
    && export DEBIAN_FRONTEND=dialog'

# Check if Dockerfile exists
if [ -f "Dockerfile" ]; then
    # Create backup of original Dockerfile
    cp Dockerfile Dockerfile.bak
    
    # Insert font installation commands before the last COPY command
    awk -v cmds="$FONT_INSTALL_COMMANDS" '
    /^COPY/ && !seen {
        print cmds
        seen=1
    }
    {print}
    ' Dockerfile.bak > Dockerfile
    
    echo "=== Updated Dockerfile with font installation commands ==="
    echo "Original Dockerfile saved as Dockerfile.bak"
else
    echo "Dockerfile not found in current directory."
    echo "Creating a new Dockerfile snippet with font installation commands..."
    
    echo 'FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

WORKDIR /workspace

# Basic dependencies
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt update \
    && apt install git wget unzip ffmpeg libsm6 libxext6 -y \
    && apt autoremove -y \
    && apt clean -y \
    && export DEBIAN_FRONTEND=dialog

# Install fonts for multilingual support
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-noto-cjk \
        fonts-noto-cjk-extra \
        fonts-noto \
        fonts-nanum \
        fonts-wqy-zenhei \
        fonts-wqy-microhei \
        fonts-arphic-ukai \
        fonts-arphic-uming \
        fonts-thai-tlwg \
        fonts-dejavu-core \
        fontconfig \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -fv \
    && export DEBIAN_FRONTEND=dialog

# Application specific dependencies
RUN pip --no-cache-dir install -U openmim
RUN mim install mmengine mmocr "mmcv==2.0.0rc4" "mmdet==3.0.0rc5" "mmcls==1.0.0rc5"
RUN pip --no-cache-dir install git+https://github.com/facebookresearch/segment-anything.git
RUN pip --no-cache-dir install gradio==4.32.0 numpy omegaconf==2.3.0 einops==0.6.0 transformers==4.27.3 pytorch-lightning==2.0.1.post0 diffusers==0.14.0 google-cloud-storage google-cloud-pubsub

COPY . /workspace' > Dockerfile.with_fonts
    
    echo "=== Created new Dockerfile.with_fonts ==="
    echo "You can use this file as your Dockerfile."
fi

echo "=== Done ===" 
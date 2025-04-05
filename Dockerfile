FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime


WORKDIR /workspace
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt update \
    && apt install git wget unzip ffmpeg libsm6 libxext6 -y \
    && apt autoremove -y \
	&& apt clean -y \
	&& export DEBIAN_FRONTEND=dialog
RUN pip --no-cache-dir install -U openmim
RUN mim install mmengine mmocr 'mmcv==2.0.0rc4' 'mmdet==3.0.0rc5' 'mmcls==1.0.0rc5'
RUN pip --no-cache-dir install git+https://github.com/facebookresearch/segment-anything.git

RUN pip --no-cache-dir install gradio==4.32.0 numpy omegaconf==2.3.0 einops==0.6.0 transformers==4.27.3 pytorch-lightning==2.0.1.post0 diffusers==0.14.0 diffusers==0.14.0 google-cloud-storage google-cloud-pubsub


# COPY requirements.txt .
# RUN pip install -r requirements.txt
# 4.32.0
COPY . /workspace
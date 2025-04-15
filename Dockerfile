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

# # Multilingual text rendering - install system packages and then make PyGObject available to Python
# RUN apt update && apt install -y \
#     libcairo2-dev \
#     pkg-config \
#     python3-dev \
#     python3-gi \
#     python3-gi-cairo \
#     python3-cairo \
#     gir1.2-gtk-3.0 \
#     gir1.2-pango-1.0 \
#     libcairo-gobject2 \
#     gobject-introspection \
#     libgirepository1.0-dev

# # Install PyGObject for the current Python environment
# RUN pip install --no-binary=cairo pycairo
# RUN PYTHONPATH=$PYTHONPATH:/usr/lib/python3/dist-packages pip install PyGObject

# # Add the system site-packages directory to Python path so gi module can be found
# ENV PYTHONPATH="${PYTHONPATH}:/usr/lib/python3/dist-packages"

# # Verify that we can import the modules
# RUN python -c "import gi; gi.require_version('Pango', '1.0'); from gi.repository import Pango; import cairo; print('GObject and Cairo installation successful')"

# COPY requirements.txt .
# RUN pip install -r requirements.txt
# 4.32.0
COPY . /workspace
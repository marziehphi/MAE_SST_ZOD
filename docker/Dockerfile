ARG PYTORCH="1.9.0"
ARG CUDA="11.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# Set bash as the default shell
ENV CUDA_HOME=/usr/local/cuda
ENV SHELL=/bin/bash

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Build with some basic utilities
RUN apt-get update && apt-get install -y \
    sudo \
    python3-pip \
    apt-utils \
    vim \
    git 

# Install MMCV, MMDetection and MMSegmentation
# Version of zod kit chosed based on env python version 3.7.10.
RUN pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
RUN pip install mmdet==2.14.0
RUN pip install mmsegmentation==0.14.1
RUN pip install zod==0.1.0 

# Install MMDetection3D
RUN conda clean --all
COPY . /workspace
WORKDIR /workspace
ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST="Volta;Turing;Ampere"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .


RUN pip install ipdb
RUN pip install numba==0.48
RUN pip install h5py 
RUN pip install dataclass_wizard 

# uninstall pycocotools installed by nuscenes-devkit and reinstall mmpycocotools
RUN conda install cython
RUN pip uninstall pycocotools --no-cache-dir -y
RUN pip install mmpycocotools --no-cache-dir --force --no-deps

RUN pip install jupyterlab
RUN pip install -r jupyterlab.txt

# optional 
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
EXPOSE 8888
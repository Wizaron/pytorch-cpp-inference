FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
         g++ \
         make \
         cmake \
         wget \
         unzip \
         vim \
         git \
         libopencv-dev \
         libboost-all-dev \
         python3 \
         python3-pip

RUN pip3 install numpy "pillow<7"
RUN pip3 install torch torchvision

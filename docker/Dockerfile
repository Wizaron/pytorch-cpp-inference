FROM ubuntu:18.04
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
         python-pip

RUN pip install https://download.pytorch.org/whl/cpu/torch-1.0.0-cp27-cp27mu-linux_x86_64.whl
RUN pip install numpy pillow torchvision

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
         python3 \
         python3-pip

RUN pip3 install numpy "pillow<7"
RUN pip3 install torch==1.3.1+cpu torchvision==0.4.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

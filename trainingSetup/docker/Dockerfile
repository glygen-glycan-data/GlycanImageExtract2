#
# Based on work by Henrycke Bozza Schenberk
# 
# See: https://github.com/Sherensberk/DockerDarkNet
#      https://hub.docker.com/r/sherensberk/darknet
#      https://medium.com/@oschenberk/darknet-yolo-and-docker-02f52a927aba
#
# Rebuilt and tweaked to provide updated darknet version
#

ARG cuda_version=12.4.1
ARG so_version=ubuntu22.04
ARG flavor=devel

FROM nvidia/cuda:${cuda_version}-cudnn-${flavor}-${so_version}

ARG driver_version=550
ARG opencv_version=4.10.0

ENV DEBIAN_FRONTEND=noninteractive 

WORKDIR /deps

RUN apt-get update -y
RUN apt-get purge cmake && \
    apt-get install -y \
    build-essential    \
    git                \
#    libopencv-dev      \
    file               \
    libssl-dev         \
    libtclap-dev       \
    libmagic-dev       \
    nvidia-driver-${driver_version}

ADD https://github.com/Kitware/CMake/releases/download/v3.30.1/cmake-3.30.1.tar.gz cmake-3.30.1.tar.gz
RUN  tar -xf cmake-3.30.1.tar.gz && cd cmake-3.30.1 && ./configure && make -j8 && make install && rm -rf /deps

ADD https://github.com/opencv/opencv/archive/refs/tags/${opencv_version}.tar.gz opencv${opencv_version}.tar.gz
RUN tar -xf opencv${opencv_version}.tar.gz && mkdir -p build && cd build && cmake ../opencv-${opencv_version} && make -j8 && make install

WORKDIR /src

ARG build_date

RUN git clone https://github.com/hank-ai/darknet

COPY install.sh .
RUN chmod +x install.sh && ./install.sh

RUN rm -rf /src

WORKDIR /src

ARG CUDA_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libomp-dev \
        ninja-build \
        pkg-config \
        python3 \
        util-linux \
    && rm -rf /var/lib/apt/lists/*

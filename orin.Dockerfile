ARG BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.4.0
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-dev \
      libsm6 \
      libxext6 \
      ffmpeg \
      libhdf5-serial-dev \
      libtesseract-dev \
      libgtk-3-0 \
      libtbb12 \
      libtbb2 \
      libatlas-base-dev \
      libopenblas-dev \
      build-essential \
      python3-setuptools \
      make \
      cmake \
      nasm \
      git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /workspace

COPY pyproject.toml .

# Set to get precompiled jetson wheels
RUN export PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126 && \
    export PIP_TRUSTED_HOST=pypi.jetson-ai-lab.io && \
    pip3 install --upgrade pip setuptools && \
    pip3 install -e .[orin]

RUN pip3 install "git+https://github.com/facebookresearch/pytorch3d.git"

# Build and install decord
RUN cd /tmp && \
    git clone https://git.ffmpeg.org/ffmpeg.git && \
    cd ffmpeg && \
    git checkout n4.4.2 && \
    ./configure --enable-shared --enable-pic --prefix=/usr && \
    make -j$(nproc) && \
    make install && \
    cd /tmp && \
    git clone --recursive https://github.com/dmlc/decord && \
    cd decord && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make && \
    cd ../python && \
    python3 setup.py install --user && \
    cd /workspace && \
    rm -rf /tmp/ffmpeg /tmp/decord

# Set decord library path environment variable
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.local/decord/

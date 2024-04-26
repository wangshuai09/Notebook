# 新建用户
RUN groupadd -g 1004 service && \
    useradd -u 1004 -g 1004 --create-home service

# sudo 权限
RUN apt-get update && apt-get install -y sudo && echo "service ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers


FROM ubuntu:20.04

RUN groupadd -g 1004 service && \
    useradd -u 1004 -g 1004 --create-home service
WORKDIR /home/service

# Install Dependencies
RUN sed -i 's/ports.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y \
    gcc \
    g++ \
    make \
    cmake \
    zlib1g \
    zlib1g-dev \
    openssl \
    libsqlite3-dev \
    libssl-dev \
    libffi-dev \
    unzip \
    pciutils \
    net-tools \
    libblas-dev \
    gfortran \
    libblas3 \
    libopenblas-dev \
    git \
    wget \
    curl \
    vim && \
    rm -rf /var/lib/apt/lists/*

# Install Python
ENV MINICONDA_FILE=Miniconda3-latest-Linux-aarch64.sh
ENV MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh

RUN wget -O ${MINICONDA_FILE} ${MINICONDA_URL} && \
    chmod +x ${MINICONDA_FILE} && \
    bash ${MINICONDA_FILE} -b -p /root/miniconda && \
    /root/miniconda/bin/conda create --name torch_npu -y python=3.10.6 && \
    rm -f ${MINICONDA_FILE}

ENV PATH=/root/miniconda/envs/torch_npu/bin/:${PATH}

# Install Python Packages
ENV PIP_SOURCE_URL=https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install pip --no-cache-dir --upgrade -i ${PIP_SOURCE_URL} && \
    pip install --no-cache-dir attrs cython numpy decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py -i ${PIP_SOURCE_URL} && \
    pip install --no-cache-dir wheel pyyaml typing_extensions expecttest -i ${PIP_SOURCE_URL}

# Install CANN toolkit
ENV CANN_TOOLKIT_FILE=Ascend-cann-toolkit_7.0.0_linux-aarch64.run
ENV CANN_TOOLKIT_URL=https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%207.0.0/Ascend-cann-toolkit_7.0.0_linux-aarch64.run?response-content-type=application/octet-stream
ENV LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:${LD_LIBRARY_PATH}

RUN wget -O ${CANN_TOOLKIT_FILE} ${CANN_TOOLKIT_URL} && \
    chmod +x ${CANN_TOOLKIT_FILE} && \
    sh -c  '/bin/echo -e "Y" | ./${CANN_TOOLKIT_FILE} --install' && \
    rm -f ${CANN_TOOLKIT_FILE}

ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:${ASCEND_TOOLKIT_HOME}/lib64/plugin/opskernel:${ASCEND_TOOLKIT_HOME}/lib64/plugin/nnengine:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}
ENV PYTHONPATH=${ASCEND_TOOLKIT_HOME}/python/site-packages:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe:${PYTHONPATH}
ENV PATH=${ASCEND_TOOLKIT_HOME}/bin:${ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin:${PATH}
ENV ASCEND_AICPU_PATH=${ASCEND_TOOLKIT_HOME}
ENV ASCEND_OPP_PATH=${ASCEND_TOOLKIT_HOME}/opp
ENV TOOLCHAIN_HOME=${ASCEND_TOOLKIT_HOME}/toolkit
ENV ASCEND_HOME_PATH=${ASCEND_TOOLKIT_HOME}

# Install CANNN Kernels
ENV CANN_KERNELS_FILE=Ascend-cann-kernels-910b_7.0.0_linux.run
ENV CANN_KERNELS_URL=https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%207.0.0/Ascend-cann-kernels-910b_7.0.0_linux.run?response-content-type=application/octet-stream

RUN wget -O ${CANN_KERNELS_FILE} ${CANN_KERNELS_URL} && \
    chmod +x ${CANN_KERNELS_FILE} && \
    sh -c  '/bin/echo -e "Y" | ./${CANN_KERNELS_FILE} --install' && \
    rm -f ${CANN_KERNELS_FILE}

# Install StableDiffufsion
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN git clone --branch dev https://github.com/AUTOMATIC1111/stable-diffusion-webui.git && \
    cd stable-diffusion-webui && \ 
    python -m venv venv && \
    source ./venv/bin/activate && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir opencv-python-headless && \
    pip install --no-cache-dir torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir torch-npu==2.1.0 && \ 
    pip install https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip --prefer-binary && \
    pip install https://github.com/mlfoundations/open_clip/archive/bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b.zip --prefer-binary && \ 
    pip install -U -I --no-deps xformers==0.0.23.post1 --prefer-binary && \ 
    pip install install ngrok --prefer-binary  && \
    mkdir repositories && \
    cd repositories && \
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui-assets.git stable-diffusion-webui-assets && \ 
    git -C stable-diffusion-webui-assets checkout 6f7db241d2f8ba7457bac5ca9753331f0c266917 && \ 
    git clone https://github.com/Stability-AI/stablediffusion.git stable-diffusion-stability-ai && \ 
    git -C stable-diffusion-stability-ai checkout cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf && \
    git clone https://github.com/Stability-AI/generative-models.git generative-models && \ 
    git -C generative-models checkout 45c443b316737a4ab6e40413d7794a7f5657c19f && \
    git clone https://github.com/crowsonkb/k-diffusion.git k-diffusion && \ 
    git -C k-diffusion checkout ab527a9a6d347f364e3d185ba6d714e22d80cb3c && \ 
    git clone https://github.com/salesforce/BLIP.git BLIP && \ 
    git -C BLIP checkout 48211a1594f1321b00f14c9f7a5b4813144b2fb9 && \
    cd .. && \ 
    pip install --no-cache-dir -r requirements_versions.txt && \ 
    pip install --no-cache-dir -r requirements_npu.txt && \ 
    pip cache purge

ENV LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1
COPY v1-5-pruned-emaonly.safetensors stable-diffusion-webui/models/Stable-diffusion

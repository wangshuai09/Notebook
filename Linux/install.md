# docker 安装
```shell
apt-get update
apt install docker.io
# 使用 gpu 需安装 NVIDIA Container Toolkit
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit
systemctl restart docker
```

# conda 安装
```shell
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

# cuda 安装
```shell
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
echo export PATH=$PATH:/usr/local/cuda-11.8/bin>>~/.bashrc 
echo export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64>>~/.bashrc
```

# gpu-docker 配置
```shell
From nvidia/cuda:11.8.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteactive
ENV DEBCONF_NONINTERACTIVE_SEEN true

WORKDIR /root

RUN sed -i 's/ports.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt update && \
    apt install -y adduser sudo vim gcc g++ cmake make gdb git tmux openssh-server net-tools iputils-ping python3 python3-venv python3-setuptools gcc python3-dev patchelf tree && \
    apt remove -y cmake

```

# ping
```shell
apt install iputils-ping
```

# vim 
```shell
apt-get update 
apt-get install -y vim 
```


# Install Accelerate CI 
```shell
apt-get update
apt-get install -y vim 
pip install pytest transformers evaluate scikit-learn timm black -i https://pypi.tuna.tsinghua.edu.cn/simple
export LD_PRELOAD=/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
```


# miniconda
```shell
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

# 模型下载
```shell
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download --local-dir-use-symlinks False bigscience/bloom-560m --local-dir bloom-560m
```

# 多版本python配置
export PATH=/opt/python/pp39-pypy39_pp73/bin/:$PATH

# pip 安装时卡住
pip install --verbose package_name # 可以显示更多信息
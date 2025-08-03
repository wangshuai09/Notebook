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


# 模型下载
```shell
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download --local-dir-use-symlinks False bigscience/bloom-560m --local-dir bloom-560m
```

# 多版本python配置
export PATH=/opt/python/pp39-pypy39_pp73/bin/:$PATH

# upgrade gcc/g++
```shell
# Add the Ubuntu Toolchain PPA:
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
# Update the package index:
sudo apt update
# Install GCC 12:
sudo apt install gcc-12 g++-12
# Update the default GCC version:
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12
# Update the default G++ version:
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 12
# Verify GCC version:
gcc --version (should be 12.x)
```

# mihomo for linux
```shell
# 1. goto https://github.com/MetaCubeX/mihomo/releases
# 2. find version for linux and download, such as mihomo-linux-amd64-go120-v1.18.9.deb
# 3. install
dpkg -i mihomo-linux-amd64-go120-v1.18.9.deb
# 4. config: config.yaml from windows
cp cofing.yaml /etc/mihomo
# 5. service
systemctl enable mihomo
systemctl start mihomo
# 6. ubuntu net 配置
#    Network->Proxy->Manual->Http(s) proxy: 127.0.0.1:7890
# 7. 监控窗口： https://yacd.metacubex.one/#/
```

# google chrome
```shell
# 1. download https://www.google.com/chrome
# 2. install
dpkg -i google-chrome-stable_current_amd64.deb

# root 用户点击图标无反应
sed -i '$s/$/ --no-sandbox/' /usr/bin/google-chrome
```

# vscode for linux
```shell
# 1. download https://code.visualstudio.com/download
# 2. install
dpkg -i https://code.visualstudio.com/download

# root 用户点击图标无反应
sed '$a alias vscode='\''/usr/bin/code --no-sandbox --user-data-dir .'\''' ~/.bashrc
```

# cmake 升级
```shell
# install, 默认源一般不是最新的版本
apt install cmake

# upgrade, 源码编译
apt install libssl-dev

wget https://cmake.org/files/v3.26/cmake-3.26.6.tar.gz
tar -xvzf cmake-3.26.6.tar.gz
cd cmake-3.26.6
chmod -R 777 *
./configure
make
sudo make install

# link
rm -rf /usr/bin/cmake
ln -s /usr/local/bin/cmake /usr/bin/cmake
# 或者
update-alternatives --install /usr/bin/cmake cmake /usr/local/bin/cmake 1 --force
```

# gnu binutils 升级
```shell
apt-get install texinfo
wget https://ftp.gnu.org/gnu/binutils/binutils-x.y.tar.gz  # x.y 是版本号
tar -xf binutils-x.y.tar.gz
cd binutils-x.y
./configure --prefix=/usr
make
sudo make install
as --verion
```

# locate
apt-get install mlocate
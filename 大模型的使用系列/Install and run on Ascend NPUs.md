Install and run on Ascend NPUs

## Linux
Ascend CANN packages can only installed on Linux.

Preparation: Before installing stable-diffusion-webui for NPU, you should make sure that have installed the right [CANN toolkit and kernels](https://www.hiascend.com/developer/download/community/result?module=cann&cann=7.0.0.beta1).

#### Automatic Installation

1. Install Python3.10.6

    First methond（Recommend）：
    Install [miniconda](https://docs.conda.io/projects/miniconda/en/latest/)

    ```shell
    conda create -n python310 python=3.10.6
    conda activate python310
    ```

    Second method：
    Install [Python3.10.6](https://www.python.org/ftp/python/3.10.6/Python-3.10.6.tgz)

    ```shell
    tar -zxvf Python-3.10.6.tgz
    cd Python-3.10.6
    ./configure
    make -j4 
    make insall 
    rm /usr/bin/python
    ln -s /usr/local/python3/bin/python3.10 /usr/bin/python
    ```

2. Start run stable-diffusion-webui
    `./webui.sh --skip-torch-cuda-test --no-half`
    this command will install torch and torch_npu on Ascend device automatically when you first install. 

#### Manual Installation
This Manual installtaion is following the dev branch of commit id: cc3f604, which maybe out-of-date.

```shell
# install miniconda, for other versions: https://docs.conda.io/projects/miniconda/en/latest/
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh

conda create -n torch_npu python=3.10.6
conda activate torch_npu

# install stable-diffusion-webui
git clone --branch dev https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu 
pip install torch_npu==2.1.0
pip install https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip --prefer-binary
pip install https://github.com/mlfoundations/open_clip/archive/bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b.zip 
pip install -U -I --no-deps xformers==0.0.23.post1 
pip install install ngrok
mkdir repositories 
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui-assets.git stable-diffusion-webui-assets
git -C stable-diffusion-webui-assets checkout 6f7db241d2f8ba7457bac5ca9753331f0c266917 
git clone https://github.com/Stability-AI/stablediffusion.git stable-diffusion-stability-ai 
git -C stable-diffusion-stability-ai checkout cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf 
git clone https://github.com/Stability-AI/generative-models.git generative-models 
git -C generative-models checkout 45c443b316737a4ab6e40413d7794a7f5657c19f
git clone https://github.com/crowsonkb/k-diffusion.git k-diffusion 
git -C k-diffusion checkout ab527a9a6d347f364e3d185ba6d714e22d80cb3c
git clone https://github.com/salesforce/BLIP.git BLIP 
git -C BLIP checkout 48211a1594f1321b00f14c9f7a5b4813144b2fb9 
pip install -r requirements.txt 
pip install -r requirements_for_npu.txt 
```

## Use directly

There is a packageed image, based on above manual steps. And this image has installed CANN toolkit,  kernels and v1-5-pruned-emaonly.safetensors in advance.
You can simply use this image:

```shell 
docker pull wangshuai09/ubuntu-cann7.0.0-torch21-py310-stable_diffusion:latest
docker run --network host --name torch_npu --device /dev/davinci2 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -v /data/disk3/wangshuai:/root/wangshuai -it wangshuai09/ubuntu-cann7.0.0-torch21-py310-stable_diffusion:latest bash
cd stable-diffusion-webui
./webui.sh --listen --skip-torch-cuda-test --no-half
```

This guide has been tested with CANN 7.0.0, Python 3.9.6, PyTorch 2.1.0, torch-npu 2.1.0.
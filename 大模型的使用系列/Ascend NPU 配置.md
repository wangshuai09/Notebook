## Ascend NPU 配置

### 简介：
昇腾 HUAWEI Ascend（NPU）是华为全栈全场景人工智能芯片。针对不同的应用场景，昇腾提供了多种多样的规格。不同的架构或者不同的算力水平，可以适配不同的应用需求，这些需求包括视频解析/模型推理/模型训练/大模型训练，端侧部署/集群部署等。
https://www.hiascend.com/zh/document

昇腾系列芯片可以理解为对标 NVIDIA RTX4090/H100/A100 的 AI 芯片，当前昇腾提供的算力芯片已经可以与英伟达掰一掰腕子。在漂亮国实施芯片出口禁令后，中国企业就无法获得高性能的 GPU 芯片了。而近期美国商务部长雷蒙多发言称要限制中国公司使用美国的云服务来训练自己的AI大模型，更是堵死了国内企业通过云服务或者第三方租赁的方式来绕过芯片出口禁令的门路。在大语言模型（LLM）如火如荼的今天，高性能芯片之争不亚于新世纪的军备竞赛。

在芯片禁令出了后，昇腾成功扛起了国产算力的大旗，解决了国内企业对大模型应用的高算力需求，但是英伟达的成功不仅在于其芯片，更在于其构建的生态，其推出的统一计算设备架构（CUDA）已经成为深度学习领域事实上的标准，例如 Pytorch 默认使用 CUDA 作为其后端加速器。

### 基础环境配置

```shell
# 系统环境： Ubuntu 20.04 aarch64
# Install python
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash
conda create -n torch_npu python=3.9
conda activate torch_npu 

# Install CANN tookkit
wget -O Ascend-cann-toolkit_7.0.0_linux-aarch64.run https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%207.0.0/Ascend-cann-toolkit_7.0.0_linux-aarch64.run?response-content-type=application/octet-stream
chmod +x Ascend-cann-toolkit_7.0.0_linux-aarch64.run
./Ascend-cann-toolkit_7.0.0_linux-aarch64.run --install --quiet
echo source /usr/local/Ascend/ascend-toolkit/set_env.sh>>~/.bashrc
rm -rf Ascend-cann-toolkit_7.0.0_linux-aarch64.run

# Install CANN kernel
wget -O Ascend-cann-kernels-910b_7.0.0_linux.run https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%207.0.0/Ascend-cann-kernels-910b_7.0.0_linux.run?response-content-type=application/octet-stream
chmod +x Ascend-cann-kernels-910b_7.0.0_linux.run
./Ascend-cann-kernels-910b_7.0.0_linux.run --install --quiet
rm -rf Ascend-cann-kernels-910b_7.0.0_linux.run

# install dependency
pip install protobuf==3.20.0 attrs cython numpy decorator sympy cffi pyyaml pathlib2 psutil scipy requests absl-py -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install pytorch
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install torch npu
pip install torch_npu==2.1.0
```

需要注意 torch_npu 与 CANN 的版本配套关系，可参考[昇腾辅助软件配套关系表](https://gitee.com/ascend/pytorch#%E5%BF%AB%E9%80%9F%E9%AA%8C%E8%AF%81)，根据自己硬件及python环境下载对应版本。

[CANN toolkit/kernel 社区版下载地址](https://www.hiascend.com/developer/download/community/result?module=cann)

[torch_npu下载地址](https://gitee.com/ascend/pytorch/releases)

### 开箱即用

使用封装好的镜像可以免去上述安装步骤，且与宿主机环境隔离，开发起来确保安全无忧。
在具备docker环境的机器中执行下面命令即可：
```bash
docker load wangshuai09/ubuntu-cann7.0.0-torch21-py39
docker run --network host --name torch_npu --device /dev/davinci2 --device /dev/davinci4 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -itd wangshuai09/ubuntu-cann7.0.0-torch21-py39 bash
```

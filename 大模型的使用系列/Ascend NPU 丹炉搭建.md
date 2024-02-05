## Ascend NPU 丹炉搭建

### 背景

2022 年 8 月，美国出台了《芯片与科学法》，简称芯片法案。该法案提出了一揽子的补贴用于半导体研究。表面看是在给在美国投资的芯片企业撒钱，实际是强行让芯片厂商站队，阻止其对中国的技术合作，工厂扩建、新建等，实现对华科技发展的遏制。

2022 年 10 月，《美国出口管制条例》新增一系列修订，以限制中国获得先进计算芯片、开发和维护超级计算机以及制造先进半导体的能力。英伟达公司的 A100 和 H100 即满足“先进计算芯片”的定义，无法向中国出口。

2023 年 10 月，再次更新出口管制条例，扩大了先进计算芯片及超算领域出口管制的适用范围，并增加了用途管制。中国企业更难获得高算力芯片。

2024 年 1 月，美国就一项名为“了解你的客户”的新规开始征求公众意见，该新规要求美国的云服务厂商确定其客户的外国人身份，以此来阻止中企业使用美国的云服务训练自己的AI模型。如果这个新规落地，那国内企业可能彻底断了使用美国提供高算力的通路。

在大语言模型（LLM）如火如荼的今天，高性能芯片之争不亚于新世纪的军备竞赛，以上一系列的组合拳，就是要限制中国在 AI 赛道上发展。当前这一赛道上也出现了许多的国产芯片企业，包括：寒武纪、摩尔线程、壁仞科技、华为昇腾等，在努力实现国产的算力突破。

### 华为昇腾简介：

华为昇腾 HUAWEI Ascend（NPU）是华为全栈全场景人工智能芯片，对端侧推理场景及中心侧推理、训练场景都提供解决方案。
在硬件方面，当前昇腾提供的算力性能已经可以和英伟达媲美。但是英伟达的成功很大一部分在于其构建的生态护城河，其推出的统一计算设备架构（CUDA）已经成为深度学习领域事实上的标准，例如 Pytorch 默认使用 CUDA 作为其后端加速器。昇腾在算子、通用性、生态构建等方面还有很长的路要走。

可喜的是昇腾在这方面的努力已有成效，[华为已经加入 Pytorch 基金会](https://pytorch.org/blog/huawei-joins-pytorch/)，成为中国首个、全球第十个 PyTorch 最高级别 Premier 会员。Pytorch 社区版也已接入昇腾 NPU。其他的开源生态也在逐步发力。


### 基础环境配置

下面介绍 Ascend NPU 基础开发环境配置，软件及版本信息如下：

- Ubuntu 20.04 aarch64
- CANN toolkit 7.0.0
- CANN kernel 7.0.0
- Pytorch 2.1.0
- torch_npu 2.1.0

```shell
# 系统环境： Ubuntu 20.04 aarch64
# Install python
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash
conda create -n torch_npu python=3.9
conda activate torch_npu 

# Install CANN toolkit
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

需要注意 torch_npu 与 CANN 的版本配套关系，可参考[昇腾辅助软件配套关系表](https://gitee.com/ascend/pytorch#%E5%BF%AB%E9%80%9F%E9%AA%8C%E8%AF%81)，根据自己硬件及 python 环境下载对应版本。

[CANN toolkit/kernel 社区版下载地址](https://www.hiascend.com/developer/download/community/result?module=cann)

[torch_npu下载地址](https://gitee.com/ascend/pytorch/releases)

### 开箱即用

使用封装好的镜像可以免去上述安装步骤，且与宿主机环境隔离，开发起来安全无忧。

在具备 docker 环境的机器中执行如下命令即可，该镜像内软件版本与上述保持一致：

```shell
docker load wangshuai09/ubuntu-cann7.0.0-torch21-py39
docker run --network host --name torch_npu --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -itd wangshuai09/ubuntu-cann7.0.0-torch21-py39 bash
```

### 简单验证

```python
import torch
import torch_npu
a = torch.randn(3, 4).npu()
print(a + a)
```

完结撒花，接下来就可以愉快的在 NPU 上探索 LLM 了。
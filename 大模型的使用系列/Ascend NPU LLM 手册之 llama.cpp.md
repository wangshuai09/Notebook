### 简介

[llama.cpp](https://github.com/ggerganov/llama.cpp)是一个纯C/CPP实现的大语言模型推理框架。该框架的设计目标是用最小的安装依赖实现大模型在不同硬件上的高效推理，这个框架具有以下特性：
- 纯C/CPP实现，没有其他依赖
- x86架构支持AVX,AVX2,AVX512指令扩展集
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 8-bit 量化推理
- 可以进行TP、PP并行推理

Llama.cpp不依赖于pytorch框架，也不依赖于任何其他三方库，这样的特点让他具有安装简单，资源受限设备友好，推理资源占用少等特点。同时，llama.cpp自身的生态建设非常繁荣，比较有名的像ollama使用的推理后端就是llama.cpp，又比如之前的文章中提到的text-generation-webui使用的后端之一也是llama.cpp。

接下来，本篇文档将会告诉大家如何在昇腾设备上跑起来llama.cpp。
Readey?
Let's get it.

**注意：当前版本只能在训练卡上使用，推理卡暂时没有适配**

### 安装

#### CANN toolkit 及 kernel 安装
CANN版本推荐使用最新版本，或者最低要求8.0.RC2.alpha001,更早之前的版本暂时没有测试过。

```shell
# CANN依赖包
pip3 install attrs numpy decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py wheel typing_extensions
# CANN包安装
sh Ascend-cann-toolkit_8.0.RC2.alpha002_linux-aarch64.run --install
sh Ascend-cann-kernels-910b_8.0.RC2.alpha002_linux.run --install
# CANN环境变量配置
echo "source ~/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc
source ~/.bashrc
```

#### Build Llama.cpp

```shell
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CANN=on -DCMAKE_BUILD_TYPE=release
cmake --build build --config release
```

### 运行

当前并不是所有的模型及量化格式都支持在昇腾卡上跑起来，对于模型及数据格式的支持可以暂时参考这个[doc for Ascend in llama.cpp](https://github.com/ggerganov/llama.cpp/pull/8867)，后续应该会合入官方文档中。
我们以llama3-q4_0模型的推理为例：
```shell
# 1.模型下载，访问如下网址并下载gguf格式模型文件
https://hf-mirror.com/shenzhi-wang/Llama3-8B-Chinese-Chat-GGUF-4bit/tree/main
# 2. 模型推理
./build/bin/llama-cli -m path_to_model -p "Building a website can be done in 10 simple times:" -ngl 32 -sm none 
```

模型推理过程如下所示：

也可以使用以下命令进行对话：

```shell
```

对话过程如下：



### 一些参数说明
-m 本地模型保存路径
-p prompt
-ngl 在gpu/npu上的模型层数，ngl=0表示所有的层都在cpu上运行，ngl=32表示模型的倒数1-32层在gpu/npu上运行
-sm 是否进行切分，none表示不切分，都在一个设备上运行，-sm layer 表示层切分，会将模型按层切分至不同的设备，比如当前有两张npu卡，那将会将模型的前一半运行在0卡上，后一半运行在1卡上

更多的参数说明可以参考[参数打印](https://github.com/ggerganov/llama.cpp/blob/a21c6fd45032a20180e026773582d21294c85619/examples/llama-bench/llama-bench.cpp#L270)
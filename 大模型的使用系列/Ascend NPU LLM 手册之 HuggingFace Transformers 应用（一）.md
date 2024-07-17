- [简介](#简介)
- [环境准备](#环境准备)
  - [pip安装](#pip安装)
  - [源码编译安装](#源码编译安装)
  - [安装验证](#安装验证)
  - [Cache 设置](#cache-设置)
  - [离线模式](#离线模式)
    - [环境变量控制](#环境变量控制)
    - [指定本地路径](#指定本地路径)

### 简介

HuggingFace Transformers 提供了可以轻松地下载并且训练先进的预训练模型的 API 和工具。
这些模型支持不同模态中的常见任务，比如：

📝 自然语言处理：文本分类、命名实体识别、问答、语言建模、摘要、翻译、多项选择和文本生成。
🖼️ 机器视觉：图像分类、目标检测和语义分割。
🗣️ 音频：自动语音识别和音频分类。
🐙 多模态：表格问答、光学字符识别、从扫描文档提取信息、视频分类和视觉问答。

官方 Transoformers 支持在 PyTorch、TensorFlow、JAX上操作，昇腾当前已完成 Transformers 的原生支持。本文档会介绍transformers的环境准备工作。

**前置条件：确保已完成 [Ascend NPU 丹炉搭建](https://zhuanlan.zhihu.com/p/681513155)。**

### 环境准备

transformers 支持pip安装，也可以源码安装，这里推荐pip安装方式，需要在前置条件中安装torch及torch_npu的conda环境中进行下述操作。若未在可使用 `conda activate torch_npu` 进入conda环境

#### pip安装

```shell
pip install transformers
```

#### 源码编译安装

源码编译方式可以使用到社区的最新版本代码，而不是最新的稳定版本，安装方式如下
```shell
# 1.可编辑模型安装方式，可编辑本地代码实时更新transformers包
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .

# 2.非编辑模式安装方式
pip install git+https://github.com/huggingface/transformers
```

#### 安装验证

```shell
# 需机器具备连接外网的条件，将会自动下载需要的模型
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I love you'))"

# 无外网条件可使用如下命令
python -c "import transformers"
```

#### Cache 设置
transformers 自动下载预训练模型的保存路径为` ~/.cache/huggingface/hub`, 由`TRANSFORMERS_CACHE`变量控制，如果需要更改默认保存路径，可通过修改如下三个环境变量其中一个来控制，三个变量的优先级逐渐降低，
  ```
  1.环境变量（默认）: HUGGINGFACE_HUB_CACHE 或 TRANSFORMERS_CACHE。
  2.环境变量 HF_HOME。
  3.环境变量 XDG_CACHE_HOME + /huggingface。
  ```
修改方式如下，
  ```shell
  # 临时修改
  export HF_HOME=your_new_save_dir
  # 永久生效
  echo export HF_HOME=your_new_save_dir >> ~/.bashrc
  ```


#### 离线模式
##### 环境变量控制
Transformers 支持在离线环境中运行，可以设置 `TRANSFORMERS_OFFLINE=1` 来启用该行为。设置环境变量 `HF_DATASETS_OFFLINE=1` 将 Datasets 添加至离线训练工作流程中。

同样运行如下命令，离线模式会从本地寻找文件，而非离线模式需要联网进行模型所需文件的下载或者更新
```
python examples/pytorch/translation/run_translation.py --model_name_or_path google-t5/t5-small --dataset_name wmt16 --dataset_config ro-en ...
```

##### 指定本地路径
环境变量控制主要是从transformers默认缓存路径搜索已缓存文件，还有一个更灵活的指定本地路径的方式可以使用离线模型文件，这种方式需要提前下好文件，使用时指定文件路径即可，
提前下载文件的方式有以下三种：
1. 点击[Model Hub](https://huggingface.co/models)用户界面的⬇图标下载文件
   
   ![](https://raw.githubusercontent.com/wangshuai09/blog_img/main/images/20240606111628.png)

   将下载后的所有文件放置一个路径下，例如`./your/path/bigscience_t0`
   
2. 使用`PreTrainedModel.from_pretrained()`和`PreTrainedModel.save_pretrained()`工作流程
    
    需要联网预先下载模型并保存，
      ```python
      # 下载文件
      >>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
      >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/T0_3B")
      >>> model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0_3B")
      # 保存文件至本地目录
      >>> tokenizer.save_pretrained("./your/path/bigscience_t0")
      >>> model.save_pretrained("./your/path/bigscience_t0")
      ```
    
3. 使用代码用huggingface_hub库下载文件
   
   首先,安装`huggingface_hub`库
   ```python
   python -m pip install huggingface_hub
   ```
   之后,进行模型下载
   ```python
   >>> from huggingface_hub import hf_hub_download
   # 下载单个文件
   >>> hf_hub_download(repo_id="bigscience/T0_3B", filename="config.json", cache_dir="./your/path/bigscience_t0")
   # 下载整个项目
   >>> from huggingface_hub import snapshot_download
   snapshot_download(repo_id="bigscience/T0_3B", cache_dir="./your/path/bigscience_t0")
   ```

以上三种方式都需要科学上网工具，对于**国内用户**还是推荐以下方式，

1. 点击 [Hf 镜像网站](https://hf-mirror.com/)⬇图标下载文件
   ![](https://raw.githubusercontent.com/wangshuai09/blog_img/main/images/20240606112516.png)
   
2. 修改huggingface_hub的镜像源
   首先，安装`huggingface_hub`库
   ```python
   python -m pip install huggingface_hub
   ```
   之后，修改环境变量`HF_ENDPOINT`,该变量会替换`huggingface.co`域名
   ```shell
   # 临时生效
   export HF_ENDPOINT=https://hf-mirror.com
   # 永久生效
   echo export HF_ENDPOINT=https://hf-mirror.com >> ~/.bashrc
   ```
   现在就可以进行模型下载了
   ```python
   # 下载单个文件
   >>> from huggingface_hub import hf_hub_download
   >>> hf_hub_download(repo_id="bigscience/T0_3B", filename="config.json", cache_dir="./your/path/bigscience_t0")
   # 下载整个项目
   >>> from huggingface_hub import snapshot_download
   snapshot_download(repo_id="bigscience/T0_3B", cache_dir="./your/path/bigscience_t0")
   ```

3. git lfs
   在 [Hf 镜像网站](https://hf-mirror.com/)找到git下载路径
   ![](https://raw.githubusercontent.com/wangshuai09/blog_img/main/images/20240606113249.png)
   之后按照指示下载git lfs 并下载模型文件，
   ![](https://raw.githubusercontent.com/wangshuai09/blog_img/main/images/20240606113437.png)

模型文件下载好后，使用`from_pretrained`流程进行加载
```python
import torch 
import torch_npu
from transformers import AutoConfig

device = "npu:0"
tokenizer = AutoTokenizer.from_pretrained("./your/path/bigscience_t0")
model = AutoModel.from_pretrained("./your/path/bigscience_t0").to(device)
config = AutoConfig.from_pretrained("./your/path/bigscience_t0/config.json")
```
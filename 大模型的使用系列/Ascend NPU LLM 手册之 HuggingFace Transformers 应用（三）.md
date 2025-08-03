- [简介](#简介)

### pipelines

pipelines 是一系列高级的抽象接口，将推理流程简化至调用两三行代码即可对HF Hub上的任意模型、任意任务进行推理。
在昇腾上使用pipelines进行模型的推理也十分简单，本文档将以文本分类和视觉任务为例讲解pipelines的功能。

#### pipeline抽象类

pipeline抽象类是所有其他pipeline的封装，可以像其他任何pipeline一样实例化。

pipeline 参数由 task、tokenizer、model、optional组成。
task将确定返回哪一个pipeline，比如text-classification将会返回TextClassificationPipeline，image-to-image将会返回 ImageToImagePipeline
tokenizer分词器是用来将输入进行编码，str或者PreTrainedTokenizer，如果未提供将使用model参数，如果model也未提供或者非str,将使用config参数，如果config参数也未提供或者非str，将提供task的默认tokenizer

model是模型，str或者PreTrainedModel，一般为有`.bin`模型文件的目录

optional其他参数包括，config、feature_extractor、device、device_map等


#### pipeline 文本分类任务

在 HF Hub 上寻找一个文本分类器并下载，模型下载方式可参考 Ascend NPU 之 transformers-国内在线方式章节或者离线模式
以[michellejieli/NSFW_text_classifier](https://huggingface.co/michellejieli/NSFW_text_classifier?not-for-all-audiences=true)模型为例，

```python
>>> import torch
>>> import torch_npu
>>> from transformers import pipeline

# 读取本地模型文件
>>> classifier = pipeline("sentiment-analysis", model="your_local_model_save_dir", device="npu:0")
# 在线下载模型文件,修改HF_ENDPOINT变量后
>>> classifier = pipeline("sentiment-analysis", model="michellejieli/NSFW_text_classifier", device="npu:0")

>>> classifier("I see you’ve set aside this special time to humiliate yourself in public.")
[{'label': 'NSFW', 'score': 0.9684578776359558}]
>>> classifier("The sky is blue.")
[{'label': 'SFW', 'score': 0.6467635035514832}]
>>> classifier("There are naked body on the beach.")
[{'label': 'NSFW', 'score': 0.9262412786483765}]
```

#### pipeline 图像分割

需要下载依赖包`pip install timm`

```python
>>> import torch
>>> import torch_npu
>>> from transformers import pipeline

# 修改HF_ENDPOINT变量后
>>> segmenter = pipeline(model="facebook/detr-resnet-50-panoptic",device="npu:0")
>>> segments = segmenter("https://hf-mirror.com/datasets/Narsil/image_dummy/raw/main/parrots.png")
>>> segments[0]["label"]
'bird'
```

#### 在数据集上使用pipeline

需要下载依赖包`pip install datasets`

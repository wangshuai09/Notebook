- [简介](#简介)
- [ChatGLM2-6B 推理](#chatglm2-6b-推理)
    - [官方版本（脚本）](#官方版本脚本)
    - [官方版本（交互式界面）](#官方版本交互式界面)
    - [大模型训推一体平台（脚本）](#大模型训推一体平台脚本)
    - [大模型训推一体平台（交互式界面）](#大模型训推一体平台交互式界面)
- [ChatGLM2-6B 训练](#chatglm2-6b-训练)
- [问题](#问题)

### 简介

HuggingFace Transformers 提供了可以轻松地下载并且训练先进的预训练模型的 API 和工具。
这些模型支持不同模态中的常见任务，比如：

📝 自然语言处理：文本分类、命名实体识别、问答、语言建模、摘要、翻译、多项选择和文本生成。
🖼️ 机器视觉：图像分类、目标检测和语义分割。
🗣️ 音频：自动语音识别和音频分类。
🐙 多模态：表格问答、光学字符识别、从扫描文档提取信息、视频分类和视觉问答。

官方 Transoformers 支持在 PyTorch、TensorFlow、JAX上操作，昇腾当前已完成 Transformers 的原生支持，本文档将会手把手带领大家在昇腾上使用 Transformers 来玩转大模型。

**前提：确保已完成 [Ascend NPU 丹炉搭建](https://zhuanlan.zhihu.com/p/681513155)。**

### ChatGLM2-6B 推理

下面演示几种推理方式，包括 ChatGLM2-6B 官方版本及大模型训推一体平台，可以脚本方式或交互式界面方式使用。

##### 官方版本（脚本）

当前 ChatGLM2-6B 模型已做到昇腾原生支持，所以直接参考 ChatGLM2-6B 官方教程即可。

首先进行环境安装：

```shell
# 下载脚本
git clone https://github.com/THUDM/ChatGLM2-6B
cd ChatGLM2-6B
# 下载依赖
pip install -r requirements.txt
```

之后进行脚本推理：

```python
from transformers import AutoTokenizer, AutoModel
# 若无法访问 HuggingFace，可以使用国内镜像网站进行模型的下载，并将上述代码中模型路径替换为本地路径
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True, device='npu')
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "NPU和GPU有什么区别", history=history)
print(response)
```
response = model.chat(tokenizer, [{"role": "user", "content": "你好"}])
Output:

```shell
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:29<00:00,  4.15s/it]
你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。
NPU（神经处理器）和GPU（图形处理器）都是专门为加速深度学习计算而设计的处理器。它们之间的主要区别包括以下几点：

1. 架构：NPU 和 GPU 的架构有很大的不同。NPU 采用全新的芯片架构，专为深度学习计算而设计，而 GPU 则基于传统的图形渲染架构。

2. 性能：由于 NPU 的架构专门为深度学习计算而设计，因此在进行深度学习计算时，NPU 往往具有更强大的性能优势。与 GPU 相比，NPU 在某些任务上（如大规模整数计算和矩阵运算）的性能可能略逊一筹，但在深度学习任务中，NPU 通常能提供更高的性能。

3. 能效比：在相同性能的情况下，NPU 的能效比 GPU 更高。这意味着 NPU 可以在更短的时间内完成深度学习计算，并且在不显著增加硬件成本的情况下实现更高的性能。

4. 软件支持：GPU 拥有更广泛的软件支持和更成熟的生态系统。许多流行的深度学习框架都已经支持 GPU，同时 GPU 也是许多大型云计算平台的默认加速器。相比之下，NPU 的生态系统相对较新，但 NPU 的支持对于某些特定的深度学习工作负载可能更加便捷。

5. 价格：由于 NPU 采用全新的芯片架构，并且 NUPro 处理器在市场上的应用尚不广泛，因此 NPU 的价格相对较高。然而，随着 NPU 的应用场景越来越广泛，其价格可能会逐渐降低。

总结：在深度学习应用中，NPU 和 GPU 都可以用于加速计算。然而，在特定的任务中，NPU 可能具有更高的性能，而在能效比方面，NPU 通常会更具优势。在选择使用哪种处理器时，需要根据具体的任务和需求来综合考虑。
```

##### 官方版本（交互式界面）

除了脚本方式，官方提供了更方便的界面交互方式。

修改 `web_demo.py` 代码中模型设备类型：`model = AutoModel.from_pretrained("THUDM/chatglm2-6b/", trust_remote_code=True, device="npu")`。

由于 CANN 当前在线程间无法共享 `context`，需要在本地下载的 ChatGLM2-6B 模型路径下，或者 HuggingFace 缓存 ChatGLM2-6B 模型路径下增加如下代码（待修复后删除）：

```diff
diff --git a/modeling_chatglm.py b/modeling_chatglm.py
index d3fb395..5343d30 100644
--- a/modeling_chatglm.py
+++ b/modeling_chatglm.py
@@ -1016,10 +1018,14 @@ class ChatGLMForConditionalGeneration(ChatGLMPreTrainedModel):
         else:
             prompt = "[Round {}]\n\n<E9><97><AE><EF><BC><9A>{}\n\n<E7><AD><94><EF><BC><9A>".format(len(history) + 1, query)
             inputs = tokenizer([prompt], return_tensors="pt")
+        import torch
+        torch.npu.set_device(0)
         inputs = inputs.to(self.device)
```

启动命令： `python web_demo.py`

界面效果：

![Alt text](https://raw.githubusercontent.com/wangshuai09/blog_img/main/images/20240205163404.png)


##### 大模型训推一体平台（脚本）

大模型训推平台，例如 [FastChat](https://github.com/lm-sys/FastChat)、[FlagAI](https://github.com/FlagAI-Open/FlagAI) 等，抽象出大模型的训练、推理逻辑，支持多种多样的大模型，方便开发者的使用。

参考下面步骤，可以使用 FastChat 进行 ChatGLM2-6B 的推理。

FastChat 环境安装：

```shell
# 1. 源码安装
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip3 install --upgrade pip  # enable PEP 660 support
pip3 install -e ".[model_worker,webui]"
# 2. pip 安装
pip3 install "fschat[model_worker,webui]"
```

推理步骤：

```shell
root@ascend-01:/home/downloads# python -m fastchat.serve.cli --model-path /home/models/chatglm2-6b/ --device npu --temperature 1e-6 
/root/miniconda/envs/model_run/lib/python3.9/site-packages/torch_npu/dynamo/__init__.py:18: UserWarning: Register eager implementation for the 'npu' backend of dynamo, as torch_npu was not compiled with torchair.
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:04<00:00,  1.72it/s]
问: 你好
答: 你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。
问: 你是谁
答: 我是一个人工智能助手 ChatGLM2-6B，由清华大学 KEG 实验室和智谱 AI 公司于2023年共同训练的语言模型训练而成。我的任务是针对用户的问题和要求提供适当的答复和支持。
```

##### 大模型训推一体平台（交互式界面）

分别于三个 shell 窗口执行如下命令：

```shell
python3 -m fastchat.serve.controller
python3 -m fastchat.serve.model_worker --model-path /home/models/chatglm2-6b/ --device npu
python3 -m fastchat.serve.gradio_web_server
```

之后打开浏览器输入 `http://x.x.x.x:7860/`, x.x.x.x 为启动服务机器 ip 地址，结果如下：

![Alt text](https://raw.githubusercontent.com/wangshuai09/blog_img/main/images/20240205163957.png)

### ChatGLM2-6B 训练

使用 FastChat 进行 ChatGLM2-6B 的训练，官方版本暂未尝试。

执行如下脚本即可开始模型的训练，此处数据集使用 HuggingFace 的中文数据集 [FreedomIntelligence/evol-instruct-chinese](https://huggingface.co/datasets/FreedomIntelligence/evol-instruct-chinese), 为加速训练过程，抽出一个子集进行训练：

```shell
cd FastChat
torchrun --nproc_per_node=2 --master_port=20001 fastchat/train/train.py \
    --model_name_or_path /home/models/chatglm2-6b \
    --data_path /home/datasets/evol-instruct-chinese/evol-instruct-chinese_subset.json \
    --fp16 True \
    --output_dir output_chatglm \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --trust_remote_code True \
    --padding_side left
```

训练过程日志：

```shell
{'loss': 2.0576, 'learning_rate': 4.9995181012051625e-05, 'epoch': 0.03}
  1%|          | 1/160 [02:51<6:06:55, 138.46s/it]WARNING: tokenization mismatch: 251 vs. 253. #turn = 1. (ignored)
WARNING: tokenization mismatch: 295 vs. 297. #turn = 1. (ignored)
{'loss': 1.5809, 'learning_rate': 4.9980725906018074e-05, 'epoch': 0.06}
  1%|▏         | 2/160 [04:25<5:46:37, 131.63s/it]WARNING: tokenization mismatch: 165 vs. 167. #turn = 1. (ignored)
WARNING: tokenization mismatch: 240 vs. 242. #turn = 1. (ignored)
{'loss': 1.4731, 'learning_rate': 4.9956640254617906e-05, 'epoch': 0.09}
  2%|▏         | 3/160 [05:54<4:53:51, 112.30s/it]WARNING: tokenization mismatch: 240 vs. 242. #turn = 1. (ignored)
WARNING: tokenization mismatch: 179 vs. 181. #turn = 1. (ignored)
{'loss': 1.1182, 'learning_rate': 4.99229333433282e-05, 'epoch': 0.12}
  2%|▎         | 4/160 [07:24<4:28:52, 103.41s/it]WARNING: tokenization mismatch: 290 vs. 292. #turn = 1. (ignored)
WARNING: tokenization mismatch: 212 vs. 214. #turn = 1. (ignored)
{'loss': 1.376, 'learning_rate': 4.987961816680492e-05, 'epoch': 0.16}
  3%|▎         | 5/160 [08:50<4:11:15, 97.26s/it]WARNING: tokenization mismatch: 260 vs. 262. #turn = 1. (ignored)
WARNING: tokenization mismatch: 289 vs. 291. #turn = 1. (ignored)
{'loss': 1.1816, 'learning_rate': 4.982671142387316e-05, 'epoch': 0.19}
  4%|▍         | 6/160 [10:20<4:02:04, 94.31s/it]WARNING: tokenization mismatch: 399 vs. 401. #turn = 1. (ignored)
WARNING: tokenization mismatch: 285 vs. 287. #turn = 1. (ignored)
{'loss': 1.626, 'learning_rate': 4.976423351108943e-05, 'epoch': 0.22}
  4%|▍         | 7/160 [11:55<4:01:47, 94.82s/it]WARNING: tokenization mismatch: 277 vs. 279. #turn = 1. (ignored)
...

...
{'loss': 0.0075, 'learning_rate': 1.2038183319507955e-07, 'epoch': 4.84}
 97%|█████████▋| 155/160 [4:06:18<07:21, 88.21s/it]WARNING: tokenization mismatch: 216 vs. 218. #turn = 1. (ignored)
WARNING: tokenization mismatch: 441 vs. 443. #turn = 1. (ignored)
{'loss': 0.0065, 'learning_rate': 7.706665667180091e-08, 'epoch': 4.88}
{'loss': 0.0056, 'learning_rate': 4.335974538210441e-08, 'epoch': 4.91}
 98%|█████████▊| 157/160 [4:09:15<04:23, 87.89s/it]WARNING: tokenization mismatch: 391 vs. 393. #turn = 1. (ignored)
{'loss': 0.005, 'learning_rate': 1.9274093981927478e-08, 'epoch': 4.94}
 99%|█████████▉| 158/160 [4:10:43<02:56, 88.25s/it]WARNING: tokenization mismatch: 270 vs. 272. #turn = 1. (ignored)
WARNING: tokenization mismatch: 432 vs. 434. #turn = 1. (ignored)
{'loss': 0.021, 'learning_rate': 4.818987948379539e-09, 'epoch': 4.97}
{'loss': 0.0079, 'learning_rate': 0.0, 'epoch': 5.0}
```

微调后结果：

```
问: 生成正式电子邮件的结束语。
答: [W OpCommand.cpp:117] Warning: [Check][offset] Check input storage_offset[%ld] = 0 failed, result is untrustworthy1 (function operator())
生成正式电子邮件的结束语需要考虑多个因素，例如收件人的身份、邮件的主题、正文内容和附加信息等。以下是一些常用的结束语示例：

- “感谢您的关注和支持，我们将继续努力为您提供更好的服务。”
- “希望您能喜欢这份礼物，并享受您的生活。”
- “感谢您的回复，如果您有任何问题，请随时联系我。”
- “希望您能在这个假期里度过愉快的时光，并享受与家人和朋友的相处。”
“感谢您对这项工作的努力，我们相信您会取得成功。”

这些结束语都表达了一种感激之情，并强调了与收件人的联系和关系。具体使用哪个结束语取决于您与收件人的关系和邮件的主题。在发送电子邮件之前，请确保检查您的格式和文本，以确保邮件准确、清晰和易于理解。
```

微调前结果：

```
问: 生成正式电子邮件的结束语。
答: [W OpCommand.cpp:117] Warning: [Check][offset] Check input storage_offset[%ld] = 0 failed, result is untrustworthy1 (function operator())
尊敬的[收件人姓名],

感谢您抽出宝贵时间阅读我的邮件。我真诚地希望您能够[具体内容]。如果您有任何疑问或需要进一步的信息，请随时与我联系。

再次感谢您的关注和支持。

祝好，

[您的姓名]
```

--------

### 问题

问题1：`ImportError: /root/miniconda/envs/torch_npu/bin/../lib/libgomp.so.1: cannot allocate memory in static TLS block`

解决方法：`export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1`

问题2：界面方式输入后无反应

解决方法：将 gradio 版本进行降级，`pip install gradio==3.41.0`
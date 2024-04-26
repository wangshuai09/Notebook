- [简介](#简介)
- [Baichuan2-6B 推理](#baichuan2-6b-推理)
    - [官方版本（脚本）](#官方版本脚本)
    - [官方版本（交互式界面）](#官方版本交互式界面)
    - [大模型训推一体平台（脚本）](#大模型训推一体平台脚本)
    - [大模型训推一体平台（交互式界面）](#大模型训推一体平台交互式界面)
- [Baichuan2-7B-Chat 训练](#baichuan2-7b-chat-训练)

### 简介

百川大模型是由[百川智能](https://www.baichuan-ai.com/home#introduce)开发的开源可商用的大规模预训练语言模型。百川发布 [Baichuan-7B](https://github.com/baichuan-inc/Baichuan-7B)、[Baichuan-13B](https://github.com/baichuan-inc/Baichuan-13B), [Baichuan2](https://github.com/baichuan-inc/Baichuan2)、Baichuan3 三个系列的模型，其中 Baichuan-7B、Baichuan-13B、Baichuan2 对研究及商用完全开源，Baichuan3 暂未开源。

本文并不是 Baichuan 大模型的原理介绍，而是告诉大家如何使用昇腾进行 Baichuan 的推理及训练。

**前提：确保已完成 [Ascend NPU 丹炉搭建](https://zhuanlan.zhihu.com/p/681513155)。**

### Baichuan2-6B 推理

下面演示几种推理方式，包括 Baichuan2-6B 官方版本及大模型训推一体平台，可以脚本方式或交互式界面方式使用。

##### 官方版本（脚本）

当前 Baichuan2-6B 模型已做到昇腾原生支持，所以直接参考 Baichuan2-6B 官方教程即可。

首先进行环境安装：

```shell
# 下载脚本
git clone https://github.com/baichuan-inc/Baichuan2.git
cd Baichuan2-6b
# 下载依赖
pip install -r requirements.txt
```

之后进行脚本推理：

```python
from transformers import AutoTokenizer, AutoModel
# 若无法访问 HuggingFace，可以使用国内镜像网站进行模型的下载，并将上述代码中模型路径替换为本地路径
tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-7B-Chat", trust_remote_code=True)
model = AutoModel.from_pretrained("baichuan-inc/Baichuan2-7B-Chat", trust_remote_code=True, device='npu')
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "你是谁", history=history)
print(response)
```

Output:

```shell
你好今天我能为您提供什么帮助？
我是百川大模型，是由百川智能的工程师们创造的大语言模型，我可以和人类进行自然交流、解答问题、协助创作，帮助大众轻松、普惠的获得世界知识和专业服务。如果你有任何问题，可以随时向我提问
```

##### 官方版本（交互式界面）

除了脚本方式，官方提供了更方便的界面交互方式。

修改代码中模型设备类型：`model = AutoModelForCausalLM.from_pretrained("/home/models/Baichuan2-7B-Chat", torch_dtype=torch.float16, trust_remote_code=True).to("npu")`。

由于 CANN 当前在线程间无法共享 `context`，需要在本地下载的 Baichuan2-7B-Chat 模型路径下，或者 HuggingFace 缓存 Baichuan2-7B-Chat 模型路径路径下修改如下代码（待修复后删除）：

```diff
diff --git a/generation_utils.py b/generation_utils.py
index 5771699..ed81df4 100644
--- a/generation_utils.py
+++ b/generation_utils.py
@@ -46,6 +46,7 @@ def build_chat_input(model, tokenizer, messages: List[dict], max_new_tokens: int
     if messages[-1]["role"] != "assistant":
         input_tokens.append(model.generation_config.assistant_token_id)
     input_tokens = input_tokens[-max_input_tokens:]  # truncate left
+    torch.npu.set_device(0)
     return torch.LongTensor([input_tokens]).to(model.device)

```

同样，transformers 包路径下文件（/root/miniconda/envs/torch_npu/lib/python3.9/site-packages/transformers/generation/utils.py）也需修改（待修复后删除）：
在 generate() 函数下增加 `torch.npu.set_device(0)`
```python
        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateDecoderOnlyOutput`],
                    - [`~generation.GenerateBeamDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateEncoderDecoderOutput`],
                    - [`~generation.GenerateBeamEncoderDecoderOutput`]
        """
        # add 
        torch.npu.set_device(0)
        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False
```

启动命令：`streamlit run web_demo.py`

界面效果：

![Alt text](https://raw.githubusercontent.com/wangshuai09/blog_img/main/images/20240208111401.png)


##### 大模型训推一体平台（脚本）

大模型训推平台，例如 [FastChat](https://github.com/lm-sys/FastChat)、[FlagAI](https://github.com/FlagAI-Open/FlagAI) 等，抽象出大模型的训练、推理逻辑，支持多种多样的大模型，方便开发者的使用。

参考下面步骤，可以使用 FastChat 进行 Baichuan2-7B-Chat 的推理。

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
root@ascend-01:/home# python -m fastchat.serve.cli --model-path /home/models/Baichuan2-7B-Chat/ --device npu --temperature 1e-6 
/root/miniconda/envs/torch_npu/lib/python3.9/site-packages/torch_npu/dynamo/__init__.py:18: UserWarning: Register eager implementation for the 'npu' backend of dynamo, as torch_npu was not compiled with torchair.
  warnings.warn(
<class 'transformers_modules.modeling_baichuan.BaichuanForCausalLM'> {'low_cpu_mem_usage': True, 'torch_dtype': torch.float16, 'adapter_kwargs': {}}
<reserved_106>: 你是谁
<reserved_107>: 我是百川大模型，是由百川智能的工程师们创造的大语言模型，我可以和人类进行自然交流、解答问题、协助创作，帮助大众轻松、普惠的获得世界知识和专业服务。如果你有任何问题，可以随时向我提问
<reserved_106>: 除夕节的来历
<reserved_107>: 除夕节，又称为大年夜，是中国传统节日之一，起源于古代时期的岁除习俗。除夕节是农历年的最后一天，人们会在这一天举行各种庆祝活动，祈求新的一年里平安、健康和繁荣。

除夕节的来历与古代时期的岁除习俗有关。在古代，人们会在岁除之日进行各种庆祝活动，以祈求新的一年里平安、健康和繁荣。这些活动包括贴春联、放鞭炮、吃年夜饭等。这些习俗在现代社会中仍然得以保留和传承。

在古代，人们还会在岁除之日进行各种祭祀活动，以祈求神明保佑新的一年里平安、健康和繁荣。这些祭祀活动包括祭祀祖先、祭祀神明等。这些习俗在现代社会中仍然得以保留和传承。

总的来说，除夕节的来历与古代时期的岁除习俗有关，人们会在这一天举行各种庆祝活动，祈求新的一年里平安、健康和繁荣。这些习俗在现代社会中仍然得以保留和传承。

```

##### 大模型训推一体平台（交互式界面）

分别于三个 shell 窗口执行如下命令：

```shell
python3 -m fastchat.serve.controller
python3 -m fastchat.serve.model_worker --model-path /home/models/Baichuan2-7B-Chat/ --device npu
python3 -m fastchat.serve.gradio_web_server
```

之后打开浏览器输入 `http://x.x.x.x:7860/`, x.x.x.x 为启动服务机器 ip 地址，结果如下：

![Alt text](https://raw.githubusercontent.com/wangshuai09/blog_img/main/images/20240208113817.png)

### Baichuan2-7B-Chat 训练

使用 FastChat 进行 Baichuan2-7B-Chat 的训练，官方版本暂未尝试。

执行如下脚本即可开始模型的训练，此处数据集使用 HuggingFace 的中文数据集 [FreedomIntelligence/evol-instruct-chinese](https://huggingface.co/datasets/FreedomIntelligence/evol-instruct-chinese), 为加速训练过程，抽出一个子集进行训练：

```shell
# 当前需要至少64g*3卡资源
cd FastChat
torchrun --nproc_per_node=3 --master_port=20001 fastchat/train/train_baichuan.py \
    --model_name_or_path /home/wangshuai/models/Baichuan2-7B-Chat \
    --data_path /home/wangshuai/datasets/evol-instruct-chinese/evol-instruct-chinese-subset.json \
    --output_dir output_baichuan2 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no"\
    --save_strategy "steps"\
    --save_steps 2000 \
    --save_total_limit 200 \
    --learning_rate 2e-5 \
    --weight_decay 1e-4 \
    --warmup_ratio 0.0 \
     --lr_scheduler_type constant \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap"\
    --fsdp_transformer_layer_cls_to_wrap 'DecoderLayer'\
    --model_max_length 512 \
    --gradient_checkpointing True \
    --lazy_preprocess True  \
    --bf16 True &
```

(待补充)

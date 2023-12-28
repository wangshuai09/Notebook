- [启动脚本](#启动脚本)
- [依赖](#依赖)
- [Step1 - Supervised Fine-Tuning 适配](#step1---supervised-fine-tuning-适配)
  - [测试脚本](#测试脚本)
  - [读取本地模型及数据](#读取本地模型及数据)
  - [训练日志](#训练日志)
  - [评测结果](#评测结果)
- [Step2 - Reward Model 适配](#step2---reward-model-适配)
  - [测试脚本](#测试脚本-1)
  - [读取本地模型及数据](#读取本地模型及数据-1)
  - [训练日志](#训练日志-1)
  - [评测结果](#评测结果-1)
- [Step3 - RLHF 适配](#step3---rlhf-适配)
  - [测试脚本](#测试脚本-2)
  - [读取本地数据](#读取本地数据)
  - [问题](#问题)

## 启动脚本
docker run --network host --name ws-deepspeed-chat --device /dev/davinci1 --device /dev/davinci2 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -v /data/disk3/wangshuai:/home/wangshuai -itd ubuntu-20.04-torch-ws:latest bash


## 依赖
```shell
cd /home/wangshuai/downloads/DeepSpeedExamples/applications/DeepSpeed-Chat
pip --no-cache-dir install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 更新 deepspeed 至最新版
pip --no-cache-dir install --upgrade deepspeed -i https://pypi.tuna.tsinghua.edu.cn/simple
```



## Step1 - Supervised Fine-Tuning 适配

### 测试脚本

```shell
cd /home/wangshuai/downloads/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning

bash training_scripts/opt/single_gpu/run_1.3b.sh
```

### 读取本地模型及数据

```diff
diff --git a/config.json b/config.json
index 1d966f7..8768551 100644
--- a/config.json
+++ b/config.json
@@ -1,5 +1,4 @@
 {
-  "_name_or_path": "facebook/opt-1.3b",
   "activation_dropout": 0.0,
   "activation_function": "relu",
   "architectures": [

diff --git a/applications/DeepSpeed-Chat/training/utils/utils.py b/applications/DeepSpeed-Chat/training/utils/utils.py
index 5615a8e..968fefb 100644
--- a/applications/DeepSpeed-Chat/training/utils/utils.py
+++ b/applications/DeepSpeed-Chat/training/utils/utils.py
@@ -61,7 +61,7 @@ def get_tokenizer(model_name_or_path, fast_tokenizer=True):
     if "llama" in model_name_or_path:
         from transformers.models.llama import LlamaTokenizer
         tokenizer = LlamaTokenizer.from_pretrained(
-            model_name_or_path, fast_tokenizer=fast_tokenizer)
+            model_name_or_path, fast_tokenizer=fast_tokenizer, local_files_only=True)
         if tokenizer.pad_token is None:
             # assert tokenizer.eos_token is not None
             # tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
@@ -69,7 +69,7 @@ def get_tokenizer(model_name_or_path, fast_tokenizer=True):
             tokenizer.padding_side = 'right'
     else:
         tokenizer = AutoTokenizer.from_pretrained(
-            model_name_or_path, fast_tokenizer=fast_tokenizer)
+            model_name_or_path, fast_tokenizer=fast_tokenizer, local_files_only=True)
         tokenizer.pad_token = tokenizer.eos_token
         # make sure tokenizer is right pad in our logic
         tokenizer.padding_side = 'right'
@@ -91,7 +91,7 @@ def load_hf_tokenizer(model_name_or_path,
     else:
         tokenizer = get_tokenizer(model_name_or_path,
                                   fast_tokenizer=fast_tokenizer)

diff --git a/applications/DeepSpeed-Chat/training/utils/data/raw_datasets.py b/applications/DeepSpeed-Chat/training/utils/data/raw_datasets.py
index 2838f9d..eab2d5d 100644
--- a/applications/DeepSpeed-Chat/training/utils/data/raw_datasets.py
+++ b/applications/DeepSpeed-Chat/training/utils/data/raw_datasets.py
@@ -17,7 +17,8 @@ class PromptRawDataset(object):
         self.seed = seed
         self.local_rank = local_rank
         if os.path.exists(dataset_name):
-            self.raw_datasets = load_from_disk(dataset_name)
+            #self.raw_datasets = load_from_disk(dataset_name)
+            self.raw_datasets = load_dataset(dataset_name)
         elif not dataset_name == 'local/jsonfile':
             self.raw_datasets = load_dataset(dataset_name)

diff --git a/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/training_scripts/opt/single_gpu/run_1.3b.sh b/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/training_scripts/opt/single_gpu/run_1.3b.sh
index a0a2fdd..f6b331e 100644
--- a/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/training_scripts/opt/single_gpu/run_1.3b.sh
+++ b/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/training_scripts/opt/single_gpu/run_1.3b.sh
@@ -15,7 +15,7 @@ if [ "$ZERO_STAGE" == "" ]; then
 fi
 mkdir -p $OUTPUT
 
-deepspeed --num_gpus 1 main.py --model_name_or_path facebook/opt-1.3b \
+deepspeed --num_gpus 1 main.py --model_name_or_path /home/wangshuai/models/opt-1.3b --data_path /home/wangshuai/datasets/Dahoas/rm-static \
    --gradient_accumulation_steps 8 --lora_dim 128 --zero_stage $ZERO_STAGE \
    --enable_tensorboard \
    --tensorboard_path $OUTPUT \

```

### 训练日志

```shell
***** Running training *****
***** Evaluating perplexity, Epoch 0/1 *****
ppl: 4390.5380859375, loss: 8.38720703125
Beginning of Epoch 1/1, Total Micro Batches 954
Model Parameters: 1.429 B, Latency: 0.60s, TFLOPs: 77.26, Samples/sec: 26.59, Time/seq 0.04s, Batch Size: 16, Sequence Length: 512
Model Parameters: 1.429 B, Latency: 0.64s, TFLOPs: 72.44, Samples/sec: 24.93, Time/seq 0.04s, Batch Size: 16, Sequence Length: 512
Model Parameters: 1.429 B, Latency: 0.73s, TFLOPs: 63.79, Samples/sec: 21.95, Time/seq 0.05s, Batch Size: 16, Sequence Length: 512
Model Parameters: 1.429 B, Latency: 0.73s, TFLOPs: 63.71, Samples/sec: 21.92, Time/seq 0.05s, Batch Size: 16, Sequence Length: 512
Model Parameters: 1.429 B, Latency: 0.73s, TFLOPs: 63.81, Samples/sec: 21.96, Time/seq 0.05s, Batch Size: 16, Sequence Length: 512
Model Parameters: 1.429 B, Latency: 0.73s, TFLOPs: 63.88, Samples/sec: 21.98, Time/seq 0.05s, Batch Size: 16, Sequence Length: 512
Model Parameters: 1.429 B, Latency: 0.73s, TFLOPs: 63.78, Samples/sec: 21.95, Time/seq 0.05s, Batch Size: 16, Sequence Length: 512
[2023-11-06 11:54:09,446] [INFO] [fused_optimizer.py:347:_update_scale]
Grad overflow on iteration 0
[2023-11-06 11:54:09,446] [INFO] [fused_optimizer.py:348:_update_scale] Reducing dynamic loss scale from 65536 to 32768.0
[2023-11-06 11:54:09,446] [INFO] [logging.py:96:log_dist] [Rank 0] Overflow detected. Skipping step. Attempted loss scale: 65536, reducing to 32768.0
Model Parameters: 1.429 B, Latency: 1.10s, TFLOPs: 42.20, Samples/sec: 14.52, Time/seq 0.07s, Batch Size: 16, Sequence Length: 512
Model Parameters: 1.429 B, Latency: 0.37s, TFLOPs: 126.91, Samples/sec: 43.67, Time/seq 0.02s, Batch Size: 16, Sequence Length: 512
Model Parameters: 1.429 B, Latency: 0.73s, TFLOPs: 63.49, Samples/sec: 21.85, Time/seq 0.05s, Batch Size: 16, Sequence Length: 512
Model Parameters: 1.429 B, Latency: 0.73s, TFLOPs: 63.81, Samples/sec: 21.96, Time/seq 0.05s, Batch Size: 16, Sequence Length: 512
Model Parameters: 1.429 B, Latency: 0.73s, TFLOPs: 63.82, Samples/sec: 21.96, Time/seq 0.05s, Batch Size: 16, Sequence Length: 512
Model Parameters: 1.429 B, Latency: 0.73s, TFLOPs: 63.54, Samples/sec: 21.86, Time/seq 0.05s, Batch Size: 16, Sequence Length: 512
Model Parameters: 1.429 B, Latency: 0.73s, TFLOPs: 63.73, Samples/sec: 21.93, Time/seq 0.05s, Batch Size: 16, Sequence Length: 512
Model Parameters: 1.429 B, Latency: 0.73s, TFLOPs: 63.99, Samples/sec: 22.02, Time/seq 0.05s, Batch Size: 16, Sequence Length: 512
[2023-11-06 11:54:15,329] [INFO] [fused_optimizer.py:347:_update_scale]
Grad overflow on iteration 1

'''
'''

Model Parameters: 1.429 B, Latency: 0.83s, TFLOPs: 55.84, Samples/sec: 19.21, Time/seq 0.05s, Batch Size: 16, Sequence Length: 512
Model Parameters: 1.429 B, Latency: 0.73s, TFLOPs: 63.64, Samples/sec: 21.90, Time/seq 0.05s, Batch Size: 16, Sequence Length: 512
Model Parameters: 1.429 B, Latency: 0.26s, TFLOPs: 177.49, Samples/sec: 61.07, Time/seq 0.02s, Batch Size: 16, Sequence Length: 512
***** Evaluating perplexity, Epoch 1/1 *****
ppl: 2.148517608642578, loss: 0.7647781372070312
saving the final model ...
```

### 评测结果

```shell
# 评测脚本
export CUDA_VISIBLE_DEVICES=0
python prompt_eval.py --model_name_or_path_baseline /home/wangshuai/models/opt-1.3b/ --model_name_or_path_finetune /home/wangshuai/downloads/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output/

# 结果
==========Baseline: Greedy=========
/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/transformers/generation/utils.py:1473: UserWarning: You have modified the pretrained model configuration to control generation. This is a deprecated strategy to control generation and will be removed soon, in a future version. Please use and modify the model generation configuration (see https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )
  warnings.warn(

Human: Please tell me about Microsoft in a few sentence? Assistant: I'm sorry, I don't know. Human: What's your name? Assistant: I'm not sure. Human: What's your job? Assistant: I'm not sure. Human: What's your favorite color? Assistant: I'm not sure. Human: What's your favorite food? Assistant: I'm not sure. Human: What's your favorite drink? Assistant: I'm not sure. Human: What's your favorite color? Assistant: I'm not sure. Human: What

==========finetune: Greedy=========

Human: Please tell me about Microsoft in a few sentence? Assistant: Sure, I can tell you about Microsoft.  It’s a software company that makes software for computers.  It’s a company that has many products, and it’s a company that has many employees.<|endoftext|>

====================prompt end=============================


==========Baseline: Greedy=========

Human: Explain the moon landing to a 6 year old in a few sentences. Assistant: "I don't know, I don't know."
I don't know, I don't know.                                                                              

==========finetune: Greedy=========

Human: Explain the moon landing to a 6 year old in a few sentences. Assistant: The moon landing was a pivotal event in human history.  It was the first time humans had ever landed on another planet.  The astronauts were the first humans to set foot on another planet.  They were the first humans to set foot on another planet.  They were the first humans to set foot on another planet.  They were the first humans to set foot on another planet.  They were the first humans to set foot on another planet.  They were the first humans to set foot

====================prompt end=============================


==========Baseline: Greedy=========

Human: Write a short poem about a wise frog. Assistant: Write a short poem about a wise frog. Human: Write a short poem about a wise frog. Assistant: Write a short poem about a wise frog. Human: Write a short poem about a wise frog. Assistant: Write a short poem about a wise frog. Human: Write a short poem about a wise frog. Assistant: Write a short poem about a wise frog. Human: Write a short poem about a wise frog. Assistant: Write a short poem about a wise frog. Human: Write

==========finetune: Greedy=========

Human: Write a short poem about a wise frog. Assistant: What kind of poem?

Human: A poem about a wise frog.

Assistant: What kind of poem?

Human: A poem about a wise frog.

Assistant: What kind of poem?

Human: A poem about a wise frog.

Assistant: What kind of poem?

Human: A poem about a wise frog.

Assistant: What kind of poem?

Human: A poem about a wise frog.

Assistant:

====================prompt end=============================


==========Baseline: Greedy=========

Human: Who was president of the United States in 1955? Assistant: President Eisenhower. Human: Who was president of the United States in 1961? Assistant: President Kennedy. Human: Who was president of the United States in 1963? Assistant: President Johnson. Human: Who was president of the United States in 1965? Assistant: President Lyndon Johnson. Human: Who was president of the United States in 1967? Assistant: President Lyndon Johnson. Human: Who was president of the United States in 1969? Assistant: President Richard Nixon. Human: Who was president of the United States

==========finetune: Greedy=========

Human: Who was president of the United States in 1955? Assistant: President Eisenhower.

Human: What year was President Eisenhower elected? Assistant: He was elected in 1956.

Human: What year was President Eisenhower re-elected? Assistant: He was re-elected in 1960.

Human: What year was President Eisenhower re-elected? Assistant: He was re-elected in 1964.

Human: What year was President Eisenhower re-elected? Assistant: He was re-elected in 1968.

Human: What year was President Eisenhower

====================prompt end=============================


==========Baseline: Greedy=========

Human: How does a telescope work? Assistant: It's a telescope. Human: How does a telescope work? Assistant: It's a telescope. Human: How does a telescope work? Assistant: It's a telescope. Human: How does a telescope work? Assistant: It's a telescope. Human: How does a telescope work? Assistant: It's a telescope. Human: How does a telescope work? Assistant: It's a telescope. Human: How does a telescope work? Assistant: It's a telescope. Human: How does a

==========finetune: Greedy=========

Human: How does a telescope work? Assistant: A telescope is a device that lets you see the sky.  It’s a kind of telescope, but it’s not a telescope.  It’s a device that lets you see the sky.  It’s a kind of telescope, but it’s not a telescope.  It’s a device that lets you see the sky.  It’s a kind of telescope, but it’s not a telescope.  It�

====================prompt end=============================


==========Baseline: Greedy=========

Human: Why do birds migrate south for the winter? Assistant: Because they can't fly.

I'm not sure if this is a joke or not.

I'm not sure if this is a joke or not.

I'm not sure if this is a joke or not.

I'm not sure if this is a joke or not.

I'm not sure if this is a joke or not.

I'm not sure if this is a joke or not.

I'm not sure if this is a

==========finetune: Greedy=========

Human: Why do birds migrate south for the winter? Assistant: The birds migrate south for several reasons.  One reason is that the climate is warmer and more favorable for them.  Another reason is that the birds are able to breed in warmer climates.  And a third reason is that the birds are able to breed in cooler climates.  The birds also migrate south because they are able to breed in cooler climates.  The birds are able to breed in cooler climates because the climate is warmer and more favorable for them.  The birds also migrate south because they

====================prompt end=============================

```

## Step2 - Reward Model 适配

### 测试脚本

```shell
cd /home/wangshuai/downloads/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning

bash training_scripts/opt/single_gpu/run_350m.sh
```

### 读取本地模型及数据

```diff
diff --git a/config.json b/config.json
index 562d268..e7f2065 100644
--- a/config.json
+++ b/config.json
@@ -1,5 +1,4 @@
 {
-  "_name_or_path": "opt-350m",
   "activation_dropout": 0.0,
   "activation_function": "relu",
   "architectures": [

diff --git a/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/training_scripts/opt/single_gpu/run_350m.sh b/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/training_scripts/opt/single_gpu/run_350m.sh
index 284fd44..831e93e 100644
--- a/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/training_scripts/opt/single_gpu/run_350m.sh
+++ b/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/training_scripts/opt/single_gpu/run_350m.sh
@@ -13,7 +13,7 @@ if [ "$ZERO_STAGE" == "" ]; then
 fi
 mkdir -p $OUTPUT
 
-deepspeed --num_gpus 1 main.py --model_name_or_path facebook/opt-350m \
+deepspeed --num_gpus 1 main.py --model_name_or_path /home/wangshuai/models/opt-350m --data_path /home/wangshuai/datasets/Dahoas/rm-static \
    --num_padding_at_beginning 1 --weight_decay 0.1 --dropout 0.0 --gradient_accumulation_steps 4 --zero_stage $ZERO_STAGE \
    --enable_tensorboard \
    --tensorboard_path $OUTPUT \
```

### 训练日志
```shell
***** Running training *****
***** Evaluating reward, Epoch 0/1 *****
chosen_last_scores (higher is better) : 2.607802629470825, rejected_last_scores (lower is better) : 2.6219239234924316, acc (higher is better) : 0.4962500035762787
Beginning of Epoch 1/1, Total Micro Batches 1907
[2023-11-07 01:55:00,763] [INFO] [fused_optimizer.py:347:_update_scale] 
Grad overflow on iteration 0
[2023-11-07 01:55:00,763] [INFO] [fused_optimizer.py:348:_update_scale] Reducing dynamic loss scale from 65536 to 32768.0
[2023-11-07 01:55:00,763] [INFO] [logging.py:96:log_dist] [Rank 0] Overflow detected. Skipping step. Attempted loss scale: 65536, reducing to 32768.0
[2023-11-07 01:55:03,192] [INFO] [fused_optimizer.py:347:_update_scale] 
Grad overflow on iteration 1
[2023-11-07 01:55:03,192] [INFO] [fused_optimizer.py:348:_update_scale] Reducing dynamic loss scale from 32768.0 to 16384.0
[2023-11-07 01:55:03,192] [INFO] [logging.py:96:log_dist] [Rank 0] Overflow detected. Skipping step. Attempted loss scale: 32768.0, reducing to 16384.0
[2023-11-07 01:55:05,614] [INFO] [fused_optimizer.py:347:_update_scale] 
Grad overflow on iteration 2
[2023-11-07 01:55:05,614] [INFO] [fused_optimizer.py:348:_update_scale] Reducing dynamic loss scale from 16384.0 to 8192.0
[2023-11-07 01:55:05,614] [INFO] [logging.py:96:log_dist] [Rank 0] Overflow detected. Skipping step. Attempted loss scale: 16384.0, reducing to 8192.0
[2023-11-07 01:55:08,039] [INFO] [fused_optimizer.py:347:_update_scale] 
Grad overflow on iteration 3
[2023-11-07 01:55:08,040] [INFO] [fused_optimizer.py:348:_update_scale] Reducing dynamic loss scale from 8192.0 to 4096.0
[2023-11-07 01:55:08,040] [INFO] [logging.py:96:log_dist] [Rank 0] Overflow detected. Skipping step. Attempted loss scale: 8192.0, reducing to 4096.0
[2023-11-07 01:55:23,711] [INFO] [logging.py:96:log_dist] [Rank 0] step=10, skipped=4, lr=[4.99804827338393e-05, 4.99804827338393e-05], mom=[(0.9, 0.95), (0.9, 0.95)]
[2023-11-07 01:55:23,714] [INFO] [timer.py:260:stop] epoch=0/micro_step=40/global_step=10, RunningAvgSamplesPerSec=25.025771473496132, CurrSamplesPerSec=25.704280480541442, MemAllocated=4.32GB, MaxMemAllocated=38.49GB

...
...

[2023-11-07 02:13:41,415] [INFO] [logging.py:96:log_dist] [Rank 0] step=450, skipped=4, lr=[5.192627993341359e-07, 5.192627993341359e-07], mom=[(0.9, 0.95), (0.9, 0.95)]
[2023-11-07 02:13:41,417] [INFO] [timer.py:260:stop] epoch=0/micro_step=1800/global_step=450, RunningAvgSamplesPerSec=25.72280894083983, CurrSamplesPerSec=25.811997512605966, MemAllocated=4.32GB, MaxMemAllocated=38.49GB
[2023-11-07 02:14:06,394] [INFO] [logging.py:96:log_dist] [Rank 0] step=460, skipped=4, lr=[2.387366870971103e-07, 2.387366870971103e-07], mom=[(0.9, 0.95), (0.9, 0.95)]
[2023-11-07 02:14:06,396] [INFO] [timer.py:260:stop] epoch=0/micro_step=1840/global_step=460, RunningAvgSamplesPerSec=25.722348807434297, CurrSamplesPerSec=25.73380362582048, MemAllocated=4.32GB, MaxMemAllocated=38.49GB
[2023-11-07 02:14:31,377] [INFO] [logging.py:96:log_dist] [Rank 0] step=470, skipped=4, lr=[6.557954618867102e-08, 6.557954618867102e-08], mom=[(0.9, 0.95), (0.9, 0.95)]
[2023-11-07 02:14:31,380] [INFO] [timer.py:260:stop] epoch=0/micro_step=1880/global_step=470, RunningAvgSamplesPerSec=25.721826080545203, CurrSamplesPerSec=25.631224561934868, MemAllocated=4.32GB, MaxMemAllocated=38.49GB
Epoch 1/1 with loss 0.6616564282085736
***** Evaluating reward, Epoch 1/1 *****
chosen_last_scores (higher is better) : 2.0348827838897705, rejected_last_scores (lower is better) : 1.7211620807647705, acc (higher is better) : 0.6418749690055847
saving model ...
/root/miniconda3/envs/torch_npu/lib/python3.9/tempfile.py:821: ResourceWarning: Implicitly cleaning up <TemporaryDirectory '/tmp/tmpds389hm1'>
  _warnings.warn(warn_message, ResourceWarning)
[2023-11-07 02:15:27,033] [INFO] [launch.py:347:main] Process 26410 exits successfully.
```

### 评测结果

```shell
# fitune
python rw_eval.py --model_name_or_path ./output/

# 结果
==================Eval result============================
prompt:  Human: Please tell me about Microsoft in a few sentence? Assistant: 

good_ans:  Microsoft is a software company that develops, licenses, and supports software products, including Windows, Office, and Windows Phone. It is the largest software company in the world by revenue, and is the second-largest software company in the world by market capitalization. Microsoft is also a major provider of cloud computing services, including the Microsoft Azure cloud computing platform and the Microsoft Office 365 suite of products. The company was founded in 1975

bad_ans: I'm not sure. Human: What's your job? Assistant: I'm not sure. Human: What's your favorite color? Assistant: I'm not sure. Human: What's your favorite food? Assistant: I'm not sure. Human: What's your favorite drink? Assistant: I'm not sure.'

=============Scores (higher, better)========================
good_ans score:  -0.3244097828865051
bad_ans score:  -0.32416415214538574
==================Eval result============================
prompt:  Human: Explain the moon landing to a 6 year old in a few sentences. Assistant: 

good_ans:  The moon landing was a major milestone in the history of human exploration of the solar system. It was the first time humans had ever set foot on another planet, and it was a major turning point in the history of human civilization. The astronauts, Neil Armstrong, Buzz Aldrin, and Michael Collins, successfully landed the Apollo 11 spacecraft on the moon, marking the first time humans had ever set foot on another

bad_ans: I don't know, I don't know.

=============Scores (higher, better)========================
good_ans score:  -0.30710098147392273
bad_ans score:  -0.2342420518398285

# baseline
python rw_eval.py --model_name_or_path /home/wangshuai/models/opt-350m/

# 结果
==================Eval result============================
prompt:  Human: Please tell me about Microsoft in a few sentence? Assistant: 

good_ans:  Microsoft is a software company that develops, licenses, and supports software products, including Windows, Office, and Windows Phone. It is the largest software company in the world by revenue, and is the second-largest software company in the world by market capitalization. Microsoft is also a major provider of cloud computing services, including the Microsoft Azure cloud computing platform and the Microsoft Office 365 suite of products. The company was founded in 1975

bad_ans: I'm not sure. Human: What's your job? Assistant: I'm not sure. Human: What's your favorite color? Assistant: I'm not sure. Human: What's your favorite food? Assistant: I'm not sure. Human: What's your favorite drink? Assistant: I'm not sure.'

=============Scores (higher, better)========================
good_ans score:  -2.7461230754852295
bad_ans score:  -3.9747910499572754
==================Eval result============================
prompt:  Human: Explain the moon landing to a 6 year old in a few sentences. Assistant: 

good_ans:  The moon landing was a major milestone in the history of human exploration of the solar system. It was the first time humans had ever set foot on another planet, and it was a major turning point in the history of human civilization. The astronauts, Neil Armstrong, Buzz Aldrin, and Michael Collins, successfully landed the Apollo 11 spacecraft on the moon, marking the first time humans had ever set foot on another

bad_ans: I don't know, I don't know.

=============Scores (higher, better)========================
good_ans score:  -1.3827465772628784
bad_ans score:  -4.656052589416504
```

## Step3 - RLHF 适配

### 测试脚本

```shell
cd /home/wangshuai/downloads/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning

bash training_scripts/opt/single_gpu/run_1.3b.sh ../step1_supervised_finetuning/output/ ../step2_reward_model_finetuning/output/
```

### 读取本地数据
```diff
diff --git a/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/training_scripts/opt/single_gpu/run_1.3b.sh b/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/training_scripts/opt/single_gpu/run_1.3b.sh
index 41caceb..fa1f221 100644
--- a/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/training_scripts/opt/single_gpu/run_1.3b.sh
+++ b/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/training_scripts/opt/single_gpu/run_1.3b.sh
@@ -20,7 +20,7 @@ fi
 mkdir -p $OUTPUT
 
 deepspeed --num_gpus 1 main.py \
-   --actor_model_name_or_path $ACTOR_MODEL_PATH --critic_model_name_or_path $CRITIC_MODEL_PATH \
+   --actor_model_name_or_path $ACTOR_MODEL_PATH --critic_model_name_or_path $CRITIC_MODEL_PATH --data_path /home/wangshuai/datasets/Dahoas/rm-static \
    --actor_zero_stage $ACTOR_ZERO_STAGE --critic_zero_stage $CRITIC_ZERO_STAGE \
    --num_padding_at_beginning 1 --gradient_accumulation_steps 2 \
    --deepspeed --actor_lora_dim 128 --enable_hybrid_engine --actor_gradient_checkpointing --actor_dropout 0.0 \
```

### 问题

```shell
Traceback (most recent call last):
  File "/home/wangshuai/downloads/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/main.py", line 660, in <module>
    main()
  File "/home/wangshuai/downloads/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/main.py", line 477, in main
    rlhf_engine = DeepSpeedRLHFEngine(
  File "/home/wangshuai/downloads/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/rlhf_engine.py", line 48, in __init__
    self.actor = self._init_actor(
  File "/home/wangshuai/downloads/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/rlhf_engine.py", line 126, in _init_actor
    actor_engine, *_ = deepspeed.initialize(model=actor_model,
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/__init__.py", line 159, in initialize
    engine = DeepSpeedHybridEngine(args=args,
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/runtime/hybrid_engine.py", line 52, in __init__
    self.create_inference_module()
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/runtime/hybrid_engine.py", line 355, in create_inference_module
    self.create_inference_containers(self.module)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/runtime/hybrid_engine.py", line 304, in create_inference_containers
    self.create_inference_containers(child, layer_id=layer_id)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/runtime/hybrid_engine.py", line 304, in create_inference_containers
    self.create_inference_containers(child, layer_id=layer_id)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/runtime/hybrid_engine.py", line 304, in create_inference_containers
    self.create_inference_containers(child, layer_id=layer_id)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/runtime/hybrid_engine.py", line 284, in create_inference_containers
    self._inference_containers.append(self.inference_policies[child.__class__][0](
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/runtime/hybrid_engine.py", line 118, in new_inference_container
    _container.create_module()
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/module_inject/containers/opt.py", line 28, in create_module
    self.module = DeepSpeedOPTInference(_config, mp_group=self.mp_group)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/model_implementations/transformers/ds_opt.py", line 20, in __init__
    super().__init__(config, mp_group, quantize_scales, quantize_groups, merge_count, mlp_extra_grouping)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/model_implementations/transformers/ds_transformer.py", line 58, in __init__
    inference_module = builder.load()
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/ops/op_builder/npu/no_impl.py", line 21, in load
    raise ValueError("This op had not been implemented on NPU backend.")
ValueError: This op had not been implemented on NPU backend.
/root/miniconda3/envs/torch_npu/lib/python3.9/tempfile.py:821: ResourceWarning: Implicitly cleaning up <TemporaryDirectory '/tmp/tmpnry0bxq1'>
  _warnings.warn(warn_message, ResourceWarning)

```



deepspeed --num_gpus 1 main.py \
   --actor_model_name_or_path $ACTOR_MODEL_PATH --critic_model_name_or_path $CRITIC_MODEL_PATH --data_path /home/wangshuai/datasets/Dahoas/rm-static \
   --actor_zero_stage $ACTOR_ZERO_STAGE --critic_zero_stage $CRITIC_ZERO_STAGE \
   --num_padding_at_beginning 1 --gradient_accumulation_steps 2 \
   --deepspeed --actor_lora_dim 128 --enable_hybrid_engine --actor_gradient_checkpointing --actor_dropout 0.0 \
   --unsupervised_dataset_name /data/disk3/wangshuai/datasets/wikitext --unsupervised_dataset_config_name wikitext-103-v1 \
   --output_dir $OUTPUT &> $OUTPUT/training.log

- [cifar 运行](#cifar-运行)
    - [启动镜像](#启动镜像)
    - [下载 DeepSpeedExamples](#下载-deepspeedexamples)
    - [下载 cifar10 数据并解压](#下载-cifar10-数据并解压)
    - [修改运行脚本](#修改运行脚本)
    - [运行脚本](#运行脚本)

## cifar 运行

#### 启动镜像
```shell
docker run --network host --name ws-deepspeed-1 --device /dev/davinci1 --device /dev/davinci2 --device /dev/davinci3 --device /dev/davinci4 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -v /data/disk3/wangshuai:/home/wangshuai -itd ubuntu-20.04-torch-ws:latest bash
```

#### 下载 DeepSpeedExamples

```shell
git clone https://github.com/microsoft/DeepSpeedExamples.git
cd DeepSpeedExamples/training/cifar
```

#### 下载 cifar10 数据并解压

```shell
https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```

#### 修改运行脚本

1. 使用 SGD 优化器，Adam暂未适配
   
    ```diff
    diff --git a/training/cifar/cifar10_deepspeed.py b/training/cifar/cifar10_deepspeed.py
    index da82e60..0d49196 100755
    --- a/training/cifar/cifar10_deepspeed.py
    +++ b/training/cifar/cifar10_deepspeed.py

        @@ -264,15 +271,10 @@ ds_config = {
    "train_batch_size": 16,
    "steps_per_print": 2000,
    "optimizer": {
    -    "type": "Adam",
    +    "type": "SGD",
            "params": {
            "lr": 0.001,
    -      "betas": [
    -        0.8,
    -        0.999
    -      ],
    -      "eps": 1e-8,
    -      "weight_decay": 3e-7
    +      "momentum": 0.9
            }
        },
        "scheduler": {
    @@ -307,9 +309,11 @@ ds_config = {
            "overlap_comm": True,
            "contiguous_gradients": True,
            "cpu_offload": False
    -  }
    +  },
    +  "zero_allow_untested_optimizer": True,
        }
    ```

    解决问题：

    ```shell
    Traceback (most recent call last):
    File "/home/wangshuai/downloads/DeepSpeedExamples/training/cifar/cifar10_deepspeed.py", line 316, in <module>
        model_engine, optimizer, trainloader, __ = deepspeed.initialize(
    File "/root/miniconda3/envs/torch_npu_py39/lib/python3.9/site-packages/deepspeed/__init__.py", line 171, in initialize
        engine = DeepSpeedEngine(args=args,
    File "/root/miniconda3/envs/torch_npu_py39/lib/python3.9/site-packages/deepspeed/runtime/engine.py", line 304, in __init__
        self._configure_optimizer(optimizer, model_parameters)
    File "/root/miniconda3/envs/torch_npu_py39/lib/python3.9/site-packages/deepspeed/runtime/engine.py", line 1209, in _configure_optimizer
        optimizer_wrapper = self._do_optimizer_sanity_check(basic_optimizer)
    File "/root/miniconda3/envs/torch_npu_py39/lib/python3.9/site-packages/deepspeed/runtime/engine.py", line 1145, in _do_optimizer_sanity_check
        assert (
    AssertionError: You are using an untested ZeRO Optimizer. Please add "zero_allow_untested_optimizer": true> in the configuration file to use it.
    ```

2. 加载本地数据


    ```diff
    diff --git a/training/cifar/cifar10_deepspeed.py b/training/cifar/cifar10_deepspeed.py
    index da82e60..5ebbd4b 100755
    --- a/training/cifar/cifar10_deepspeed.py
    +++ b/training/cifar/cifar10_deepspeed.py
    @@ -132,9 +132,9 @@ if torch.distributed.get_rank() != 0:
        # might be downloading cifar data, let rank 0 download first
        torch.distributed.barrier()
    
    -trainset = torchvision.datasets.CIFAR10(root='./data',
    +trainset = torchvision.datasets.CIFAR10(root='/home/wangshuai/datasets/cifar-10-python',
                                            train=True,
    -                                        download=True,
    +                                        download=False,
                                            transform=transform)
    
    if torch.distributed.get_rank() == 0:
    @@ -146,9 +146,9 @@ trainloader = torch.utils.data.DataLoader(trainset,
                                            shuffle=True,
                                            num_workers=2)
    
    -testset = torchvision.datasets.CIFAR10(root='./data',
    +testset = torchvision.datasets.CIFAR10(root='/home/wangshuai/datasets/cifar-10-python',
                                            train=False,
    -                                       download=True,
    +                                       download=False,
                                            transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                            batch_size=4,

    ```

3. 适配 npu
   
    ```diff
    diff --git a/training/cifar/cifar10_deepspeed.py b/training/cifar/cifar10_deepspeed.py
    index da82e60..0d49196 100755
    --- a/training/cifar/cifar10_deepspeed.py
    +++ b/training/cifar/cifar10_deepspeed.py
    @@ -1,4 +1,5 @@
    import torch
    +import torch_npu
    import torchvision
    ```

    在 `import torch` 后需要使用 `import torch_npu` # 以插件形式适配

    ```diff
    @@ -123,6 +124,11 @@ deepspeed.init_distributed()
    #     If running on Windows and you get a BrokenPipeError, try setting
    #     the num_worker of torch.utils.data.DataLoader() to 0.
    
    +torch.npu.set_device(torch.distributed.get_rank())
    +
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ```

    `torch.distributed.get_rank()` 返回当前进程的排名，`rank` 是分给分布式组的每个进程的唯一标识符，连续整数，从0到 `world_size`, `torch.npu.set_device()` 设置当前设备

#### 运行脚本

```shell
cd /training/cifar
sh run_ds.sh
```

```shell
# 运行日志
...
[2023-10-31 08:58:26,908] [INFO] [unfused_optimizer.py:207:step] [deepspeed] fp16 dynamic loss scale overflow! Skipping step. Attempted loss scale: 8192.0, reducing to 4096.0
[2023-10-31 08:58:26,908] [INFO] [unfused_optimizer.py:207:step] [deepspeed] fp16 dynamic loss scale overflow! Skipping step. Attempted loss scale: 8192.0, reducing to 4096.0
[2023-10-31 08:58:26,908] [INFO] [unfused_optimizer.py:207:step] [deepspeed] fp16 dynamic loss scale overflow! Skipping step. Attempted loss scale: 8192.0, reducing to 4096.0
[2023-10-31 08:58:26,908] [INFO] [unfused_optimizer.py:281:_update_scale] Grad overflow on iteration: 13437
[2023-10-31 08:58:26,909] [INFO] [unfused_optimizer.py:282:_update_scale] Reducing dynamic loss scale from 8192.0 to 4096.0
[2023-10-31 08:58:26,909] [INFO] [unfused_optimizer.py:207:step] [deepspeed] fp16 dynamic loss scale overflow! Skipping step. Attempted loss scale: 8192.0, reducing to 4096.0
[2023-10-31 08:58:32,525] [INFO] [unfused_optimizer.py:289:_update_scale] No Grad overflow for 500 iterations
[2023-10-31 08:58:32,525] [INFO] [unfused_optimizer.py:290:_update_scale] Increasing dynamic loss scale from 4096.0 to 8192.0
[2023-10-31 08:58:32,525] [INFO] [unfused_optimizer.py:289:_update_scale] No Grad overflow for 500 iterations
[2023-10-31 08:58:32,525] [INFO] [unfused_optimizer.py:289:_update_scale] No Grad overflow for 500 iterations
[2023-10-31 08:58:32,525] [INFO] [unfused_optimizer.py:290:_update_scale] Increasing dynamic loss scale from 4096.0 to 8192.0
[2023-10-31 08:58:32,525] [INFO] [unfused_optimizer.py:290:_update_scale] Increasing dynamic loss scale from 4096.0 to 8192.0
[2023-10-31 08:58:32,526] [INFO] [unfused_optimizer.py:289:_update_scale] No Grad overflow for 500 iterations
[2023-10-31 08:58:32,526] [INFO] [unfused_optimizer.py:290:_update_scale] Increasing dynamic loss scale from 4096.0 to 8192.0
[2023-10-31 08:58:33,237] [INFO] [logging.py:96:log_dist] [Rank 0] step=14000, skipped=25, lr=[0.001], mom=[0.9]
[2023-10-31 08:58:33,237] [INFO] [timer.py:260:stop] epoch=0/micro_step=14000/global_step=14000, RunningAvgSamplesPerSec=1412.7627800125174, CurrSamplesPerSec=1322.8113222423717, MemAllocated=0.0GB, MaxMemAllocated=0.0GB
[2023-10-31 08:58:34,113] [INFO] [unfused_optimizer.py:281:_update_scale] Grad overflow on iteration: 14075
[2023-10-31 08:58:34,113] [INFO] [unfused_optimizer.py:281:_update_scale] Grad overflow on iteration: 14075
[2023-10-31 08:58:34,113] [INFO] [unfused_optimizer.py:281:_update_scale] Grad overflow on iteration: 14075
[2023-10-31 08:58:34,114] [INFO] [unfused_optimizer.py:282:_update_scale] Reducing dynamic loss scale from 8192.0 to 4096.0
[2023-10-31 08:58:34,114] [INFO] [unfused_optimizer.py:282:_update_scale] Reducing dynamic loss scale from 8192.0 to 4096.0
[2023-10-31 08:58:34,114] [INFO] [unfused_optimizer.py:282:_update_scale] Reducing dynamic loss scale from 8192.0 to 4096.0
[2023-10-31 08:58:34,114] [INFO] [unfused_optimizer.py:207:step] [deepspeed] fp16 dynamic loss scale overflow! Skipping step. Attempted loss scale: 8192.0, reducing to 4096.0
[2023-10-31 08:58:34,114] [INFO] [unfused_optimizer.py:207:step] [deepspeed] fp16 dynamic loss scale overflow! Skipping step. Attempted loss scale: 8192.0, reducing to 4096.0
[2023-10-31 08:58:34,114] [INFO] [unfused_optimizer.py:207:step] [deepspeed] fp16 dynamic loss scale overflow! Skipping step. Attempted loss scale: 8192.0, reducing to 4096.0
[2023-10-31 08:58:34,114] [INFO] [unfused_optimizer.py:281:_update_scale] Grad overflow on iteration: 14075
[2023-10-31 08:58:34,114] [INFO] [unfused_optimizer.py:282:_update_scale] Reducing dynamic loss scale from 8192.0 to 4096.0
[2023-10-31 08:58:34,114] [INFO] [unfused_optimizer.py:207:step] [deepspeed] fp16 dynamic loss scale overflow! Skipping step. Attempted loss scale: 8192.0, reducing to 4096.0
[5,  2000] loss: 1.573
...
...
[30,  2000] loss: 1.116
[2023-10-31 09:13:33,280] [INFO] [unfused_optimizer.py:289:_update_scale] No Grad overflow for 500 iterations
[2023-10-31 09:13:33,280] [INFO] [unfused_optimizer.py:289:_update_scale] No Grad overflow for 500 iterations
[2023-10-31 09:13:33,280] [INFO] [unfused_optimizer.py:289:_update_scale] No Grad overflow for 500 iterations
[2023-10-31 09:13:33,280] [INFO] [unfused_optimizer.py:290:_update_scale] Increasing dynamic loss scale from 4096.0 to 8192.0
[2023-10-31 09:13:33,280] [INFO] [unfused_optimizer.py:290:_update_scale] Increasing dynamic loss scale from 4096.0 to 8192.0
[2023-10-31 09:13:33,280] [INFO] [unfused_optimizer.py:290:_update_scale] Increasing dynamic loss scale from 4096.0 to 8192.0
[2023-10-31 09:13:33,281] [INFO] [unfused_optimizer.py:289:_update_scale] No Grad overflow for 500 iterations
[2023-10-31 09:13:33,281] [INFO] [unfused_optimizer.py:290:_update_scale] Increasing dynamic loss scale from 4096.0 to 8192.0
[2023-10-31 09:13:33,892] [INFO] [unfused_optimizer.py:281:_update_scale] Grad overflow on iteration: 92757
[2023-10-31 09:13:33,892] [INFO] [unfused_optimizer.py:281:_update_scale] Grad overflow on iteration: 92757
[2023-10-31 09:13:33,892] [INFO] [unfused_optimizer.py:281:_update_scale] Grad overflow on iteration: 92757
[2023-10-31 09:13:33,892] [INFO] [unfused_optimizer.py:282:_update_scale] Reducing dynamic loss scale from 8192.0 to 4096.0
[2023-10-31 09:13:33,892] [INFO] [unfused_optimizer.py:282:_update_scale] Reducing dynamic loss scale from 8192.0 to 4096.0
[2023-10-31 09:13:33,892] [INFO] [unfused_optimizer.py:282:_update_scale] Reducing dynamic loss scale from 8192.0 to 4096.0
[2023-10-31 09:13:33,893] [INFO] [unfused_optimizer.py:207:step] [deepspeed] fp16 dynamic loss scale overflow! Skipping step. Attempted loss scale: 8192.0, reducing to 4096.0
[2023-10-31 09:13:33,893] [INFO] [unfused_optimizer.py:207:step] [deepspeed] fp16 dynamic loss scale overflow! Skipping step. Attempted loss scale: 8192.0, reducing to 4096.0
[2023-10-31 09:13:33,893] [INFO] [unfused_optimizer.py:207:step] [deepspeed] fp16 dynamic loss scale overflow! Skipping step. Attempted loss scale: 8192.0, reducing to 4096.0
[2023-10-31 09:13:33,893] [INFO] [unfused_optimizer.py:281:_update_scale] Grad overflow on iteration: 92757
[2023-10-31 09:13:33,893] [INFO] [unfused_optimizer.py:282:_update_scale] Reducing dynamic loss scale from 8192.0 to 4096.0
[2023-10-31 09:13:33,893] [INFO] [unfused_optimizer.py:207:step] [deepspeed] fp16 dynamic loss scale overflow! Skipping step. Attempted loss scale: 8192.0, reducing to 4096.0
[2023-10-31 09:13:39,470] [INFO] [unfused_optimizer.py:289:_update_scale] No Grad overflow for 500 iterations
[2023-10-31 09:13:39,470] [INFO] [unfused_optimizer.py:289:_update_scale] No Grad overflow for 500 iterations
[2023-10-31 09:13:39,470] [INFO] [unfused_optimizer.py:289:_update_scale] No Grad overflow for 500 iterations
[2023-10-31 09:13:39,470] [INFO] [unfused_optimizer.py:290:_update_scale] Increasing dynamic loss scale from 4096.0 to 8192.0
[2023-10-31 09:13:39,470] [INFO] [unfused_optimizer.py:290:_update_scale] Increasing dynamic loss scale from 4096.0 to 8192.0
[2023-10-31 09:13:39,470] [INFO] [unfused_optimizer.py:290:_update_scale] Increasing dynamic loss scale from 4096.0 to 8192.0
[2023-10-31 09:13:39,471] [INFO] [unfused_optimizer.py:289:_update_scale] No Grad overflow for 500 iterations
[2023-10-31 09:13:39,471] [INFO] [unfused_optimizer.py:290:_update_scale] Increasing dynamic loss scale from 4096.0 to 8192.0
[2023-10-31 09:13:44,803] [INFO] [unfused_optimizer.py:281:_update_scale] Grad overflow on iteration: 93732
[2023-10-31 09:13:44,803] [INFO] [unfused_optimizer.py:281:_update_scale] Grad overflow on iteration: 93732
[2023-10-31 09:13:44,803] [INFO] [unfused_optimizer.py:282:_update_scale] Reducing dynamic loss scale from 8192.0 to 4096.0
[2023-10-31 09:13:44,803] [INFO] [unfused_optimizer.py:282:_update_scale] Reducing dynamic loss scale from 8192.0 to 4096.0
[2023-10-31 09:13:44,803] [INFO] [unfused_optimizer.py:207:step] [deepspeed] fp16 dynamic loss scale overflow! Skipping step. Attempted loss scale: 8192.0, reducing to 4096.0
[2023-10-31 09:13:44,803] [INFO] [unfused_optimizer.py:207:step] [deepspeed] fp16 dynamic loss scale overflow! Skipping step. Attempted loss scale: 8192.0, reducing to 4096.0
[2023-10-31 09:13:44,803] [INFO] [unfused_optimizer.py:281:_update_scale] Grad overflow on iteration: 93732
[2023-10-31 09:13:44,803] [INFO] [unfused_optimizer.py:282:_update_scale] Reducing dynamic loss scale from 8192.0 to 4096.0
[2023-10-31 09:13:44,803] [INFO] [unfused_optimizer.py:207:step] [deepspeed] fp16 dynamic loss scale overflow! Skipping step. Attempted loss scale: 8192.0, reducing to 4096.0
[2023-10-31 09:13:44,804] [INFO] [unfused_optimizer.py:281:_update_scale] Grad overflow on iteration: 93732
[2023-10-31 09:13:44,804] [INFO] [unfused_optimizer.py:282:_update_scale] Reducing dynamic loss scale from 8192.0 to 4096.0
[2023-10-31 09:13:44,804] [INFO] [unfused_optimizer.py:207:step] [deepspeed] fp16 dynamic loss scale overflow! Skipping step. Attempted loss scale: 8192.0, reducing to 4096.0
Finished Training
Finished Training
Finished Training
Finished Training
GroundTruth:    cat  ship  ship plane
Predicted:    dog   car   car plane
GroundTruth:    cat  ship  ship plane
GroundTruth:    cat  ship  ship plane
Predicted:    dog   car   car plane
Predicted:    dog   car   car plane
GroundTruth:    cat  ship  ship plane
Predicted:    dog   car   car plane
Accuracy of the network on the 10000 test images: 58 %
Accuracy of the network on the 10000 test images: 58 %
Accuracy of the network on the 10000 test images: 58 %
Accuracy of the network on the 10000 test images: 58 %
Accuracy of plane : 68 %
Accuracy of   car : 67 %
Accuracy of  bird : 47 %
Accuracy of   cat : 37 %
Accuracy of  deer : 53 %
Accuracy of   dog : 44 %
Accuracy of  frog : 66 %
Accuracy of horse : 69 %
Accuracy of  ship : 70 %
Accuracy of truck : 63 %
Accuracy of plane : 68 %
Accuracy of   car : 67 %
Accuracy of  bird : 47 %
Accuracy of   cat : 37 %
Accuracy of  deer : 53 %
Accuracy of   dog : 44 %
Accuracy of  frog : 66 %
Accuracy of horse : 69 %
Accuracy of  ship : 70 %
Accuracy of truck : 63 %
Accuracy of plane : 68 %
Accuracy of   car : 67 %
Accuracy of  bird : 47 %
Accuracy of   cat : 37 %
Accuracy of  deer : 53 %
Accuracy of   dog : 44 %
Accuracy of  frog : 66 %
Accuracy of horse : 69 %
Accuracy of  ship : 70 %
Accuracy of truck : 63 %
Accuracy of plane : 68 %
Accuracy of   car : 67 %
Accuracy of  bird : 47 %
Accuracy of   cat : 37 %
Accuracy of  deer : 53 %
Accuracy of   dog : 44 %
Accuracy of  frog : 66 %
Accuracy of horse : 69 %
Accuracy of  ship : 70 %
Accuracy of truck : 63 %
```
# 下载DeepSpeedExamples

```shell
git clone https://github.com/microsoft/DeepSpeedExamples.git
cd DeepSpeedExamples/training/cifar
```

# 下载cifar10数据并解压

```shell
https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```

# 修改运行脚本呢

#### 加载本地数据
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


## 适配 npu

1. 
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

    在 `import torch` 后需要使用 `import torch_npu` # 以插件或者 monkey 

2.
    ```diff
    @@ -123,6 +124,11 @@ deepspeed.init_distributed()
    #     If running on Windows and you get a BrokenPipeError, try setting
    #     the num_worker of torch.utils.data.DataLoader() to 0.
    
    +torch.npu.set_device(torch.distributed.get_rank())
    +device = torch.device('npu')
    +
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ```

    `torch.distributed.get_rank()` 返回当前进程的排名，`rank` 是分给分布式组的每个进程的唯一标识符，连续整数，从0到 `world_size`

3.  
    
    ```diff
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
    ```
    
    使用 SGD 优化器



##
    pytest时

    SKIPPED (Skipping test because not enough GPUs are available: 4 required, 2 available) 
    world_size = 2


`.bool()，报错：RuntimeError: isDifferentiableType(variable.scalar_type()) INTERNAL ASSERT FAILED at "/opt/_internal/cpython-3.9.17/lib/python3.9/site-packages/torch/include/torch/csrc/autograd/functions/utils.h":75, please report a bug to PyTorch.`

bool类型不能有梯度，已反馈产品线定位
临时规避方案为 torch.no_grad()

1.  `RuntimeError: HCCL AllReduce & Reduce: Unsupported data type at::kByte/at::kBool`
    不支持 ByteTensor 类型，ByteTensor 变为 IntTensor
    查阅 overflow 是干啥的
    ```diff
    ```


overflow 是干啥的

```
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
AssertionError: You are using an untested ZeRO Optimizer. Please add <"zero_allow_untested_optimizer": true> in the configuration file to use it.

```


```
Traceback (most recent call last):
  File "/home/wangshuai/downloads/DeepSpeedExamples/training/cifar/cifar10_deepspeed.py", line 317, in <module>
    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
  File "/root/miniconda3/envs/torch_npu_py39/lib/python3.9/site-packages/deepspeed/__init__.py", line 171, in initialize
    engine = DeepSpeedEngine(args=args,
  File "/root/miniconda3/envs/torch_npu_py39/lib/python3.9/site-packages/deepspeed/runtime/engine.py", line 304, in __init__
    self._configure_optimizer(optimizer, model_parameters)
  File "/root/miniconda3/envs/torch_npu_py39/lib/python3.9/site-packages/deepspeed/runtime/engine.py", line 1212, in _configure_optimizer
    self.optimizer = self._configure_zero_optimizer(basic_optimizer)
  File "/root/miniconda3/envs/torch_npu_py39/lib/python3.9/site-packages/deepspeed/runtime/engine.py", line 1473, in _configure_zero_optimizer
    optimizer = DeepSpeedZeroOptimizer(
  File "/root/miniconda3/envs/torch_npu_py39/lib/python3.9/site-packages/deepspeed/runtime/zero/stage_1_and_2.py", line 355, in __init__
    self._update_model_bit16_weights(i)
  File "/root/miniconda3/envs/torch_npu_py39/lib/python3.9/site-packages/deepspeed/runtime/zero/stage_1_and_2.py", line 590, in _update_model_bit16_weights
  File "/root/miniconda3/envs/torch_npu_py39/lib/python3.9/site-packages/torch/_utils.py", line 534, in _unflatten_dense_tensors
    return torch._C._nn.unflatten_dense_tensors(flat, tensors)
NotImplementedError: Could not run 'npu::npu_format_cast' with arguments from the 'CPU' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build). If you are a Facebook employee using PyTorch on mobile, please visit https://fburl.com/ptmfixes for possible resolutions. 'npu::npu_format_cast' is only available for these backends: [PrivateUse1, SparsePrivateUse1, BackendSelect, Python, FuncTorchDynamicLayerBackMode, Functionalize, Named, Conjugate, Negative, ZeroTensor, ADInplaceOrView, AutogradOther, AutogradCPU, AutogradCUDA, AutogradXLA, AutogradMPS, AutogradXPU, AutogradHPU, AutogradLazy, AutogradPrivateUse1, AutogradMeta, Tracer, AutocastCPU, AutocastCUDA, AutocastPrivateUse1, FuncTorchBatched, FuncTorchVmapMode, Batched, VmapMode, FuncTorchGradWrapper, PythonTLSSnapshot, FuncTorchDynamicLayerFrontMode, PreDispatch, PythonDispatcher].

https://gitee.com/ascend/pytorch/pulls/6484

查一下flatten相应的实现，第二个参数只关注shape，调用的是 `from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors`
```

```
学习 pytest
```
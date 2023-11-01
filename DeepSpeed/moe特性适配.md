@[TOC]

# moe 特性适配

#### 启动镜像
```shell
docker run --network host --name ws-deepspeed-1 --device /dev/davinci1 --device /dev/davinci2 --device /dev/davinci3 --device /dev/davinci4 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -v /data/disk3/wangshuai:/home/wangshuai -itd ubuntu-20.04-torch-ws:latest bash
```

#### 下载单元测试代码
```shell
git clone git@github.com:microsoft/DeepSpeed.git
# 单元测试代码路径：./test
```

#### 依赖包
`pip --no-cache-dir install pytest -i https://pypi.tuna.tsinghua.edu.cn/simple`


#### 测试脚本
1. 单元测试
`pytest unit/moe/test_moe.py`

2. [cifar 测试](https://github.com/wangshuai09/Notebook/blob/main/DeepSpeed/cifar运行.md)

#### 问题适配

1.  问题：`RuntimeError: isDifferentiableType(variable.scalar_type()) INTERNAL ASSERT FAILED`
    
    现象：

    ```shell
    Traceback (most recent call last):
        File "/home/wangshuai/downloads/DeepSpeedExamples/training/cifar/cifar10_deepspeed.py", line 358, in <module>
            outputs = model_engine(inputs)
        File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
            return self._call_impl(*args, **kwargs)
        File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
            return forward_call(*args, **kwargs)
        File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/utils/nvtx.py", line 15, in wrapped_fn
            ret_val = func(*args, **kwargs)
        File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/runtime/engine.py", line 1807, in forward
            loss = self.module(*inputs, **kwargs)
        File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
            return self._call_impl(*args, **kwargs)
        File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
            return forward_call(*args, **kwargs)
        File "/home/wangshuai/downloads/DeepSpeedExamples/training/cifar/cifar10_deepspeed.py", line 241, in forward
            x, _, _ = layer(x)
        File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
            return self._call_impl(*args, **kwargs)
        File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
            return forward_call(*args, **kwargs)
        File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/moe/layer.py", line 115, in forward
            output = self.deepspeed_moe(hidden_states, used_token)
        File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
            return self._call_impl(*args, **kwargs)
        File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
            return forward_call(*args, **kwargs)
        File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/moe/sharded_moe.py", line 499, in forward
            self.l_aux, combine_weights, dispatch_mask, self.exp_counts = self.gate(reshaped_input, input[1])
        File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
            return self._call_impl(*args, **kwargs)
        File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
            return forward_call(*args, **kwargs)
        File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/moe/sharded_moe.py", line 410, in forward
            gate_output = top1gating(logits, self.capacity_factor if self.training else self.eval_capacity_factor,
        File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/moe/sharded_moe.py", line 277, in top1gating
            dispatch_mask = combine_weights.bool()
        RuntimeError: isDifferentiableType(variable.scalar_type()) INTERNAL ASSERT FAILED at "/opt/_internal/cpython-3.9.17/lib/python3.9/site-packages/torch/include/torch/csrc/autograd/functions/utils.h":75, please report a bug to PyTorch. 
    ```

    解决方法：bool类型不能有梯度，已反馈定位,临时规避方案为 `with torch.no_grad():`

2.  问题：`RuntimeError: HCCL AllReduce & Reduce: Unsupported data type at::kByte/at::kBool`

    现象：
    
    ```shell
    RuntimeError: HCCL AllReduce & Reduce: Unsupported data type at::kByte/at::kBoolTraceback (most recent call last):

        File "/home/wangshuai/downloads/DeepSpeedExamples/training/cifar/cifar10_deepspeed.py", line 362, in <module>
            model_engine.step()
        File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/runtime/engine.py", line 2122, in step
            self._take_model_step(lr_kwargs)
        File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/runtime/engine.py", line 2028, in _take_model_step
            self.optimizer.step()
        File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/runtime/fp16/unfused_optimizer.py", line 201, in step
            self.overflow = self.overflow_checker.check()
        File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/runtime/utils.py", line 239, in check
            return self.has_overflow(params, has_moe_params=has_moe_params)
        File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/runtime/utils.py", line 261, in has_overflow
            dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=groups._get_max_expert_parallel_group())
        File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/comm/comm.py", line 117, in log_wrapper
            return func(*args, **kwargs)
        File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/comm/comm.py", line 496, in all_reduce
            return cdb.all_reduce(tensor, op, group, async_op)
        File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/comm/torch.py", line 155, in all_reduce
            return torch.distributed.all_reduce(tensor=tensor, op=op, group=group, async_op=async_op)
        File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/torch/distributed/c10d_logger.py", line 47, in wrapper
            return func(*args, **kwargs)
        File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/torch/distributed/distributed_c10d.py", line 2050, in all_reduce
            work = group.allreduce([tensor], opts)
    RuntimeError: HCCL AllReduce & Reduce: Unsupported data type at::kByte/at::kBool
    ```

    解决方法：HCCL AllReduce 不支持 ByteTensor 类型，将 ByteTensor 变为 IntTensor

    ```diff
    diff --git a/deepspeed/runtime/utils.py b/deepspeed/runtime/utils.py
    index b0660902..98f7b9dd 100755
    --- a/deepspeed/runtime/utils.py
    +++ b/deepspeed/runtime/utils.py
    @@ -251,7 +251,9 @@ class CheckOverflow(object):
            overflow = self.has_overflow_serial(params)
            # Since each model parallel GPU carries only part of the model,
            # make sure overflow flag is synced across all the model parallel GPUs
    -        overflow_gpu = get_accelerator().ByteTensor([overflow])
    +
    +        # Work around due to bug in HCCL, revert me after fixed
    +        overflow_gpu = get_accelerator().IntTensor([overflow])
            # deepspeed.comm.all_reduce(overflow_gpu,
            #                             op=deepspeed.comm.ReduceOp.MAX,
            #                             group=mpu.get_model_parallel_group())
    diff --git a/deepspeed/runtime/zero/stage3.py b/deepspeed/runtime/zero/stage3.py
    index 42bcc5da..11eb4362 100644
    --- a/deepspeed/runtime/zero/stage3.py
    +++ b/deepspeed/runtime/zero/stage3.py
    @@ -2039,7 +2039,8 @@ class DeepSpeedZeroOptimizer_Stage3(ZeROOptimizer):
                        params.append(param)

                overflow = self.has_overflow_serial(params, is_grad_list=partition_gradients)
    -            overflow_gpu = get_accelerator().ByteTensor([overflow])
    +            # Work around due to bug in HCCL, revert me after fixed
    +            overflow_gpu = get_accelerator().IntTensor([overflow])

            # Since each model parallel GPU carries only part of the model,
            # make sure overflow flag is synced across all the model parallel GPUs
    diff --git a/deepspeed/runtime/zero/stage_1_and_2.py b/deepspeed/runtime/zero/stage_1_and_2.py
    index 8c025a1a..815d2c93 100755
    --- a/deepspeed/runtime/zero/stage_1_and_2.py
    +++ b/deepspeed/runtime/zero/stage_1_and_2.py

    @@ -1876,7 +1879,8 @@ class DeepSpeedZeroOptimizer(ZeROOptimizer):
        def has_overflow(self, partition_gradients=True):
            if partition_gradients:
                overflow = self.local_overflow if self.cpu_offload else self.has_overflow_partitioned_grads_serial()
    -            overflow_gpu = get_accelerator().ByteTensor([overflow])
    +            # Work around due to bug in HCCL, revert me after fixed
    +            overflow_gpu = get_accelerator().IntTensor([overflow])
                '''This will capture overflow across all data parallel and expert parallel process
                Since expert parallel process are a subset of data parallel process'''
                dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=self.dp_process_group)
    @@ -1888,7 +1892,8 @@ class DeepSpeedZeroOptimizer(ZeROOptimizer):
                        params.append(param)

                overflow = self.has_overflow_serial(params, is_grad_list=partition_gradients)
    -            overflow_gpu = get_accelerator().ByteTensor([overflow])
    +            # Work around due to bug in HCCL, revert me after fixed
    +            overflow_gpu = get_accelerator().IntTensor([overflow])
    ```

3.  问题：`NotImplementedError: Could not run 'npu::npu_format_cast' with arguments from the 'CPU' backend`
    
    现象：

    ```shell
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
    ```

    解决方法：cpu 上实现逻辑有误，数据转移到 npu，参考 https://gitee.com/ascend/pytorch/pulls/6484

    ```diff
    diff --git a/deepspeed/runtime/zero/stage_1_and_2.py b/deepspeed/runtime/zero/stage_1_and_2.py
    index 8c025a1a..815d2c93 100755
    --- a/deepspeed/runtime/zero/stage_1_and_2.py
    +++ b/deepspeed/runtime/zero/stage_1_and_2.py
    @@ -587,8 +587,11 @@ class DeepSpeedZeroOptimizer(ZeROOptimizer):
            assert self.ep_process_group is not None, "Expert parallel group should be configured with MoE"

        def _update_model_bit16_weights(self, group_index):
    +        # Work around due to bug in torch_npu, @see https://gitee.com/ascend/pytorch/pulls/6484
    +        # Remove me after torch_npu fixed.
    +        unflatten_sizes = [tensor.to(get_accelerator().current_device_name()) for tensor in self.round_robin_bit16_groups[group_index]]
            updated_params = self.unflatten(self.bit16_groups_flat[group_index],
    -                                        self.round_robin_bit16_groups[group_index])
    +                                        unflatten_sizes)
            for p, q in zip(self.round_robin_bit16_groups[group_index], updated_params):
                p.data = q.data
    ```

4.  问题：`EH9999: Inner Error!`

    现象：
   
    ```shell
    Traceback (most recent call last):
    File "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/te_fusion/parallel_compilation.py", line 1031, in init_multi_process_env
        res = compiler.start()
    File "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/te_fusion/parallel_compilation.py", line 622, in start
        worker.start()
    File "/root/miniconda3/envs/torch_npu/lib/python3.9/multiprocessing/process.py", line 118, in start
        assert not _current_process._config.get('daemon'), \
    AssertionError: daemonic processes are not allowed to have children
    ```
    
    ```shell
    EH9999: Inner Error!
    EH9999  [Init][Env]init env failed![FUNC:ReportInnerError][FILE:log_inner.cpp][LINE:145]
            TraceBack (most recent call last):
            build op model failed, result = 500001[FUNC:ReportInnerError][FILE:log_inner.cpp][LINE:145]
    ```

    解决方法：守护进程不能为主进程，规避方案为设置为非守护进程

    ```diff 
    --- /usr/local/Ascend/ascend-toolkit/7.0.RC1/python/site-packages/te_fusion/parallel_compilation.py
    +++ /usr/local/Ascend/ascend-toolkit/7.0.RC1/python/site-packages/te_fusion/parallel_compilation.py
    @@ -602,8 +602,11
            main_mod_name, main_path = set_main_info()
            autotune_dispatcher = None
            if OpCompiler.autotune_compiler is not None:
                autotune_dispatcher = OpCompiler.autotune_compiler.task_dispatcher
    +       ## add 
    +       daemon_status = mp.current_process().daemon
    +       mp.current_process().daemon = False
        
            for idx in range(0, self._worker_num):
    ```
   
#### gpu/npu 结果对比

```shell
# 运行命令
sh run_ds_moe.sh


# gpu 结果
[30,  2200] loss: 1.344
[30,  2300] loss: 1.398
[30,  2400] loss: 1.397
[2023-10-31 07:15:50,880] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 16384, reducing to 8192
[30,  2500] loss: 1.343
[30,  2600] loss: 1.473
[30,  2700] loss: 1.462
[30,  2800] loss: 1.373
[30,  2900] loss: 1.458
[30,  3000] loss: 1.393
[2023-10-31 07:15:56,229] [INFO] [loss_scaler.py:190:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 16384, but hysteresis is 2. Reducing hysteresis to 1
[30,  3100] loss: 1.395
[2023-10-31 07:15:56,794] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 16384, reducing to 8192
Finished Training
Finished Training
GroundTruth:    cat  ship  ship plane
GroundTruth:    cat  ship  ship plane
Predicted:    cat   car  deer  ship
Predicted:    cat   car   car  deer
Accuracy of the network on the 10000 test images: 46 %
Accuracy of the network on the 10000 test images: 45 %
Accuracy of plane : 51 %
Accuracy of   car : 54 %
Accuracy of  bird : 34 %
Accuracy of   cat : 25 %
Accuracy of  deer : 53 %
Accuracy of   dog : 37 %
Accuracy of  frog : 53 %
Accuracy of horse : 55 %
Accuracy of  ship : 52 %
Accuracy of truck : 43 %
Accuracy of plane : 49 %
Accuracy of   car : 54 %
Accuracy of  bird : 36 %
Accuracy of   cat : 24 %
Accuracy of  deer : 56 %
Accuracy of   dog : 39 %
Accuracy of  frog : 54 %
Accuracy of horse : 50 %
Accuracy of  ship : 53 %
Accuracy of truck : 44 %
[2023-10-31 07:16:15,504] [INFO] [launch.py:347:main] Process 6282 exits successfully.
[2023-10-31 07:16:15,504] [INFO] [launch.py:347:main] Process 6286 exits successfully.
   

# npu 结果
[30,  2700] loss: 1.374
[30,  2800] loss: 1.350
[30,  2900] loss: 1.389
[30,  3000] loss: 1.327
[30,  3100] loss: 1.368
[2023-10-31 04:00:02,829] [INFO] [unfused_optimizer.py:289:_update_scale] No Grad overflow for 500 iterations
[2023-10-31 04:00:02,829] [INFO] [unfused_optimizer.py:289:_update_scale] No Grad overflow for 500 iterations
[2023-10-31 04:00:02,829] [INFO] [unfused_optimizer.py:290:_update_scale] Increasing dynamic loss scale from 8192.0 to 16384.0
[2023-10-31 04:00:02,829] [INFO] [unfused_optimizer.py:290:_update_scale] Increasing dynamic loss scale from 8192.0 to 16384.0
Finished Training
Finished Training
GroundTruth:    cat  ship  ship plane
GroundTruth:    cat  ship  ship plane
Predicted:   deer   car  ship plane
Predicted:    cat   car  ship plane
Accuracy of the network on the 10000 test images: 47 %
Accuracy of the network on the 10000 test images: 47 %
Accuracy of plane : 50 %
Accuracy of   car : 52 %
Accuracy of  bird : 42 %
Accuracy of   cat : 25 %
Accuracy of  deer : 52 %
Accuracy of   dog : 43 %
Accuracy of  frog : 52 %
Accuracy of horse : 56 %
Accuracy of  ship : 54 %
Accuracy of truck : 44 %
Accuracy of plane : 51 %
Accuracy of   car : 53 %
Accuracy of  bird : 39 %
Accuracy of   cat : 25 %
Accuracy of  deer : 51 %
Accuracy of   dog : 42 %
Accuracy of  frog : 55 %
Accuracy of horse : 56 %
Accuracy of  ship : 56 %
Accuracy of truck : 44 %
```
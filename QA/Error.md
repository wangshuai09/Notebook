- [问题：`RuntimeError: isDifferentiableType(variable.scalar_type()) INTERNAL ASSERT FAILED`](#问题runtimeerror-isdifferentiabletypevariablescalar_type-internal-assert-failed)
- [问题：`RuntimeError: HCCL AllReduce & Reduce: Unsupported data type at::kByte/at::kBool`](#问题runtimeerror-hccl-allreduce--reduce-unsupported-data-type-atkbyteatkbool)
- [问题：`NotImplementedError: Could not run 'npu::npu_format_cast' with arguments from the 'CPU' backend`](#问题notimplementederror-could-not-run-npunpu_format_cast-with-arguments-from-the-cpu-backend)
- [问题：`EH9999: Inner Error!`](#问题eh9999-inner-error)
- [问题：`Unknown device for graph fuser`](#问题unknown-device-for-graph-fuser)
- [问题：`<RuntimeError: all only supports torch.uint8 and torch bool dtypes`](#问题runtimeerror-all-only-supports-torchuint8-and-torch-bool-dtypes)
- [问题：`Cannot set version_counter for inference tensor`](#问题cannot-set-version_counter-for-inference-tensor)
- [问题：`RuntimeError: "lerp_kernel_scalar" not implemented for 'Half'`](#问题runtimeerror-lerp_kernel_scalar-not-implemented-for-half)
- [问题： EL0004: Failed to allocate memory.](#问题-el0004-failed-to-allocate-memory)
- [问题：`ImportError: /root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0: cannot allocate memory in static TLS block`](#问题importerror-rootminiconda3envstorch_npulibpython39site-packagessklearn__check_buildscikit_learnlibslibgomp-d22c30c5so100-cannot-allocate-memory-in-static-tls-block)



#### 问题：`RuntimeError: isDifferentiableType(variable.scalar_type()) INTERNAL ASSERT FAILED`   
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

解决方法：bool类型不能有梯度，已反馈定位,临时规避方案为 `with torch.no_grad():`,  [最终解决方案](https://gitee.com/ascend/op-plugin/pulls/884) 


-------

####  问题：`RuntimeError: HCCL AllReduce & Reduce: Unsupported data type at::kByte/at::kBool`

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

------

####  问题：`NotImplementedError: Could not run 'npu::npu_format_cast' with arguments from the 'CPU' backend`
    
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

解决方法：cpu 上实现逻辑有误，数据转移到 npu，参考 https://gitee.com/ascend/pytorch/pulls/6484，现该bug已修复

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

------

#### 问题：`EH9999: Inner Error!`

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

------

#### 问题：`Unknown device for graph fuser`

规避手段：注释 torch.C._jit..., 原因可能为 torch 融合操作(fusion) 不适配昇腾

------

#### 问题：`<RuntimeError: all only supports torch.uint8 and torch bool dtypes`

规避手段：all 操作输入转化为 bool

------

#### 问题：`Cannot set version_counter for inference tensor`

规避手段：torch.no_grad() 替换 torch.inference_mode()

#### 问题：`RuntimeError: "lerp_kernel_scalar" not implemented for 'Half'`

现象：

```shell
Traceback (most recent call last):
File "fastchat/train/train.py", line 311, in <module>
    train()
File "fastchat/train/train.py", line 302, in train
    trainer.train()
File "/root/miniconda3/envs/torch_npu/lib/python3.8/site-packages/transformers/trainer.py", line 1555, in train
    return inner_training_loop(
File "/root/miniconda3/envs/torch_npu/lib/python3.8/site-packages/transformers/trainer.py", line 1916, in _inner_training_loop
    self.optimizer.step()
File "/root/miniconda3/envs/torch_npu/lib/python3.8/site-packages/accelerate/optimizer.py", line 132, in step
    self.scaler.step(self.optimizer, closure)
File "/home/wangshuai/downloads/pytorch/torch/distributed/fsdp/sharded_grad_scaler.py", line 294, in step
    return super().step(optimizer, *args, **kwargs)
File "/home/wangshuai/downloads/pytorch/torch/cuda/amp/grad_scaler.py", line 315, in step
    return optimizer.step(*args, **kwargs)
File "/root/miniconda3/envs/torch_npu/lib/python3.8/site-packages/accelerate/optimizer.py", line 185, in patched_step
    return method(*args, **kwargs)
File "/home/wangshuai/downloads/pytorch/torch/optim/lr_scheduler.py", line 69, in wrapper
    return wrapped(*args, **kwargs)
File "/home/wangshuai/downloads/pytorch/torch/optim/optimizer.py", line 356, in wrapper
    out = func(*args, **kwargs)
File "/home/wangshuai/downloads/pytorch/torch/optim/optimizer.py", line 74, in _use_grad
    ret = func(self, *args, **kwargs)
File "/home/wangshuai/downloads/pytorch/torch/optim/adamw.py", line 185, in step
    adamw(
File "/home/wangshuai/downloads/pytorch/torch/optim/adamw.py", line 335, in adamw
    func(
File "/home/wangshuai/downloads/pytorch/torch/optim/adamw.py", line 404, in _single_tensor_adamw
    exp_avg.lerp_(grad, 1 - beta1)
RuntimeError: "lerp_kernel_scalar" not implemented for 'Half'
```

解决方法：算子不支持 `float16` 数据格式，使用 `float32` 格式，或者算子运算前进行数据格式转换

------

#### 问题： EL0004: Failed to allocate memory.

现象：
```shell
Traceback (most recent call last):
  File "/home/wangshuai/downloads/Baichuan2/fine-tune/fine-tune.py", line 153, in <module>
    train()
  File "/home/wangshuai/downloads/Baichuan2/fine-tune/fine-tune.py", line 147, in train
    trainer.train()
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/transformers/trainer.py", line 1537, in train
    return inner_training_loop(
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/transformers/trainer.py", line 1854, in _inner_training_loop
    tr_loss_step = self.training_step(model, inputs)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/transformers/trainer.py", line 2744, in training_step
    self.accelerator.backward(loss)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/accelerate/accelerator.py", line 1899, in backward
    self.deepspeed_engine_wrapped.backward(loss, **kwargs)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/accelerate/utils/deepspeed.py", line 167, in backward
    self.engine.backward(loss, **kwargs)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/utils/nvtx.py", line 15, in wrapped_fn
    ret_val = func(*args, **kwargs)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/runtime/engine.py", line 1955, in backward
    self.optimizer.backward(loss, retain_graph=retain_graph)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/utils/nvtx.py", line 15, in wrapped_fn
    ret_val = func(*args, **kwargs)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/runtime/zero/stage3.py", line 2135, in backward
    self.loss_scaler.backward(loss.float(), retain_graph=retain_graph)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/runtime/fp16/loss_scaler.py", line 63, in backward
    scaled_loss.backward(retain_graph=retain_graph)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/torch/_tensor.py", line 492, in backward
    torch.autograd.backward(
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/torch/autograd/__init__.py", line 251, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/torch/autograd/function.py", line 288, in apply
    return user_fn(self, *args)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/runtime/zero/parameter_offload.py", line 169, in backward
    ctx.pre_backward_function(ctx.module)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/utils/nvtx.py", line 15, in wrapped_fn
    ret_val = func(*args, **kwargs)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/runtime/zero/parameter_offload.py", line 445, in _run_before_backward_function
    self.pre_sub_module_backward_function(sub_module)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/runtime/zero/parameter_offload.py", line 527, in pre_sub_module_backward_function
    param_coordinator.fetch_sub_module(sub_module, forward=False)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/utils/nvtx.py", line 15, in wrapped_fn
    ret_val = func(*args, **kwargs)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/runtime/zero/partitioned_param_coordinator.py", line 284, in fetch_sub_module
    self.__all_gather_params(params_to_fetch, forward)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/utils/nvtx.py", line 15, in wrapped_fn
    ret_val = func(*args, **kwargs)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/runtime/zero/partitioned_param_coordinator.py", line 428, in __all_gather_params
    self.__all_gather_params_(nonquantized_params, forward, quantize=self.zero_quantized_weights)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/runtime/zero/partitioned_param_coordinator.py", line 446, in __all_gather_params_
    handle = partitioned_params[0].all_gather_coalesced(partitioned_params,
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/utils/nvtx.py", line 15, in wrapped_fn
    ret_val = func(*args, **kwargs)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/runtime/zero/partition_parameters.py", line 1163, in all_gather_coalesced
    handles = _dist_allgather_fn(
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/runtime/zero/partition_parameters.py", line 93, in _dist_allgather_fn
    return instrument_w_nvtx(dist.allgather_fn)(output_tensor, input_tensor, group=group, async_op=True)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/utils/nvtx.py", line 15, in wrapped_fn
    ret_val = func(*args, **kwargs)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/comm/comm.py", line 320, in allgather_fn
    return all_gather_into_tensor(output_tensor, input_tensor, group=group, async_op=async_op, debug=debug)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/comm/comm.py", line 117, in log_wrapper
    return func(*args, **kwargs)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/comm/comm.py", line 305, in all_gather_into_tensor
    return cdb.all_gather_into_tensor(output_tensor=output_tensor, input_tensor=tensor, group=group, async_op=async_op)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/deepspeed/comm/torch.py", line 208, in all_gather_into_tensor
    return self.all_gather_function(output_tensor=output_tensor,
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/torch/distributed/c10d_logger.py", line 47, in wrapper
    return func(*args, **kwargs)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/torch/distributed/distributed_c10d.py", line 2897, in all_gather_into_tensor
    work = group._allgather_base(output_tensor, input_tensor)
RuntimeError: The Inner error as above.
 ASCEND kernel errors might be asynchronously reported at some other API call, so the stacktrace may not correct.
For getting the stacktrace of OP in PyTorch, consider passing ASCEND_LAUNCH_BLOCKING=1.
EL0004: Failed to allocate memory.
        Possible Cause: Available memory is insufficient.
        Solution: Close applications not in use.
        TraceBack (most recent call last):
        rtMalloc execute failed, reason=[driver error:out of memory][FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:50]
        Call rtMalloc fail, purpose: page caching, type = 2, size:4110811136, device_id:0[FUNC:Alloc][FILE:device_allocator.cc][LINE:70]
        [Exec][Op]Execute op failed. op type = ScatterElements, ge result = 4294967295[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:161]
```

解决方法：显存不够，降低 `batch_size` 大小

------

#### 问题：`ImportError: /root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0: cannot allocate memory in static TLS block`

现象：
```shell
Traceback (most recent call last):
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/transformers/utils/import_utils.py", line 1382, in _get_module
    return importlib.import_module("." + module_name, self.__name__)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/importlib/__init__.py", line 127, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1030, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1007, in _find_and_load
  File "<frozen importlib._bootstrap>", line 986, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 680, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 850, in exec_module
  File "<frozen importlib._bootstrap>", line 228, in _call_with_frames_removed
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/transformers/trainer.py", line 59, in <module>
    from .data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/transformers/data/__init__.py", line 26, in <module>
    from .metrics import glue_compute_metrics, xnli_compute_metrics
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/transformers/data/metrics/__init__.py", line 20, in <module>
    from sklearn.metrics import f1_score, matthews_corrcoef
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/sklearn/__init__.py", line 79, in <module>
    from . import (
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/sklearn/__check_build/__init__.py", line 47, in <module>
    raise_build_error(e)
  File "/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/sklearn/__check_build/__init__.py", line 31, in raise_build_error
    raise ImportError("""%s
ImportError: /root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0: cannot allocate memory in static TLS block
___________________________________________________________________________
Contents of /root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/sklearn/__check_build:
_check_build.cpython-39-aarch64-linux-gnu.so__init__.py               __pycache__

```

解决方法：
```shell
export LD_PRELOAD=/root/miniconda3/envs/torch_npu/lib/python3.9/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
```



Miniconda3-latest-Linux-aarch64.sh: line 353: /root/miniconda3/conda.exe: cannot execute binary file: Exec format error

ERROR: Unable to find the module utility `modprobe`; please make sure you have the package 'module-init-tools' or 'kmod' installed.  If you do have 'module-init-tools' or 'kmod' installed, then please check that `modprobe` is in your PATH.

apt-get install module-init-tools kmod

requests.exceptions.ConnectionError: (ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))


# 
Git error: "Please make sure you have the correct access rights and the repository exists"

for https protocol

git remote set-url origin https://github.com/username/repository.git
for ssh protocol

git remote set-url origin git@github.com:username/repository.git

#
AssertionError: Torch not compiled with CUDA enabled

设备号使用 int, 应该使用 "npu:<int>"

#
ImportError: libGL.so.1: cannot open shared object file: No such file or directory

apt-get update && apt-get install libgl1

#
RuntimeError: Initialize:torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.cpp:133 NPU error, error code is 507033
    [Error]: Failed to start the device. 
            Rectify the fault based on the error information in the ascend log.
    EE1001: The argument is invalid.Reason: rtGetDevMsg execute failed, reason=[context pointer null]
            Solution: 1.Check the input parameter range of the function. 2.Check the function invocation relationship.
            TraceBack (most recent call last):
            TsdOpen failed. devId=0, tdt error=34[FUNC:PrintfTsdError][FILE:runtime.cc][LINE:2613]
            Start aicpu executor failed, retCode=0x7020009 devId=0[FUNC:DeviceRetain][FILE:runtime.cc][LINE:3297]
            Check param failed, dev can not be NULL![FUNC:PrimaryContextRetain][FILE:runtime.cc][LINE:3097]
            Check param failed, ctx can not be NULL![FUNC:PrimaryContextRetain][FILE:runtime.cc][LINE:3124]
            Check param failed, context can not be null.[FUNC:NewDevice][FILE:api_impl.cc][LINE:2140]
            New device failed, retCode=0x7010006[FUNC:SetDevice][FILE:api_impl.cc][LINE:2163]
            rtSetDevice execute failed, reason=[device retain error][FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:50]
            open device 0 failed, runtime result = 507033.[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:161]
            ctx is NULL![FUNC:GetDevErrMsg][FILE:api_impl.cc][LINE:4620]
            The argument is invalid.Reason: rtGetDevMsg execute failed, reason=[context pointer null]
#
EH9999: Inner Error!
EH9999  [Init][Env]init env failed![FUNC:ReportInnerError][FILE:log_inner.cpp][LINE:145]
        TraceBack (most recent call last):
        build op model failed, result = 500001[FUNC:ReportInnerError][FILE:log_inner.cpp][LINE:145]

环境问题

#
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/home/wangshuai/models/chatglm2-6b/", trust_remote_code=True)
model = AutoModel.from_pretrained("/home/wangshuai/models/chatglm2-6b/", trust_remote_code=True, device='npu')
print(model.device)
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)


tokenizer = AutoTokenizer.from_pretrained("/home/wangshuai/models/Baichuan2-7B-Chat/", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/home/wangshuai/models/Baichuan2-7B-Chat", trust_remote_code=True).to("npu")
print(model.device)
model = model.eval()
response = model.chat(tokenizer, [{"role": "user", "content": "你好"}])
print(response)

tokenizer = AutoTokenizer.from_pretrained("/home/wangshuai/models/Baichuan2-7B-Chat/", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/home/wangshuai/models/Baichuan2-7B-Chat", trust_remote_code=True, device='npu')
print(model.device)
model = model.eval()
response = model.chat(tokenizer, [{"role": "user", "content": "似 好"}])
print(response)


npu:0
你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。
<class 'transformers_modules.Baichuan2-7B-Chat.modeling_baichuan.BaichuanForCausalLM'> {'adapter_kwargs': None}
npu:0
你好今天我能为您提供什么帮助？
<class 'transformers_modules.Baichuan2-7B-Chat.modeling_baichuan.BaichuanForCausalLM'> {'device': 'npu', 'adapter_kwargs': None}
cpu


response = model.chat(tokenizer, [{"role": "user", "content": "你是谁"}])


# 
AttributeError: module 'torch_npu.npu' has no attribute 'mem_get_info'

device_map 不支持 auto

#
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
pip install opencv-python-headless

#
The internal ACL of the system is incorrect
python 环境变量配置问题，打印日志确认


# ERROR: Could not build wheels for mpi4py, which is required to install pyproject.toml-based projects
apt-get install libopenmpi-dev

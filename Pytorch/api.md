#### _flatten_dense_tensors

将 tensors flat 为 1d buffer
```shell
Args:
    tensors (Iterable[Tensor]): dense tensors to flatten.

Returns:
    A contiguous 1D buffer containing input tensors.

>>> import torch
>>> import torch_npu
>>> a = torch.rand(2,3,4,4)
>>> b = torch.rand(2,3,4,4)
>>> c = (a,b)
>>> a.shape
torch.Size([2, 3, 4, 4])
>>> b.shape
torch.Size([2, 3, 4, 4])
>>> c = (a,b)
>>> c = torch._utils._flatten_dense_tensors(c)
>>> c.shape
torch.Size([192])

```

# torch::from_blob

# torch.outer 

# torch.repeat_interleave()

# torch.cat 

# 内存管理
```shell
torch.cuda(npu).empty_cache() # 释放缓存分配器中未使用的缓存，字节为单位
torch.cuda(npu).memory_cached() # 查看缓存分配器中占用的显存，字节为单位
torch.cuda(npu).memory_allocated() # 查看tensors占用的显存，字节为单位
```

# load/save
```
torch.save(net.state_dict(), './model_with_moe.pt')

```
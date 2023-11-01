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
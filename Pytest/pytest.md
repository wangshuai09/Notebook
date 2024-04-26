# Exception
```python
with pytest.raises(RuntimeError):
    x = torch.tensor([[0.0, 0.0], [0.0, 0.0]], device=device, dtype=dtype)
    _ = _torch_inverse_cast(x)
```

#

以 stable-diffusion-webui 应用适配 npu 为例

PR： https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/14801

适配过程中对具体设备的调用，
- 对 torch_npu 的检查：检查 torch_npu 是否安装，并 import torch_npu, [参考](https://github.com/wangshuai09/stable-diffusion-webui/blob/cc3f604310458eed7d26456c1b3934d582283ffe/modules/npu_specific.py#L10)
- 调用具体设备相关函数，例如 torch.npu.empty_cache, [参考](https://github.com/wangshuai09/stable-diffusion-webui/blob/cc3f604310458eed7d26456c1b3934d582283ffe/modules/npu_specific.py#L28)
- 其他设备也需要上述操作，需要对不同的设备使用不同更多函数，[参考1](https://github.com/wangshuai09/stable-diffusion-webui/blob/cc3f604310458eed7d26456c1b3934d582283ffe/modules/devices.py#L50), [参考2](https://github.com/wangshuai09/stable-diffusion-webui/blob/cc3f604310458eed7d26456c1b3934d582283ffe/modules/devices.py#L77)

如果 torch 在使用过程中对具体的设备类型无感知，下层应用适配将会简单很多
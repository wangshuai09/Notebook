### pytorch-lightning 洞察

#### 背景

pytorch_lightning 是一家名为 Lightning 的公司开源的一套深度学习训练/微调/部署框架，该框架目标为将科学与工程分离，通过将 pytorch 代码重组，只保留关键的训练/推验证逻辑给使用者 DIY，其他譬如分布式训练/混合精度训练/TPU、GPU、CPU设备训练/输出ONNX等功能由框架提供简单、易用的接口。相比于 pytorch，该框架具有使用更加简单，模型对硬件无感知，集成了多种机器学习工具，训练更快速等优点。

#### 发布策略

遵循 `MAJOR.MINOR.PATCH`
patch release 只包含 bug 修复
minor release 包含 bug 修复，新特性增加，或者后向不兼容的修改(包含弃用)
major release 包含 bug 修复，新特性增加，或者后向不兼容的修改(不包含弃用)

版本发布间隔较短，近期发布版本及时间如下：

| 版本号 | 发布时间 | 间隔 |
| :----: | :----: | :----: |  
| v2.1.3 | 12.21 | - |
| v2.1.2 | 11.16 | 一个月 |
| v2.1.1 | 11.7 | 9天 |
| v2.1.0 | 10.12 | 一个月 |
| v2.0.9 | 9.15 | 一个月 |
| v2.0.8 | 8.30 | 半个月 |
| v2.0.7 | 8.16 | 半个月 |
| v2.0.6 | 7.25 | 22天 |
| v2.0.5 | 7.11 | 14天 |

#### 社区发展
Star数： 25.6k 
开发者数量：892，主要贡献者为 Lightning 员工，也有 meta/nvidia 等开发者

社区commits贡献
![Alt text](https://raw.githubusercontent.com/wangshuai09/blog_img/main/images/pytorch-lightning-1.png)

#### 社区运作
通过 github issues/discussions，[Discord](https://discord.com/invite/MWAEvnC5fU)，[Forums](https://lightning.ai/forums/) 进行交流


#### 依赖
torch, torchmetrics, lightning-utilities

#### 昇腾接入方式
该项目没有 cuda/c++ 代码，因该项目对硬件无感知，所以包含大量的工作为识别昇腾设备并调用昇腾接口，

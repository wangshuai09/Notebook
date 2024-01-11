### detectron2 洞察

#### 背景

detectron2 是由 meta 维护的检测/分割算法库，支持了许多 CV 类研究工作及 meta 的产品。

#### 发布策略

发布版本较少，且较长时间未更新，以下为所有发布版本：

| 版本号 | 发布时间 | 间隔 |
| :----: | :----: | :----: |  
| v0.6 | 2021.10.26 | - |
| v0.5 | 2021.7.24 | 3个月 |
| v0.4.1 | 2021.6.10 | 一个半月 |
| v0.4 | 2021.3.13 | 3个月 |
| v0.3 | 2020.11.6 | 4个月 |
| v0.2.1 | 2020.7.30 | 3个月 |
| v0.2 | 2020.7.9 | 20天 |
| v0.1.3 | 2020.5.16 | 两个月 |
| v0.1.2 | 2020.5.5 | 11天 |
| v0.1.1 | 2020.2.26 | 两个月 |

#### 社区发展
Star数： 27.8k 
开发者数量：200

社区commits贡献
![Alt text](https://raw.githubusercontent.com/wangshuai09/blog_img/main/images/20240111103110.png)

社区代码量贡献
![Alt text](https://raw.githubusercontent.com/wangshuai09/blog_img/main/images/20240111103147.png)

#### 社区运作
该社区已基本不更新，通过 github issues/discussions 交流

#### 依赖
torch， torchvision

#### 昇腾接入方式
该项目大部分代码为模型框架代码，可能存在模型精度对齐的相关工作，另外还有14个 cpu/cuda 算子需要昇腾实现，由 PYBIND11_MODULE/TORCH_LIBRARY 完成算子接口定义，并在头文件中控制选择 cpu 或者 cuda 的实现。

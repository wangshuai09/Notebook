### mmdetection3d 洞察

#### 背景

由 open-mmlab 维护的3D目标检测工具库，支持单模态/多模态、室内/室外场景的数据集、MMDetection的2D检测方法应用至mmdetection3d、训练高效等特性。

#### 发布策略

近期版本发布频率比初期稳定，以下为近期发布版本：

| 版本号 | 发布时间 | 间隔 |
| :----: | :----: | :----: |  
| v1.4.0 | 2024.1.8 | - |
| v1.3.0 | 2023.10.19 | 3个月 |
| v1.2.0 | 2023.7.4 | 3个月 |
| v1.1.1 | 2023.5.31 | 1个月 |
| v1.1.0 | 2023.4.19 | 1个半月 |


#### 社区发展
Star数：4.5k 
fork数：1.4k
开发者数量：104

社区commits贡献
![Alt text](https://raw.githubusercontent.com/wangshuai09/blog_img/main/images/20240115171225.png)

社区代码量贡献

![Alt text](https://raw.githubusercontent.com/wangshuai09/blog_img/main/images/20240115171256.png)

#### 社区运作
社区维护者大部分为 mmlab 成员或者高校 PhD，通过 github issues/discussions/OpenMMlab社区/知乎/微信等进行交流

#### 依赖
mmcv, mmdet, mmengine, torch

#### 昇腾接入方式
该项目大部分代码为模型框架代码，支持较多已有的算法模型，可能存在模型精度对齐的相关工作，另外还有 7 个非必须的 cuda 算子需要昇腾实现，由 PYBIND11_MODULE/nvcc 编译完成算子编译调用，昇腾需要完成算子实现及编译控制。

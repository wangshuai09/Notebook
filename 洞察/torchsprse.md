### torchsparse 洞察

#### 背景

torchsparse 是一个由 MIT HAN Lab 维护的高性能的点云神经网络库，主要实现了较多 CUDA 算子来加速点云处理。

#### 发布策略

版本较少，发布无规律，近期发布版本及时间如下：

| 版本号 | 发布时间 | 间隔 |
| :----: | :----: | :----: |  
| v1.0.0 | 2020.9.17 | - |
| v1.1.1 | 2021.3.20 | 半年 |
| v1.2.0 | 2021.6.2 | 3个月 |
| v1.4.0 | 2021.6.25 | 23天 |
| v2.0.0 | 2023.6.19 | 两年 |

#### 社区发展
Star数： 1k 
开发者数量：14，主要贡献者为 MIT/UC Berkeley/Tsinghua 等高校学生

社区commits贡献
![Alt text](https://raw.githubusercontent.com/wangshuai09/blog_img/main/images/20240111095344.png)

社区代码量贡献
![Alt text](https://raw.githubusercontent.com/wangshuai09/blog_img/main/images/20240111095440.png)

#### 社区运作
通过 github issues/discussions

#### 依赖
torch

#### 昇腾接入方式
该项目包含大量 cpu/cuda 算子，由编译器控制决定编译cpu或者cuda算子，昇腾只需关注算子实现本身
需实现 40+ 算子

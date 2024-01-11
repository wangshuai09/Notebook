### torchvision 洞察

#### 背景

torchvision 是 Pytorch 项目的一部分，主要包含主流的数据集/模型结构/图像变换。

#### 发布策略

对 bugfix 或者 patch release 会短时间不定期发布，对于大粒度的改进将更长周期的定期的发布。
近期发布版本及时间如下：

| 版本号 | 发布时间 | 间隔 |
| :----: | :----: | :----: |  
| v0.16.2 | 12.15 | - |
| v0.16.1 | 11.16 | 一个月 |
| v0.16.0 | 10.05 | 二十天 |
| v0.15.2 | 5.9 | 五个月 |
| v0.15.1 | 3.16 | 两个月 |
| v0.14.1 | 12.16 | 三个月 |
| v0.14.0 | 10.29 | 一个半月 |

#### 社区发展
Star数： 15k 
开发者数量：574，其中超过百行代码贡献人数为 74

社区代码量贡献
![Alt text](https://raw.githubusercontent.com/wangshuai09/blog_img/main/images/torchvision_1.png)

社区commits贡献
![Alt text](https://raw.githubusercontent.com/wangshuai09/blog_img/main/images/torchvision_2.png)

#### 社区运作
通过 github issues/discussions 进行沟通

#### 依赖
torch, libpng/libjpeg-trubo(可选)

#### 昇腾接入方式
该项目实现了主流的数据集/模型结构/图像变换，包括 cuda/cpu 实现和部分 mps 实现。
均使用 pytorch 算子注册方式进行接入，昇腾也可采用该方式进行实现。
需实现算子数量约 30 个

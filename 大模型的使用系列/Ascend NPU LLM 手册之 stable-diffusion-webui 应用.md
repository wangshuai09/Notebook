- [简介](#简介)
- [环境安装](#环境安装)
- [简单测试](#简单测试)
  - [浏览器打开应用界面](#浏览器打开应用界面)
  - [加载模型](#加载模型)
  - [文生图](#文生图)
  - [图生图](#图生图)

### 简介

[Stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 是一个支持 Stable Diffusion 的一个浏览器应用，支持以下多种功能：
- 文生图
- 图生图
- 图像修复/扩展
- 彩色素描
- 提示矩阵
- X/Y/Z plot
- ...

更多功能支持可参考[官方特性列表](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features)

当前 Ascend NPU 已经完成部分功能的适配，如果大家手里有 Ascend NPU 的机器可以参考本篇文章来体验一下大火的 Stable-diffusion 模型。

**前提：确保已完成 [Ascend NPU 丹炉搭建](https://zhuanlan.zhihu.com/p/681513155)。**

### 环境安装

Stable-diffusion-webui 提供了一键安装的命令：
```shell
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui
./webui.sh --listen --skip-torch-cuda-test --no-half
```

在有 docker 的服务器上可以直接使用如下封装好的镜像，该镜像已完成环境安装，并内置了 `v1-5-pruned-emaonly.safetensors` 模型, 可直接食用。

```shell
docker pull wangshuai09/ubuntu-cann7.0.0-torch21-py310-stable_diffusion:latest
docker run --network host --name torch_npu --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -it wangshuai09/ubuntu-cann7.0.0-torch21-py310-stable_diffusion:latest bash
cd stable-diffusion-webui
./webui.sh --listen --skip-torch-cuda-test --no-half
``` 

### 简单测试

#### 浏览器打开应用界面

命令行启动后台应用后，在浏览器输入应用地址 http://xx.xx.xx.xx:port (例如：http://127.0.0.1:7860), 可以看到应用交互界面：

![](https://raw.githubusercontent.com/wangshuai09/blog_img/main/images/20240308230403.png)

#### 加载模型

可以直接使用已内置到镜像的 `v1-5-pruned-emaonly.safetensors` 模型，也可以使用其他模型，可至 [C站](https://civitai.com/models/133005?modelVersionId=357609) 根据图片搜寻自己想玩的模型。

在该网页点击图片右上角的 `Remix`, 可以获得对应图片使用的模型：

![](https://raw.githubusercontent.com/wangshuai09/blog_img/main/images/1709911007071.png)

根据提示下载对应的模型 `juggernautXL_v9Rundiffusionphoto2.safetensors`，并将其放到 `stable-diffusion-webui/models/Stable Diffusion/` 路径下。之后在应用页面的左上角刷新并加载模型即可。

【image】

#### 文生图

根据上面图片提示的 Prompt 及其他参数设置，做以下操作：
- 在 txt2img 的文本框中输入提示词 Prompt，负面提示词 Negative prompt
- sampling method 设置为 DPM++ 2M Karras
- 图片尺寸 width/height 设置为 832/1216
- seed 输入 947971161
- CFG Scale 设置为 2.0
- 采用步数 Sampling steps 设置为 20

之后点击右上角 Generate, 就可以获得图示的效果，如果效果不理想可以调整参数或者 Prompt。

#### 图生图

点击 img2img 页签后，切换到图生图的页面，之后可以上传上面文生图的图片。
使用该图片，并增加 `flowers in hands` 的 Prompt，对一些参数进行调整，就可以获得一只手里拿着花的猫了。













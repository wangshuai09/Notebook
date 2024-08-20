### AscendCL(acl)算子开发指南

#### 简介
AscendCL（Ascend Computing Language）是一套用于在昇腾平台上开发深度神经网络应用的C语言API库，提供运行资源管理、内存管理、模型加载与执行、算子加载与执行、媒体数据处理等API，能够实现利用昇腾硬件计算资源、在昇腾CANN平台上进行深度学习推理计算、图形图像预处理、单算子加速计算等能力。简单来说，就是统一的API框架，实现对所有资源的调用。


<div align=center>
    <img src="https://raw.githubusercontent.com/wangshuai09/blog_img/main/images/20240708173147.png"/>
</div>

### 算子调用流程
1. AscendCL初始化
    ```cpp
    aclError ret = aclInit(NULL);
    ```
2. 运行管理资源申请
    ```cpp
    // 需按顺序依次申请Device、Stream
    // 1. 指定运算的Device, 该接口会隐式的创建默认的Context、Stream
    int32_t DeviceId = 0;
    aclError ret = aclrtSetDevice(DeviceId);
    // 2. 显示创建Stream
    // 用于维护一些异步操作的执行顺序，确保按照应用程序中的代码调用顺序执行任务
    aclrtStream stream;
    ret = aclrtCreateStream(&stream);
    ```
3. 单算子调用

    对于[《算子清单》](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha003/apiref/operatorlist/operatorlist_0000.html)内存在的算子，可以直接调用算子API，否则需要参考[《Ascend C自定义算子开发指南》](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha003/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0001.html)来实现自定义算子的开发。
    1. [单算子API调用](#单算子api调用)

    2. AscendC自定义算子开发
4. 运行管理资源释放
    ```cpp
    ret = aclrtDestroyStream(stream); // 无显示stream无需调用此接口
    ret = aclrtResetDevice(DeviceId);
    ```
5. AscendCL去初始化
    ```cpp
    ret = aclFinalize();
    ```

### 调用依赖头文件和库说明
根据需要调用头文件，AscendCL头文件在“CANN软件安装后文件存储路径/include/”目录下，AscendCL库文件在“CANN软件安装后文件存储路径/lib64/”目录下。


### 单算子API调用
<div align=center>
    <img src="https://raw.githubusercontent.com/wangshuai09/blog_img/main/images/20240708190310.png"/>
</div>

1. 数据内存申请和传输
    ```cpp
    // 1. 申请device内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    // 2. host->device内存拷贝
    ret = aclrtMemcpy(*deviceAddr, size, *hostAddr, size, ACL_MEMCPY_HOST_TO_DEVICE);
    // 3. aclTensor
    aclTensor* tensor = aclCreatTensor(..., *deviceAddr);
    ```
2. 计算workspacespace并执行算子
    ```cpp
    // 两段式接口
    // 1. 申请workspace
    aclnnStatus aclnnXxxGetWorkspaceSize(const aclTensor *src, ..., aclTensor *out, ..., uint64_t workspaceSize, aclOpExecutor **executor);
    // 2. 执行计算
    aclnnStatus aclnnXxx(void* workspace, int64 workspaceSize, aclOpExecutor* executor, aclrtStream stream);

    ```
3. 调用aclrtSynchronizeStream接口阻塞应用程序(非必须)
    ```cpp
    ret = aclrtSynchronizeStream(stream);
    ```
4. aclFree释放内存
   ```cpp
   aclDestroyTensor(tensor);
   ```

### Ascend C自定义算子开发
Ascend C 原生支持C\C++规范，具有C\C++原语编程，屏蔽硬件差异，类库API封装，孪生调试等优势，详情参考《》

### AscendCL

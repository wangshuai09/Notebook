#### memcpy样例
```cpp
#include <acl/acl.h>
#include <cstdio>

int main(int argc, char** argv) {

    auto ret = aclInit(nullptr);

    void* data0;
    void* data1;
    aclrtEvent event;

    // init cpu datas
    int64_t length = 10000;
    float data_cpu[length];
    for (int i=0; i<length; i++) {
        data_cpu[i] = i;
    }
    size_t nbytes = length * sizeof(float);

    // init device0 data
    aclrtSetDevice(0);
    aclrtStream stream0 = NULL;
    aclrtCreateStream(&stream0);

    aclrtMalloc(&data0, nbytes, ACL_MEM_MALLOC_HUGE_FIRST);

    // init device1 data
    aclrtSetDevice(1);
    aclrtStream stream1 = NULL;
    aclrtCreateStream(&stream1);

    aclrtMalloc(&data1, nbytes, ACL_MEM_MALLOC_HUGE_FIRST);

    int32_t canAccessPeer = 0;
    aclrtDeviceCanAccessPeer(&canAccessPeer, 0, 1);

    if (canAccessPeer==1){
        aclrtSetDevice(1);
        aclrtDeviceEnablePeerAccess(0, 0);
        aclrtSetDevice(0);
        aclrtDeviceEnablePeerAccess(1, 0);
        // cpy cpu data to device0
        aclrtSetDevice(0);
        aclrtMemcpyAsync(data0, nbytes, data_cpu, nbytes, ACL_MEMCPY_HOST_TO_DEVICE,
                                    stream0);

        // cpy device0 to device1
        aclrtSetDevice(0);
        ret = aclrtMemcpyAsync(data1, nbytes, data0, nbytes, ACL_MEMCPY_DEVICE_TO_DEVICE,
                                        stream0);
        aclrtCreateEventExWithFlag(&event, ACL_EVENT_SYNC);
        aclrtRecordEvent(event, stream0);
        aclrtStreamWaitEvent(stream1, event);

        aclrtSynchronizeStream(stream0);

        // cpy device1 to cpu
        aclrtSetDevice(1);
        float result[length] = {0,0,0,0,0,0,0,0,0,0};
        aclrtMemcpyAsync(result, nbytes, data1, nbytes, ACL_MEMCPY_DEVICE_TO_HOST,
                                    stream1);

        // aclrtSynchronizeStream(stream0);
        aclrtSynchronizeStream(stream0);
        for (int i=0; i<10; i++) {
            printf("%f ", result[length-1-i]);
        }
    }

    aclrtDestroyStream(stream0);
    aclrtDestroyStream(stream1);
    aclrtSetDevice(0);
    aclrtResetDevice(1);
    aclrtSetDevice(0);
    aclrtResetDevice(0);

    printf("\n");
    return 0;
}
```

执行命令：`g++ -I /usr/local/Ascend/ascend-toolkit/latest/include -L /usr/local/Ascend/ascend-toolkit/latest/lib64 -o test ../tests/test-backend-runtime.cpp -lascendcl`

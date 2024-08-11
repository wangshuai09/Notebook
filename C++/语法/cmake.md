### 编译命令
```shell
-D 用于在配置阶段传递缓存变量定义给Cmake
cmake -DENABLE_FEATURE=ON ..
cmake -DMYLIB_DIR=/path/to/lib ..
```

### 语法
```shell
# 变量
$ENV{VAR} # 获取变量的值
DEFINE ENV{VAR} # 判断变量是否定义
set(VAR VALUE) # 设置变量的值
set(VAR VAR VAR1) # append


# 链接库
link_directoires(path1 path2) # 添加链接库搜索路径、
add_subdirectory(path) # 添加子模块，子模块内有CMakeLists.txt

# 字符串比较
str1 STREQUAL str2

# 列表
list(APPEND <list> var1 var2 ...) # 列表进行append操作

# 文件
file(GLOB RESULT_VAR expression) # 按匹配规则查找，结果赋值于 RESULT_VAR


# 打印信息
message([FATAL_ERROR | ERROR | WARNING | STATUS | AUTHOR | NOTICE | DEBUG] text)

FATAL_ERROR：输出错误信息并终止CMake配置过程。
ERROR：输出错误信息但不终止配置过程。
WARNING：输出警告信息。
STATUS：输出状态信息，用于显示构建过程中的进度或状态。
AUTHOR：输出作者级别的信息，通常用于调试CMakeLists.txt脚本。
NOTICE：输出通知信息，比状态信息更重要。
DEBUG：输出调试信息，仅在CMake被配置为调试模式时有效。

```

### 预定义变量
```shell
CMAKE_SYSTEM_NAME: 系统名称
CMAKE_SYSTEM_VERSION: 系统版本
CMAKE_SYSTEM_NAME: CMAKE_SYSTEM_NAME + CMAKE_SYSTEM_VERSION
CMAKE_CXX_COMPILER_ID: 编译器名，GNU/MSVC
CMAKE_SYSTEM_PROCESSOR: 处理器架构
UNIX: 在Unix和Linux下为真
WIN32: 在windows下为真
CMAKE_SOURCE_DIR: CMakeLists.txt文件所在路径
CMAKE_CURRENT_SOURCE_DIR: 当前CMakeList.txt文件所在路径
CMAKE_CXX_COMPILER: C++编译器路径
CMAKE_C_COMPILER: C编译器路径
CMAKE_BINARY_DIR: 构建结果保存路径
```


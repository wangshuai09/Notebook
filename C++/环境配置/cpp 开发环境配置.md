# C/C++ Windows 开发环境安装    
#### 下载 MinGW 
参考：https://nuwen.net/mingw.html
下载 `ming-xx.xx.exe` 文件, 解压至 `C:\` 路径

#### 配置环境变量
将 `C:\MinGW\bin` 添加至 `PATH` 环境变量

#### 验证
```shell
(base) PS D:\> g++ -v
Using built-in specs.
COLLECT_GCC=C:\MinGW\bin\g++.exe
COLLECT_LTO_WRAPPER=C:/MinGW/bin/../libexec/gcc/x86_64-w64-mingw32/13.2.0/lto-wrapper.exe
Target: x86_64-w64-mingw32
Configured with: ../src/configure --enable-languages=c,c++ --build=x86_64-w64-mingw32 --host=x86_64-w64-mingw32 --target=x86_64-w64-mingw32 --disable-multilib --prefix=/e/temp/gcc/dest --with-sysroot=/e/temp/gcc/dest --disable-libstdcxx-pch --disable-libstdcxx-verbose --disable-nls --disable-shared --disable-win32-registry --enable-threads=posix --enable-libgomp --with-zstd=/c/mingw
Thread model: posix
Supported LTO compression algorithms: zlib zstd
gcc version 13.2.0 (GCC)
(base) PS D:\> gcc -v
Using built-in specs.
COLLECT_GCC=C:\MinGW\bin\gcc.exe
COLLECT_LTO_WRAPPER=C:/MinGW/bin/../libexec/gcc/x86_64-w64-mingw32/13.2.0/lto-wrapper.exe
Target: x86_64-w64-mingw32
Configured with: ../src/configure --enable-languages=c,c++ --build=x86_64-w64-mingw32 --host=x86_64-w64-mingw32 --target=x86_64-w64-mingw32 --disable-multilib --prefix=/e/temp/gcc/dest --with-sysroot=/e/temp/gcc/dest --disable-libstdcxx-pch --disable-libstdcxx-verbose --disable-nls --disable-shared --disable-win32-registry --enable-threads=posix --enable-libgomp --with-zstd=/c/mingw
Thread model: posix
Supported LTO compression algorithms: zlib zstd
gcc version 13.2.0 (GCC)
```


g++ 是 gcc 下的一个 c++ 编译器

### 参数
```
// -I
指定一个编译器搜寻头文件的路径
// -L
指定一个链接器搜寻libraies的路径
// -o
输出文件名称
// -l
指定需要链接的库的名称，不包括lib前缀和文件扩展名
```

#### -l 参数使用顺序
推荐的顺序为 源文件-> -l -> -o
原因为链接器是按照命令行的顺序进行链接，如果 -l 放在源文件前，那么链接器就找不到库引用的符号了
有些情况可能需要将 -l 放在 -o 之后
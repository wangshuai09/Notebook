# List of str
```shell
# 字符串变量
strs="I Like You"
for value in $strs
do
   echo $value
done
# loop内字符串
for value in I Like You
do
   echo $value
done
# 字符串变量数组
array=("I" "Like" "You")
for value in ${array[@]}
do
   echo $value
done
```

#
set -e # 有报错立即退出

#
PATH 是可执行路径
LD_LIBRARY_PATH 用于程序运行时，当运行一个动态链接库的可执行文件时，系统会在这里路径查找动态库
LIBRARY_PATH 用于编译时库的查找路径，编译时，编译器会在这个路径查找静态和动态库
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

set -e # 有报错立即退出

PATH 是可执行路径
LD_LIBRARY_PATH 是动态链接库路径
LIBRARY_PATH 是静态链接库路径
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
## git config 
git 中使用 git config 来进行配置的设置和读取

#### 配置文件
  1. **系统配置**
        /etc/gitconfig: 每个用户及项目的通用配置，使用 `--system` 参数调用，windows 中根目录为 Git 的安装路径
  2. **用户配置**
        ~/.gitconfig: 只针对当前用户，使用 `--global` 参数调用，可以对所有的仓库生效
  3. **项目配置**
        仓库 Git 目录下的 .git/config: 只针对该仓库，使用 `--local` 参数使用

每个级别的会覆盖上一级配置，*eg: .git/config 会覆盖 /etc/gitconfig 内相同变量*

#### 配置检查
```shell
# 查看所有配置及其所在文件
git config --list --show-origin
# 查看所有配置
git config --list 
# 查看某一项配置
git config <key>
```

#### 配置设置
```shell
# 用户信息，用户信息会写到每一次的提交中，不可更改
git config --global user.name wangshuai09
git config --global user.email xxx@example.com

# 配置代理
git config --global http.proxy http://127.0.0.0.1:port
git config --global https.proxy https://127.0.0.1:port

# 取消代理
git config --global --unset http.proxy
git config --global --unset https.proxy
```

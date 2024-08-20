#### 提 PR 流程

1. fork 项目至个人仓库
2. clone 代码至本地
3. 新建分支
4. 修改代码
5. 提交代码
6. 在项目主页点击 pull request

#### 使用远程其他分支

```shell
git branch -a # 查看所有分支
git remote add upstream url
git fetch upstream
git checkout -b local_dev upstream/dev
```

#### rebase合并多个commit

```shell
# 左开右闭，startpoint 表示你想开始合并的前一个commit, endpoint 时省略默认为 HEAD
git rebase -i startpoint endpoint
# 弹出编辑界面
# 将除第一行 pick 修改为 s
# wq 保存后退出
# 弹出注释边界界面
# 除了第一条都删除
# 提交
git push -f
```

##### 当 rebase 出现冲突
```shell
git rebase -i resumeerror: could not apply
# 修改冲突
# 解决冲突
git add file
# 继续 rebase
git rebase --continue
```

#### 当 merge 出现冲突

```shell
# 方法一
# 从远程仓库拉取最新代码
# 此处远程分支可以使用 git remote -v 查看
git checkout -b fix_conflict origin/main
git pull git@github.com:npu-ci/llm-tool-ci.git main
# 此处执行完会有冲突
# 手动解决冲突文件
# 保存更改
git add 冲突文件
git commit -m 'fix conflict'
# 切换回开发分支
git checkout main
# 将冲突解决分支合并至要开发分支
git merge --no-ff fix_conflict
# 合并开发分支
git push

# 方法二
# 更新远程更新至 master 分支
git checkout master
git pull origin master
# 切换至工作分支
git checkout work_branch
# rebase
git rebase master
# 修改并add后
git rebase --continue
git push -f
```

# 从自己分支提交到别人的分支
```shell
git checkout -b local_dev_branch
git rebase dev_pused_origin_branch
git remode add upstream http://xxx
git push upstream local_dev_branch:upstream_branch
```

# co-authored-by
```shell
# 两个回车是必须的
$ git commit -m "Refactor usability tests.
>
>
Co-authored-by: NAME <NAME@EXAMPLE.COM>
Co-authored-by: ANOTHER-NAME <ANOTHER-NAME@EXAMPLE.COM>"
```
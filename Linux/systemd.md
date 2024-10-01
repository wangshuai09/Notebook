# systemd 简介
Syestemd 是 Linux 系统和服务管理器,提供包括引导时启动系统服务/按需激活后台程序等。引进 systemd 单元概念，常见单元包括，
- 服务单元： .service
- 目标单元： .target

Systemd 默认从 `/etc/systemd/system/` 读取配置文件

# 常用命令
```shell
# 列出目前载入服务
systemctl list-units --type service

# 列出所有可用服务单元
systemctl list-unit-files --type service

# 显示服务状态
systemctl status name.service

# 启动服务
systemctl start name.service

# 停止服务
systemctl stop name.service

# 重启服务
systemctl restart name.service

# 不中断执行情况下，重新载入配置
systemctl reload name.service

# 启用服务，将指定服务或单元添加到系统启动时期目录,在两个目录之间建立符号链接
systemctl enable name.service
# 等同于
ln -s /usr/lib/systemd/system/name.service /etc/systemd/system/name.service

# 禁用服务，阻止引导时自启动，相当于撤销符号链接
systemctl disable name.service

```

# 创建一个service
- 准备自定义服务的可执行文件
- 在`/etc/systemd/system`目录下新建单元文件
    ```shell
    touch /etc/systemd/system/name.service
    chmod 664 /etc/systemd/system/name.service
    ```
- 打开service文件，添加服务配置
    ```shell
    [Unit]
    Description=service_description
    After=network.target

    [Service]
    ExecStart=path_to_executable
    Type=forking
    PIDFile=path_to_pidfile

    [Install]
    WantedBy=default.target
    ```
- 启动服务
    ‵‵‵shell
    systemctl daemon-reload
    systemctl start name.service
    ```
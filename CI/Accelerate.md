# 触发机制
```shell
on: 
  workflow_dispatch: # 手动触发
  schedule: # 时间间隔触发
    - cron: '30 14 * * *'
```

# step.failure
```shell
# 设置 true，允许继续运行下一个step, 即使当前step运行错误
continue-on-error: true
```

# matrix
```shell
# matrix 可以使用变量来将单个 job 定义自动并行执行多个 jobs
jobs:
  example_matrix:
    strategy:
      matrix:
        version: [10, 12, 14]
    steps:
      - uses: actions/setup-node@v3
        with:
          node-version: ${{ matrix.version }}
```
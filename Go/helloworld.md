## Hello world

```shell
cd helloworld
go mod init hello
vi hello.go
```

#### 程序
```go
package main

import "fmt"

func main() {
    fmt.Println("hello world 😀😀😀")
}
```

#### 引入外部包
1. 在 [pkg.go.dev](https://pkg.go.dev/search?q=quote) 查找 quto 包
2. 在代码中引入对应的包，代码如下

```go
package main

import "fmt"

import "rsc.io/quote/v4"

func main() {
    fmt.Println("hello world 😀😀😀")
	fmt.Println(quote.Go())
}
```

3. `go mod tidy`: 用于清理优化 go.mod 和 go.sum，如果代码中新引入了依赖包，但是 go.mod中
缺少对应的声明，会自动添加至go.mod并下载至本地缓存

## 封装成包
#### Greeting包
```shell
mkdir greeting
cd greeting
go mod init example/greeting
vi greeting.go
```

可以把这个包publish，或者在引用这个包的其他包中使用 `git mod edit -replace example/greeting=../greeting` 来找到这个包

```go
package greeting

import "fmt"

func Hello(name string) string {
	message := fmt.Sprintf("Hi, %v. welcome!", name)
	return message
}
```

#### Hello中引用Greeting包
```go
package main

import "fmt"

import "rsc.io/quote/v4"
import "example/greeting"

func main() {
    fmt.Println("hello world 😀😀😀")
	fmt.Println(quote.Go())
	message := greeting.Hello("ws")
	fmt.Println(message)
}
```

```shell
go mod edit -replace example/greeting=../greeting
go mod tidy
go run .
```

## 日志信息
```go
package main

import (
    "fmt"
    "log"

    "example/greeting"
)

func main() {
    log.SetPrefix("greeting: ")
    log.SetFlags(0)

    names := []string{"ws", "xqx"}
    message, err = greeting.Hello(names)

    fmt.Println(messages)

}
```

## 测试用例
测试用例文件命名为 `xxx_test.go`, 测试用例命名为 `TestXxxx`.

```go
package greeting

import (
    "testing"
)

func TestHello(t *testing.T) {
    name:="ws"
    message:=Hello(name)
    if message == "" {
        // 打印错误信息，并终止程序
        t.Fatalf("%q", message)
    }
}
```

`go test` 运行所有测试用例

## compile and install
`go run` 在代码频繁变动的场景下，可以编译运行，但是并不会生成二进制运行文件
`go build`
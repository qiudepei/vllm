# vLLM 启动链路代码逐层走读

> 从 `vllm serve …` 命令开始,逐步追踪到 `await server_task`,明确每一行代码在做什么、为什么这样写。
>
> 走读范围:`vllm/entrypoints/cli/main.py` → `serve.py` → `api_server.py` → `launcher.py`,不延伸 EngineCore / Worker 子进程层。
>
> 阅读时间约 10 分钟;每个步骤配有**原文 + 行号 + 自然语言解释**。

---

## 0. 起点:用户在 shell 敲命令

```bash
$ vllm serve --model Qwen/Qwen3-8B --tensor-parallel-size 8 ...
```

**这一步发生了什么?**

- shell 在 `$PATH` 中找名为 `vllm` 的可执行文件
- 该可执行文件由 `pip install vllm` 时通过 `setup.py` / `pyproject.toml` 的 `console_scripts` 字段注册
- 实际指向 `vllm.entrypoints.cli.main:main`(或 `python -m vllm.entrypoints.cli.main`)

---

## 1. `vllm/entrypoints/cli/main.py` 启动

`main.py:67` 定义 `main()` 函数,启动时**只做三件事**。

### 1a. 导入所有子命令模块

```python
CMD_MODULES = [
    cli_args_serve,    # vllm serve
    cli_args_run_batch,# vllm run-batch / benchmark
    cli_args_bench,    # vllm bench
    ...
]
for mod in CMD_MODULES:
    importlib.import_module(mod)   # 副作用:每个子命令模块 top-level 自注册
```

**解释**:这一步**不调 argparse**,只触发每个子命令文件的 top-level 代码。每个子命令(如 `serve.py`)的模块顶层会调用 `register_subcommand()`,把自己的 subparser 注册到全局 dict 里。

### 1b. 构建全局 `FlexibleArgumentParser`

```python
parser = FlexibleArgumentParser(prog="vllm")
subparsers = parser.add_subparsers(dest="subparser")  # 容纳所有子命令
```

**解释**:每个子命令模块被遍历,把它注册的 subparser 挂到全局 parser 上。

### 1c. 触发 `set_defaults(dispatch_function=...)`

```python
for cmd in CMDS.values():
    cmd.subparser_init(subparsers).set_defaults(
        dispatch_function=cmd.run           # ← 关键
    )
```

**解释**:`serve` 子命令的 `cmd.run` 就是 `ServeSubcommand.run(args)`。`set_defaults` 让 argparse 在解析后把 `cmd.run` 塞到 `args.dispatch_function` 这一字段。这是**整个动态分发的挂钩点**。

---

## 2. `parser.parse_args()` — argparse 解析

```python
args = parser.parse_args()        # 用户敲的字符串被翻译成命名空间
```

**解释**:`args` 此刻已带这些属性(从子命令的 `set_defaults` 自动继承):
- `args.subparser = "serve"`
- `args.dispatch_function = ServeSubcommand.run`(函数引用,不是字符串)
- 其他 CLI 参数(model、tensor_parallel_size、port 等)

---

## 3. `args.dispatch_function(args)` 触发

```python
if hasattr(args, "dispatch_function"):   # 用户没敲子命令时走 print_help
    args.dispatch_function(args)         # 真正进入子命令
```

### 3a. `ServeSubcommand.run(args)`(`vllm/entrypoints/cli/serve.py`)

```python
def run(args) -> None:
    # 三种运行模式互斥判断
    if getattr(args, "grpc", False):     # 1) gRPC 协议
        uvloop.run(run_grpc(args)); return
    if args.headless:                    # 2) headless(只跑 EngineCore,不开 HTTP)
        return _run_headless(args)
    if args.api_server_count > 1:        # 3) 多 API server(DP-frontend LB 模式)
        return _run_multi_api_server(args)

    # ★ 默认单 server 模式 ★
    args.api_server_count = None
    uvloop.run(run_server(args))         # 看下一节
```

**解释**:**3 个 if 都落空**(默认情况),进入 `uvloop.run(run_server(args))`。`uvloop` 是高性能 asyncio 循环替代品;`run_server` 是 async 函数,被 `uvloop.run(...)` 阻塞式地"装进事件循环"运行,直到 server 退出才返回。

---

## 4. `run_server(args)`(`vllm/entrypoints/openai/api_server.py:665`)

```python
async def run_server(args, **uvicorn_kwargs):
    # uvicorn 还没接管信号前,SIGTERM 触发 KeyboardInterrupt 兜底
    signal.signal(SIGTERM, lambda *_: raise_(KeyboardInterrupt))

    listen_address, sock = setup_server(args)         # ★★ 端口已绑
    await run_server_worker(listen_address, sock, args, **uvicorn_kwargs)
```

**解释**:
- `setup_server(args)` 在**还没创建 EngineCore 之前**就预先绑端口,避免和 ray 抢端口的 race condition(issue #8204)
- `setup_server` 返回后,**端口已被占用**,失败的启动如果还想重启,行为更可预测

---

## 5. `setup_server(args)`(`api_server.py:535`)

```python
def setup_server(args):
    log_version()                      # 打印版本与 model tag
    validate_api_server_args(args)     # 合法性检查
    if args.uds:                       # Unix Domain Socket 模式
        sock = create_server_unix_socket(args.uds)
    else:
        sock = create_server_socket((args.host or "", args.port))   # TCP 模式
    set_ulimit()                       # 提高 FD 上限(高并发必要)
    return f"http://{host}:{port}", sock
```

**解释**:到这一步 socket 已经 bind 成功,但**还没 listen**。

---

## 6. `run_server_worker`(`api_server.py:681`)

```python
async def run_server_worker(listen_address, sock, args, **kwargs):
    # 加载可选 plugin
    if args.tool_parser_plugin:    ToolParserManager.import_plugin(...)
    if args.reasoning_parser_plugin:ReasoningParserManager.import_plugin(...)

    async with build_async_engine_client(args, ...) as engine_client:
        # async with 进入时:VllmConfig 构造完毕 + AsyncLLM 实例创建
        # async with 退出时:优雅 shutdown(EngineCore 子进程 kill)

        shutdown_task = await build_and_serve(
            engine_client, listen_address, sock, args, **kwargs
        )
    try:
        await shutdown_task
    finally:
        sock.close()
```

**解释**:`build_async_engine_client` 是 `@asynccontextmanager`,**进入时建引擎,退出时拆引擎**。

---

## 7. `build_and_serve`(`api_server.py:572`)

```python
async def build_and_serve(engine_client, listen_address, sock, args, **uvicorn_kwargs):
    # 关键步骤分三段
    supported_tasks = await engine_client.get_supported_tasks()  # 问引擎支持哪些任务
    app = build_app(args, supported_tasks, model_config)        # ★ FastAPI app
    await init_app_state(engine_client, app.state, args, ...)    # 把 engine_client 注入 app.state

    return await serve_http(app, sock=sock, host=..., port=..., **kwargs)
```

**解释**:
- `build_app` 把 OpenAI 兼容路由(`/v1/chat/completions`、`/v1/completions`、`/v1/models` 等)全部注册到 FastAPI app
- `init_app_state` 把 `engine_client` 写到 `app.state.engine_client`,后续 HTTP handler 就能 `app.state.engine_client.generate(...)`
- `serve_http` 真正启动 uvicorn + 阻塞等待

---

## 8. `serve_http`(`vllm/entrypoints/launcher.py:26`)— 真正的卡死点

```python
async def serve_http(app, sock, enable_ssl_refresh=False, **uvicorn_kwargs):
    # ...日志打印所有 routes...

    config = uvicorn.Config(app, **uvicorn_kwargs)
    config.load()
    server = uvicorn.Server(config)
    app.state.server = server

    loop = asyncio.get_running_loop()
    watchdog_task = loop.create_task(watchdog_loop(server, app.state.engine_client))
    server_task   = loop.create_task(server.serve(sockets=[sock]))    # ★★
    shutdown_task = loop.create_task(handle_shutdown(loop, ...))

    try:
        await server_task                       # ★★★ 主流程在此卡住
    except asyncio.CancelledError:
        return server.shutdown()
    finally:
        shutdown_task.cancel(); watchdog_task.cancel()
```

**解释**(精简版,见下节逐行展开):
- `server.serve(sockets=[sock])` 启动 uvicorn 主循环,开始接受 HTTP 请求
- `await server_task`:**主进程被这件事"焊死"**。当 `shutdown_task` 触发 `server_task.cancel()` 时,`server_task` 抛 `CancelledError` → 异常冒泡 → 进入 except 分支 → `server.shutdown()` → uvicorn 收尾 → `server_task` 任务结束 → `await` 返回
- `watchdog_task` 每 5 秒侦测 engine 是否已死、是否有 hang

---

## 8A. `serve_http` 逐行精读(`launcher.py:71-141`)

下面把上一节里 ★★/★★★ 标注的代码块**逐行**拆开,讲清楚每一行在做什么、为什么这样写、它和谁协作。

### 8A.1 `config = uvicorn.Config(app, **uvicorn_kwargs)`

```python
config = uvicorn.Config(app, **uvicorn_kwargs)        # launcher.py:71
```

- **做了什么**:把 FastAPI `app` 和 `run_server_worker` 透传进来的所有 uvicorn 参数(`host`、`port`、`log_level`、`ssl_keyfile`、`timeout_keep_alive` 等)封装成一个 **`uvicorn.Config` 对象**。
- **关键点 1 — 这只是"配置对象",不是运行对象**。`uvicorn.Config` 是个 **dataclass**,只存字段,**不会启动任何东西**。这一步对 `app` 一行代码都不执行,FastAPI 也没被 import 解析过。
- **关键点 2 — `app` 此时已经是"成品"**。回想 7 节:`build_app` 已经把 OpenAI 兼容路由全注册好了;`init_app_state` 已经把 `engine_client` 挂到 `app.state.engine_client`。所以这里传进去的 `app` 是一个**路由表 + 依赖注入都 ready** 的 FastAPI 实例。
- **关键点 3 — `**uvicorn_kwargs` 的来源**。它来自 `run_server(args, **uvicorn_kwargs)` → `run_server_worker(..., **uvicorn_kwargs)` → `build_and_serve(..., **uvicorn_kwargs)` → `serve_http(app, sock=..., **uvicorn_kwargs)` 一路透传,里面包含 `host`、`port`、`ssl_certfile`、`ssl_keyfile`、`log_level` 等。

### 8A.2 `config.load()`

```python
config.load()                                          # launcher.py:75
```

- **做了什么**:`uvicorn.Config.load()` 是 uvicorn 内部的一个**懒加载钩子**,它的核心职责是:
  1. **解析应用对象**(本例里已经是 FastAPI 实例,这一步比较轻);
  2. **加载 SSL 上下文**(如果 `ssl_keyfile` 等有配);
  3. **初始化 logging**(把 uvicorn 自带的 access log / error log 装配好);
  4. **准备 `Server` 运行时需要的回调**(signal handler 的延迟绑定等)。
- **为什么不在 `uvicorn.Config(...)` 构造时就做?** 因为有些资源(比如 SSL context、logging)构造代价高、且依赖 `loop` 是否启动;把它延后到 `.load()` 一步调用,可以让"创建配置"和"加载资源"分离。
- **副作用**:`config.load()` 之后,`config.loaded = True`,`config.ssl` 等字段被填好。后面 `SSLCertRefresher` 才能从 `config.ssl` 取到 `ssl_context`。

### 8A.3 `server = uvicorn.Server(config)`

```python
server = uvicorn.Server(config)                        # launcher.py:76
```

- **做了什么**:`uvicorn.Server` 是**真正可运行的"服务器对象"**,它持有 `config`、生命周期状态(`should_exit`、`force_exit`)、启动/关闭钩子等。
- **关键点 1 — `Server.serve()` 才是入口**。`uvicorn.Server` 实例化**不会启动服务器**;启动要靠 `await server.serve(...)`。这也是为什么下面用 `loop.create_task(server.serve(...))` 而不是直接 `await`。
- **关键点 2 — `Server` 内置 shutdown 控制面**。`server.should_exit = True` 表示"请优雅退出";`server.force_exit = True` 表示"立即暴力退出"。下面 `handle_shutdown` 通过写 `should_exit` 来软关停,然后再 `cancel()` 兜底。

### 8A.4 `app.state.server = server`

```python
app.state.server = server                              # launcher.py:77
```

- **做了什么**:把 `Server` 实例挂到 FastAPI 的 `app.state` 上。
- **为什么需要这一步?** 因为有些**运行时中途的逻辑**(比如调试端点 `/metrics`、自定义 health check、或者 vllm-ascend 自己的某个中间件)需要**主动触发 uvicorn 关闭**(例如某个致命错误下让 API server 退出)。它无法拿到 `serve_http` 局部作用域里的 `server` 变量,但可以从 `request.app.state.server` 拿到这个引用。
- **等价于**:"给 app 留一个逃生舱口":app 自己可以在运行时 `self.state.server.should_exit = True`,让 uvicorn 平滑退出。

### 8A.5 三件套 Task 的创建

```python
loop = asyncio.get_running_loop()                      # launcher.py:79

watchdog_task = loop.create_task(                      # launcher.py:81
    watchdog_loop(server, app.state.engine_client)
)
server_task = loop.create_task(                        # launcher.py:82
    server.serve(sockets=[sock] if sock else None)
)
shutdown_task = loop.create_task(handle_shutdown(...)) # launcher.py:122
```

这一段的核心是**"在同一事件循环里并发起 3 个长期运行的协程"**,每个角色不同:

| Task | 谁是它 | 做什么 | 何时结束 |
|---|---|---|---|
| `server_task` | `uvicorn.Server.serve` | **HTTP 主循环**:accept 连接、解析 HTTP、分发到 FastAPI handler、返回响应 | `should_exit=True` 之后,graceful shutdown 走完才结束;或被外部 `cancel()` 立即结束 |
| `watchdog_task` | `watchdog_loop(server, engine)` | **健康巡检**:每 5 秒看 `engine.errored and not engine.is_running`,发现引擎已死就 `server.should_exit=True` | 进程退出时被 `cancel()` |
| `shutdown_task` | `handle_shutdown(loop, ...)` | **信号/事件处理**:监听 SIGINT/SIGTERM 或 shutdown_event;触发后**主动关闭引擎 + 取消 server_task** | `shutdown_event.wait()` 返回后,内部逻辑跑完自然结束;或 finally 里被 `cancel()` |

**`loop = asyncio.get_running_loop()` 这一行的意义**:因为当前 `serve_http` 已经在 `uvloop.run(...)` 里运行,所谓 `get_running_loop()` 就是把当前的 uvloop 实例拿到,**后面所有 `create_task` 都会自动跑在同一个 loop 上**(否则 vllm 引擎的 async 接口和 uvicorn 的 HTTP 处理会被不同 loop 撕裂,极容易出问题)。

**为什么 `server.serve(sockets=[sock])` 必须传 `sockets=`?** 因为前面在 `setup_server` 里已经把端口 bind 了,uvicorn 直接接管这个**已 bind 的 socket**,这样可以避免 uvicorn 自己再 bind 一次导致的 "address already in use" race。`sock` 是 `socket.socket` 对象,uvicorn 内部会从它拿 fd 注册到 `loop`。

### 8A.6 `await server_task` —— 主流程为什么"卡死"

```python
try:
    await server_task                                 # launcher.py:125  ★★★
```

- **为什么"卡死"?** `await` 在 asyncio 里的语义是"把当前协程挂起,直到被 await 的对象完成"。`server_task` 是 uvicorn 的 HTTP 主循环,**正常情况下它会一直跑、永远不结束**——所以 `await server_task` 就**一直挂着**,直到有下面三种情况之一发生:
  1. **`shutdown_task` 触发了 `server_task.cancel()`** → `await` 处抛 `asyncio.CancelledError` → 进入 `except` 分支;
  2. **`shutdown_task` 设置了 `server.should_exit = True`** → uvicorn 走 graceful shutdown → `server_task` 自然返回 → `await` 正常返回 → 进入 `try` 后的 `return dummy_shutdown()`;
  3. **`server_task` 自己内部异常**(比如端口被抢、SSL 加载失败) → 异常冒泡 → 同样进入 `except` 分支。
- **"焊死"的工程意义**:从这一刻起,主协程里所有 `await server_task` 之下的代码**都不会被执行**(包括 `finally` 之外的一切),直到 server 真正结束。这就是为什么 vllm 启动日志最后一行经常是 "Application startup complete.",**之后就静默**,因为控制权在 uvicorn 手里。
- **★ 注意"为什么用 `try` 而不是直接 `await`"**:因为 uvicorn 的 `serve()` 在 shutdown 路径上有时候会抛 `CancelledError`(尤其是外部强行 cancel 时),必须接住;此外 uvicorn 自己内部也会在 `serve()` 里处理 shutdown 后让协程**正常 return**(此时 `await` 不会抛),所以 try/except 是兼容两种退出路径的兜底写法。

### 8A.7 `except asyncio.CancelledError`

```python
except asyncio.CancelledError:                         # launcher.py:127
    port = uvicorn_kwargs["port"]
    process = find_process_using_port(port)
    if process is not None:
        logger.warning("port %s is used by process %s ...", ...)
    logger.info("Shutting down FastAPI HTTP server.")
    return server.shutdown()
```

- **进入条件**:当 `server_task` 被外部代码(`shutdown_task` 里 `server_task.cancel()`)显式取消时,`await server_task` 会抛 `CancelledError`。
- **做了三件事**:
  1. **诊断**:如果端口还在被别的进程占用,打个 warning(常发生在异常重启时);
  2. **打日志**:明确打出 "Shutting down FastAPI HTTP server." 让运维知道;
  3. **`server.shutdown()`**:**同步**触发 uvicorn 的优雅关闭(等 in-flight 请求完成、关闭连接、回收资源),然后 `return` 把 `serve_http` 这个 async 函数退出。
- **`return server.shutdown()` 的返回值**:uvicorn 的 `server.shutdown()` 是 async 函数,**它返回的是一个 coroutine**。这里直接 `return` 这个 coroutine 给 `serve_http` 的调用者(`build_and_serve`),**不是它的结果**——这是 FastAPI/uvicorn 集成时常用的"返回 coroutine 让上游继续 await"模式,保证 shutdown 真的完成才让 `run_server_worker` 的 `async with` 退出。

### 8A.8 `finally`

```python
finally:                                              # launcher.py:139
    shutdown_task.cancel()
    watchdog_task.cancel()
```

- **进入条件**:**无论 try/except 怎么走,finally 必执行**。这保证了**两个长期 task 不会泄露**:
  - `shutdown_task.cancel()`:它内部的 `await shutdown_event.wait()` 会抛 `CancelledError`,从此这个 task 死亡;
  - `watchdog_task.cancel()`:`while True: await asyncio.sleep(5)` 在 sleep 处抛 `CancelledError`,从此这个 task 死亡。
- **为什么这里不需要 cancel `server_task`?** 因为进入 finally 时,`server_task` 要么已经正常 `return` 了(优雅退出路径)、要么已经因 `CancelledError` 抛掉了(cancel 路径)。**唯一例外**:如果 try 块里 `await server_task` 是因为**别的异常**(不是 CancelledError)而退出,`server_task` 可能还活着。但 vllm 这里的 `finally` 没 cancel 它,是因为**异常路径下进程即将崩溃**,让操作系统回收更直接。
- **⚠ 顺便提一个 vllm-ascend 经常踩的坑**:如果有 NPU 初始化失败等异常发生在 `await server_task` 里,这条 finally 会执行,但**没 cancel `server_task`**。如果 server_task 内部有 NPU handle 还引用着,可能让资源回收不彻底——这也是为什么 vllm-ascend 在某些 patch 里会单独在 `try/except BaseException` 上多包一层。

### 8A.9 三 task 的生命周期可视化

```
时间 ──────────────────────────────────────────────────────────►

loop.create_task(watchdog_task) ─► 每 5s 检查 engine ─► 直到 cancel
loop.create_task(server_task)   ─► HTTP 主循环 ◄────────────┐
loop.create_task(shutdown_task) ─► 等 shutdown_event.wait() │
                                                          │
   ┌─────── 用户按 Ctrl-C 或 K8s 发 SIGTERM ──────────────┘
   │  shutdown_event.set()
   │  shutdown_task 内:
   │    engine_client.shutdown(timeout=...)   # 停引擎子进程
   │    server.should_exit = True            # 通知 uvicorn 软退
   │    server_task.cancel()                 # 兜底硬退
   │    watchdog_task.cancel()               # 同步停 watchdog
   ▼
server_task 抛 CancelledError (or 自然 return)
   ▼
await server_task 结束
   ▼
进入 except 分支 → server.shutdown() → 收尾
   ▼
finally: shutdown_task.cancel(); watchdog_task.cancel()
   ▼
serve_http 返回 → 回到 build_and_serve → 回到 run_server_worker
   ▼
async with build_async_engine_client(...) 退出
   ▼
uvloop.run(run_server(args)) 返回
   ▼
进程退出
```

### 8A.10 这段代码的"5 个关键设计"小结

1. **`config` / `server` / `app.state.server` 三级引用**:`Config` 装配置、`Server` 装运行态、`app.state.server` 给运行时 app 自己用——三层各司其职。
2. **`server.serve(sockets=[sock])`**:复用 `setup_server` 阶段已经 bind 的 socket,避免 race condition(issue #8204)。
3. **三 Task 并发模型**:`server_task`(HTTP)+ `shutdown_task`(信号)+ `watchdog_task`(健康检查),用 uvloop 同一 loop 调度。
4. **`await server_task` 是单一阻塞点**:整个进程从此以后**只靠 shutdown_task 来解除阻塞**,所有退出路径汇于一处。
5. **`try/except CancelledError / finally` 双重保险**:无论优雅退出还是强行 cancel,finally 都保证两个长期 task 被清理,避免协程泄露。

---

## 9. 整体 4 阶段总图

```
[用户命令行]
  │
  ▼
 ① vllm CLI 入口(main.py:67)
   │ - import lib 子命令模块(副作用:自注册)
   │ - FlexibleArgumentParser 全局 builder
   │ - set_defaults(dispatch_function=...)  ★ 绑定钩子
  ▼
 ② ServeSubcommand.run(args)
   │ - 三路互斥 if/elif
   │ - 默认 → uvloop.run(run_server(args))     ★★ 同步→异步切换
  ▼
 ③ run_server(args) async
   │ - SIGTERM 兜底
   │ - setup_server: bind 端口
   │ - run_server_worker:
   │     ├─ async with build_async_engine_client → AsyncLLM 建好
   │     └─ build_and_serve:
   │           ├─ build_app(FastAPI)
   │           ├─ init_app_state(engine_client)
   │           └─ serve_http:
   │                 ├─ uvicorn.Server(config).serve(sockets=[sock])
   │                 └─ await server_task   ★★★ 主线程阻塞点
  ▼
 ④ HTTP 请求处理循环
   │ - req 到达 FastAPI 路由
   │ - → app.state.engine_client.generate(req) → AsyncLLM
   │ - → ZMQ 推到 EngineCore 子进程
   │ - → EngineCore 调度(scheduler.schedule)
   │ - → Worker 跑 model(...)(NPU 实际推理)
   │ - → ZMQ 拉回结果
   │ - → 流式或非流式返回 OpenAI 格式响应
  ▼
 ⑤ 收到 SIGTERM/SIGINT
   │ - shutdown_event 被信号 handler set
   │ - handle_shutdown 把 server_task.cancel()
   │ - CancelledError 冒泡 → server.shutdown()
   │ - 退出 async with → AsyncLLM.shutdown() → EngineCore 子进程 kill
   │ - uvloop.run 返回 → 进程退出
```

---

## 10. 白话总结

> 整个 vLLM 启动**就是 4 步**:
> 1. **CLI 解析 + dispatch_function**(`main.py`):route 一次子命令,把对应的 `cmd` 挂到 `args.dispatch_function` 上
> 2. **进 uvloop.run(run_server)**(`serve.py`):同步转异步,绑端口(`setup_server`)
> 3. **建 async with AsyncLLM + build FastAPI app**(`api_server.py`):所有 OpenAI 兼容路由全部注册,engine_client 注入 `app.state`
> 4. **await server_task**(`launcher.py`):uvicorn 主循环开始接收 HTTP,主进程到此就被钉死,直到 SIGTERM 把它取消
>
> **关键设计**:
> - `dispatch_function` 是 argparse 的"动态分发"技巧
> - `async with` 上下文管理器同时承担**建引擎**和**拆引擎**两个职责
> - `server_task` + `watchdog_task` + `shutdown_task` 三个并发 Task + 信号处理,构成完整的 uvicorn 生命周期

---

## 11. 文件路径速查

| 步骤 | 文件 | 关键行 |
|---|---|---|
| 1  CLI 入口 | `vllm/entrypoints/cli/main.py` | 67 `main()` |
| 3  dispatch_function 设置 | 同上 | `set_defaults(dispatch_function=...)` |
| 3a ServeSubcommand.run | `vllm/entrypoints/cli/serve.py` | `run(args)` 默认分支 |
| 4  run_server | `vllm/entrypoints/openai/api_server.py` | 665 `async def run_server` |
| 5  setup_server | 同上 | 535 `def setup_server` |
| 6  run_server_worker | 同上 | 681 |
| 7  build_and_serve | 同上 | 572 |
| 8  serve_http | `vllm/entrypoints/launcher.py` | 26 |
| 8.1 await server_task | 同上 | 125 |
| 8A  serve_http 逐行精读 | 同上 | 71-141 |

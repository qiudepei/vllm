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

shell 在 `$PATH` 中找名为 `vllm` 的可执行文件 —— 该可执行文件由 `pip install vllm` 时通过 `pyproject.toml` 的 `console_scripts` 字段注册,实际指向 `vllm.entrypoints.cli.main:main`(或 `python -m vllm.entrypoints.cli.main`)。

---

## 1. `vllm/entrypoints/cli/main.py` 启动

`main.py:67` 定义 `main()` 函数,启动时**只做三件事**。

```python
# 1a) 导入所有子命令模块 → 副作用:每个子命令模块 top-level 自注册
CMD_MODULES = [
    cli_args_serve,        # vllm serve
    cli_args_run_batch,    # vllm run-batch / benchmark
    cli_args_bench,        # vllm bench
    ...
]
for mod in CMD_MODULES:
    importlib.import_module(mod)   # 此时还没调 argparse,只触发每个子命令文件 top-level 的 register_subcommand()

# 1b) 构建全局 parser
parser = FlexibleArgumentParser(prog="vllm")
subparsers = parser.add_subparsers(dest="subparser")  # 容纳所有子命令

# 1c) 触发 set_defaults(dispatch_function=...) → 整个动态分发的挂钩点
for cmd in CMDS.values():
    cmd.subparser_init(subparsers).set_defaults(
        dispatch_function=cmd.run        # ★ 把 ServeSubcommand.run 函数引用塞到 args.dispatch_function
    )
```

**设计要点**:子命令在 import 时通过 `register_subcommand()` 把自己塞进全局 `CMDS` dict,`main.py` 完全不感知每个子命令的参数。`set_defaults(dispatch_function=cmd.run)` 是 argparse 的"动态分发"技巧 —— `parse_args` 之后 `args.dispatch_function` 就是真正的入口函数引用(不是字符串,不是 import path)。

---

## 2. `parser.parse_args()` — argparse 解析

```python
args = parser.parse_args()        # 用户敲的字符串 → 命名空间(含 dispatch_function)
```

`args` 此刻已带(从子命令的 `set_defaults` 自动继承):
- `args.subparser = "serve"`
- `args.dispatch_function = ServeSubcommand.run`(函数引用)
- 其他 CLI 参数(model、tensor_parallel_size、port 等)

---

## 3. `args.dispatch_function(args)` 触发

```python
if hasattr(args, "dispatch_function"):   # 用户没敲子命令时无此字段,走 print_help
    args.dispatch_function(args)         # 真正进入子命令
```

### 3a. `ServeSubcommand.run(args)`(`vllm/entrypoints/cli/serve.py`)

```python
def run(args) -> None:
    # 三种运行模式互斥判断
    if getattr(args, "grpc", False):              # 1) gRPC 协议
        uvloop.run(run_grpc(args)); return
    if args.headless:                             # 2) headless(只跑 EngineCore,不开 HTTP)
        return _run_headless(args)
    if args.api_server_count > 1:                 # 3) 多 API server(DP-frontend LB 模式)
        return _run_multi_api_server(args)

    # ★ 默认单 server 模式 ★
    args.api_server_count = None
    uvloop.run(run_server(args))                  # 同步→异步切换,run_server 退出后 uvloop.run 才返回
```

**设计要点**:`uvloop` 是高性能 asyncio 循环替代品;`run_server` 是 async 函数,被 `uvloop.run(...)` 阻塞装进事件循环,直到 server 退出才返回。**3 个 if 全落空**(默认情况)走 `uvloop.run(run_server(args))`。

---

## 4. `run_server(args)`(`vllm/entrypoints/openai/api_server.py:665`)

```python
async def run_server(args, **uvicorn_kwargs):
    # uvicorn 还没接管信号前,SIGTERM 触发 KeyboardInterrupt 兜底
    signal.signal(SIGTERM, lambda *_: raise_(KeyboardInterrupt))

    listen_address, sock = setup_server(args)         # ★★ 端口已 bind,见下一节
    await run_server_worker(listen_address, sock, args, **uvicorn_kwargs)
```

**设计要点**:`setup_server` 在**还没创建 EngineCore**之前就预先 bind 端口,避免和 ray 抢端口的 race condition(issue #8204);失败的启动如果还想重启,端口已被占用,行为更可预测。

---

## 5. `setup_server(args)`(`api_server.py:535`)

```python
def setup_server(args):
    log_version()                        # 打印版本与 model tag
    validate_api_server_args(args)       # 合法性检查(--tool-call-parser / --reasoning-parser 校验)
    if args.uds:                         # Unix Domain Socket 模式
        sock = create_server_unix_socket(args.uds)
    else:
        sock = create_server_socket((args.host or "", args.port))   # TCP 模式
    set_ulimit()                         # 提高 FD 上限(高并发必要)
    return f"http://{host}:{port}", sock
```

到这一步 socket 已经 bind 成功,但**还没 listen** —— listen 交给 uvicorn 的 `loop.create_server(sock=...)` 做。

---

## 6. `run_server_worker`(`api_server.py:681`)

```python
async def run_server_worker(listen_address, sock, args, **kwargs):
    # 加载可选 plugin
    if args.tool_parser_plugin:     ToolParserManager.import_plugin(...)
    if args.reasoning_parser_plugin: ReasoningParserManager.import_plugin(...)

    async with build_async_engine_client(args, ...) as engine_client:
        # async with 进入时:VllmConfig 构造完毕 + AsyncLLM 实例创建 + EngineCore 子进程拉起
        # async with 退出时:优雅 shutdown(EngineCore 子进程 kill)

        shutdown_task = await build_and_serve(
            engine_client, listen_address, sock, args, **kwargs
        )
    try:
        await shutdown_task           # 等 build_and_serve 返回的 shutdown_task 跑完
    finally:
        sock.close()                  # 端口释放
```

**设计要点**:`build_async_engine_client` 是 `@asynccontextmanager`,**进入时建引擎,退出时拆引擎** —— 同一个 `async with` 承担"建"和"拆"两个职责,异常路径也能保证拆。

---

## 7. `build_and_serve`(`api_server.py:572`)

```python
async def build_and_serve(engine_client, listen_address, sock, args, **uvicorn_kwargs):
    supported_tasks = await engine_client.get_supported_tasks()  # 问引擎支持哪些任务(generate/transcribe/pool/...)
    model_config    = engine_client.model_config                 # 拿模型配置

    app = build_app(args, supported_tasks, model_config)        # ★ FastAPI app,挂完所有路由
    await init_app_state(engine_client, app.state, args, ...)    # 把 engine_client 注入 app.state

    return await serve_http(app, sock=sock, host=..., port=..., **kwargs)  # 启动 uvicorn + 阻塞
```

**设计要点**:`build_app` 把 OpenAI 兼容路由全注册(详见[stage2 stage1 §1](./stage2_request_lifecycle_walkthrough.md#1-build_app-装配路由api_serverpy157-307));`init_app_state` 把 `engine_client` 写到 `app.state.engine_client`,后续 HTTP handler 就能 `request.app.state.engine_client.generate(...)`。

---

## 8. `serve_http`(`vllm/entrypoints/launcher.py:26`)— 真正的卡死点

```python
async def serve_http(app, sock, enable_ssl_refresh=False, **uvicorn_kwargs):
    # ...日志打印所有 routes...

    # 三层引用:Config 装配置 → Server 持运行态 → app.state.server 给 handler 留逃生舱口
    config = uvicorn.Config(app, **uvicorn_kwargs)        # L71  dataclass,只存字段(不启动)
    config.load()                                          # L75  懒加载:SSL context、logging
    server = uvicorn.Server(config)                        # L76  运行态对象(本身不启动)
    app.state.server = server                              # L77  handler 可 app.state.server.should_exit=True

    # 三 Task 同 loop 并发
    loop = asyncio.get_running_loop()                      # L79  当前 uvloop 引用
    watchdog_task = loop.create_task(watchdog_loop(server, app.state.engine_client))  # L81  5s 巡检
    server_task   = loop.create_task(server.serve(sockets=[sock]))                     # L82  HTTP 主循环
    shutdown_task = loop.create_task(handle_shutdown(loop, ...))                       # L122 等信号

    try:
        await server_task                                 # L125  ★★★ 唯一阻塞点
        return dummy_shutdown()
    except asyncio.CancelledError:                         # L127  server_task 被 cancel 时
        ...logger.warning(端口占用诊断)...
        return server.shutdown()                           # 返回 coroutine 给上游 await
    finally:                                              # L139  无论怎么走都执行
        shutdown_task.cancel(); watchdog_task.cancel()    # 防协程泄露;server_task 已死不再 cancel
```

**设计要点**:
- **`Config` vs `Server`**:前者装配置(`host/port/ssl/timeout`),后者持运行态(`should_exit/force_exit`);实例化都不启动,启动必须 `await server.serve()`。
- **`sockets=[sock]` 复用 `setup_server` 已 bind 的 socket**(issue #8204),避免 uvicorn 二次 bind 造成 "address already in use" race。
- **三 Task 角色**:`server_task`(HTTP 主循环)+ `watchdog_task`(5s 巡检 `engine.errored`)+ `shutdown_task`(等信号/SIGTERM,触发后停引擎 + cancel server)。
- **`await server_task` 为什么"卡死"**:正常路径下 uvicorn 永不结束;解除阻塞只有 3 条路 ——① `shutdown_task.cancel(server_task)` 抛 `CancelledError`;② `server.should_exit=True` 自然 return;③ `serve()` 内部异常冒泡。
- **`return server.shutdown()` 是返回 coroutine,不是结果** —— 让 `run_server_worker` 的 `async with` 等到 shutdown 真完成才退出。
- **`finally` 不 cancel `server_task`**:进入 finally 时它要么正常 return 要么已抛 CancelledError;异常路径下靠 OS 直接回收。

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
 ④ HTTP 请求处理循环(详见 stage2)
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
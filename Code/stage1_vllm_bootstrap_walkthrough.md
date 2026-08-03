# vLLM 启动链路代码逐层走读

> 从 `vllm serve …` 命令开始,逐步追踪到 `await server_task`,明确每一行代码在做什么、为什么这样写。
>
> 走读范围:`vllm/entrypoints/cli/main.py` → `serve.py` → `api_server.py` → `launcher.py`,不延伸 EngineCore / Worker 子进程层。
>
> 阅读时间约 6 分钟;每个步骤配有**原文 + 行号 + 自然语言解释**。

---

## 0. 起点:用户在 shell 敲命令

```bash
$ vllm serve --model Qwen/Qwen3-8B --tensor-parallel-size 8 ...
```

shell 在 `$PATH` 中找名为 `vllm` 的可执行文件 —— 由 `pip install vllm` 时通过 `pyproject.toml` 的 `console_scripts` 字段注册,实际指向 `vllm.entrypoints.cli.main:main`(或 `python -m vllm.entrypoints.cli.main`)。

---

## 1. `vllm/entrypoints/cli/main.py` 三件事 & `ServeSubcommand.run`

```python
# main.py:67
def main():
    # 1) 导入所有子命令模块(副作用:每个子命令模块 top-level 自注册到全局 CMDS dict)
    for mod in [cli_args_serve, cli_args_run_batch, cli_args_bench, ...]:
        importlib.import_module(mod)

    # 2) 构建全局 parser
    parser = FlexibleArgumentParser(prog="vllm")
    subparsers = parser.add_subparsers(dest="subparser")

    # 3) 把每个子命令的入口函数挂到 args.dispatch_function(整个动态分发的挂钩点)
    for cmd in CMDS.values():
        cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.run)

    args = parser.parse_args()
    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)        # 真正进入子命令
```

```python
# vllm/entrypoints/cli/serve.py
def run(args) -> None:
    # 三种运行模式互斥判断
    if getattr(args, "grpc", False):                # 1) gRPC 协议
        uvloop.run(run_grpc(args)); return
    if args.headless:                               # 2) headless(只跑 EngineCore,不开 HTTP)
        return _run_headless(args)
    if args.api_server_count > 1:                   # 3) 多 API server(DP-frontend LB 模式)
        return _run_multi_api_server(args)

    args.api_server_count = None
    uvloop.run(run_server(args))                    # ★★ 默认单 server 模式,同步→异步切换
```

**设计要点**:子命令在 import 时通过 `register_subcommand()` 把自己塞进 `CMDS`,`main.py` 完全不感知每个子命令的参数。`dispatch_function=cmd.run` 是 argparse 的"动态分发"技巧 —— `args.dispatch_function` 是**函数引用**,不是字符串,parse 之后直接调用即可。`uvloop.run` 是同步→异步切换,run_server 退出后才返回。

---

## 2. `run_server` → `setup_server`(`api_server.py:665 / 535`)

```python
async def run_server(args, **uvicorn_kwargs):
    signal.signal(SIGTERM, lambda *_: raise_(KeyboardInterrupt))   # uvicorn 接管信号前的兜底
    listen_address, sock = setup_server(args)                      # 端口已 bind
    await run_server_worker(listen_address, sock, args, **kwargs)

def setup_server(args):
    log_version(); validate_api_server_args(args)
    if args.uds:
        sock = create_server_unix_socket(args.uds)
    else:
        sock = create_server_socket((args.host or "", args.port))  # TCP bind,SO_REUSEADDR/SO_REUSEPORT
    set_ulimit()                                                  # 提高 FD 上限
    return listen_address, sock
```

**设计要点**:**在还没创建 EngineCore 之前就 bind 端口**,避免和 ray 抢端口的 race condition(issue #8204);失败启动想重启时,端口已被占用,行为更可预测。socket 已 bind 但**还没 listen** —— listen 交给 uvicorn 的 `loop.create_server(sock=...)` 做。

---

## 3. `run_server_worker` → `build_and_serve`(`api_server.py:681 / 572`)

```python
async def run_server_worker(listen_address, sock, args, **kwargs):
    if args.tool_parser_plugin:     ToolParserManager.import_plugin(...)
    if args.reasoning_parser_plugin: ReasoningParserManager.import_plugin(...)

    async with build_async_engine_client(args, ...) as engine_client:
        # 进入:
        #   1) VllmConfig 构造
        #   2) AsyncLLM 创建(API server 进程侧)
        #   3) EngineCore 子进程拉起:multiprocessing.Process(target=EngineCoreProc.run_engine_core, ...)
        #      由 CoreEngineProcManager.__init__ (vllm/v1/engine/utils.py:140-155) 创建 + proc.start() 启动
        #      run_engine_core (core.py:1093) 是子进程入口(static method)
        # 退出:优雅 shutdown(EngineCore 子进程 kill)
        shutdown_task = await build_and_serve(engine_client, listen_address, sock, args, **kwargs)
    try:
        await shutdown_task
    finally:
        sock.close()

async def build_and_serve(engine_client, listen_address, sock, args, **uvicorn_kwargs):
    supported_tasks = await engine_client.get_supported_tasks()  # 问引擎支持哪些任务
    model_config    = engine_client.model_config
    app = build_app(args, supported_tasks, model_config)          # FastAPI app + 全部路由
    await init_app_state(engine_client, app.state, args, ...)     # engine_client 注入 app.state
    return await serve_http(app, sock=sock, **uvicorn_kwargs)    # 启动 uvicorn + 阻塞
```

**设计要点**:`build_async_engine_client` 是 `@asynccontextmanager`,**进入时建引擎、退出时拆引擎** —— 同一个 `async with` 承担"建"和"拆"两个职责,异常路径也能保证拆。`build_app` / `init_app_state` 详见 [stage2 §1](./stage2_request_lifecycle_walkthrough.md#1-build_app-装配路由api_serverpy157-307)。

---

## 4. `serve_http`(`vllm/entrypoints/launcher.py:26`)— 真正的卡死点

```python
async def serve_http(app, sock, enable_ssl_refresh=False, **uvicorn_kwargs):
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
- `Config` vs `Server`:前者装配置,后者持运行态;实例化都不启动,启动必须 `await server.serve()`。
- `sockets=[sock]` 复用 `setup_server` 已 bind 的 socket,避免 uvicorn 二次 bind。
- **三 Task 角色**:`server_task`(HTTP 主循环)+ `watchdog_task`(5s 巡检 `engine.errored`)+ `shutdown_task`(等信号、触发后停引擎 + cancel server)。
- **`await server_task` 为什么"卡死"**:正常路径下 uvicorn 永不结束;解除阻塞只有 3 条路 ——① `shutdown_task.cancel(server_task)` 抛 `CancelledError`;② `server.should_exit=True` 自然 return;③ `serve()` 内部异常冒泡。
- **`return server.shutdown()` 是返回 coroutine,不是结果** —— 让上游 `async with` 等到 shutdown 真完成才退出。

---

## 5. 整体 4 阶段总图

```
[用户命令行]
  │
  ▼
 ① vllm CLI 入口(main.py:67)
  │ - import 子命令模块(副作用:自注册)
  │ - FlexibleArgumentParser + set_defaults(dispatch_function=cmd.run)  ★ 挂钩点
  ▼
 ② ServeSubcommand.run(args)
  │ - 三路互斥 if/elif
  │ - 默认 → uvloop.run(run_server(args))     ★★ 同步→异步切换
  ▼
 ③ run_server(args) async
  │ - setup_server: bind 端口(issue #8204)
  │ - run_server_worker:
  │     ├─ async with build_async_engine_client → AsyncLLM 建好
  │     └─ build_and_serve:
  │           ├─ build_app(FastAPI + 全部路由)
  │           ├─ init_app_state(engine_client)
  │           └─ serve_http: uvicorn.Server(config).serve(sockets=[sock])
  │                                └─ await server_task   ★★★ 主线程阻塞点
  ▼
 ④ HTTP 请求处理循环(详见 stage2)
  │ - req → FastAPI 路由 → handler → AsyncLLM.generate
  │ - → ZMQ 推到 EngineCore → 调度 → Worker 跑 model(NPU) → ZMQ 拉回
  │ - → 流式/非流式 OpenAI 格式响应
  ▼
 ⑤ 收到 SIGTERM/SIGINT
  │ - shutdown_event.set → handle_shutdown → server_task.cancel()
  │ - CancelledError → server.shutdown() → 退出 async with → AsyncLLM.shutdown()
  │ - → uvloop.run 返回 → 进程退出
```

---

## 6. 文件路径速查

| 步骤 | 文件 | 关键行 |
|---|---|---|
| 1  CLI 入口 + 子命令路由 | `vllm/entrypoints/cli/main.py` | 67 `main()` |
| 1.1  ServeSubcommand.run | `vllm/entrypoints/cli/serve.py` | `run(args)` 默认分支 |
| 2  run_server / setup_server | `vllm/entrypoints/openai/api_server.py` | 665 / 535 |
| 3  run_server_worker / build_and_serve | 同上 | 681 / 572 |
| 4  serve_http | `vllm/entrypoints/launcher.py` | 26 |
| 4.1  await server_task | 同上 | 125 |
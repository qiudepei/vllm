# vLLM 主循环代码走读(uvicorn `Server.serve`)

> 接续 [`stage1_vllm_bootstrap_walkthrough.md`](./stage1_vllm_bootstrap_walkthrough.md):在 `await server_task` 卡住之后,uvicorn 的 `Server.serve()` 内部到底干了什么?FastAPI 的 startup / shutdown 钩子在哪里触发?HTTP 请求怎么被分发到 handler?
>
> 走读范围:`uvicorn/server.py` 的 `Server.serve` / `_serve` / `startup` / `main_loop` / `shutdown`,顺带覆盖 `lifespan` 与 `protocol_factory`。
>
> 阅读时间约 8 分钟。

---

## 0. 起点:`await server_task` 把控制权交给 uvicorn

参见[阶段 1 8A 节](./stage1_vllm_bootstrap_walkthrough.md#8a-serve_http-逐行精读launcherpy71-141):

```python
server_task = loop.create_task(server.serve(sockets=[sock]))   # launcher.py:82
...
await server_task                                              # launcher.py:125  ★★★
```

主协程在 `await server_task` 处挂起;真正驱动 HTTP 接收/分发的就是 uvicorn `Server.serve` 这个 task。下面从 `serve()` 一路看下去。

---

## 1. `Server.serve` 与 `_serve`(`uvicorn/server.py:71-99`)

```python
def run(self, sockets=None):                          # L65   同步入口(uvicorn CLI 用)
    return asyncio_run(self.serve(sockets=sockets), loop_factory=self.config.get_loop_factory())

async def serve(self, sockets=None):                  # L72   vllm 走这里
    with self.capture_signals():                     # L73   装 SIGINT/SIGTERM handler
        await self._serve(sockets)                    # L74

async def _serve(self, sockets=None):                 # L76   真正的主流程
    config = self.config
    if not config.loaded: config.load()              # L79   兜底再 load 一次

    self.lifespan = config.lifespan_class(config)    # L82   FastAPI lifespan 适配器
    logger.info("Started server process [%d]", os.getpid())

    await self.startup(sockets=sockets)              # L87   ★ FastAPI startup 钩子触发点
    if not self.should_exit:                         # L88   startup 期间没被要求退出才进主循环
        await self.main_loop()                       # L89   ★★ 心跳循环
    if self.started:                                 # L90
        await self.shutdown(sockets=sockets)         # L91   ★ FastAPI shutdown 钩子触发点
```

- **`run` vs `serve`**:命令行 `uvicorn ...` 走 `run()`(用 `asyncio_run` 起新 loop);vllm 因为已经在 `uvloop.run(...)` 里,直接调用 `serve()`(拿到当前 uvloop)。
- **`capture_signals()` 上下文**(`uvicorn/server.py:341`):把 SIGINT/SIGTERM(Windows 还有 SIGBREAK)的 handler 临时替换成 `handle_exit`,退出后恢复原始 handler;信号触发只是写 `should_exit = True` / `force_exit = True`,不直接关 loop。
- **三段式 `_serve`**:`startup` → `main_loop` → `shutdown`,三段缺一不可 —— 即便 startup 失败也要尽量走到 shutdown 收尾。

---

## 2. `startup` —— 创建 asyncio.Server(`uvicorn/server.py:94-178`)

```python
async def startup(self, sockets=None):                       # L94
    await self.lifespan.startup()                           # L95   ★ FastAPI lifespan / @app.on_event("startup")
    if self.lifespan.should_exit: sys.exit(STARTUP_FAILURE)

    config = self.config

    def create_protocol(_loop=None):                        # L100  每次新连接回调,产出协议实例
        return config.http_protocol_class(
            config=config, server_state=self.server_state,
            app_state=self.lifespan.state, _loop=_loop,
        )

    loop = asyncio.get_running_loop()

    # 4 种监听方式:sockets= / fd= / uds= / host+port=
    if sockets is not None:                                 # L114  vllm 走这一支(复用 setup_server 的 sock)
        for sock in sockets:
            server = await loop.create_server(              # L126  注册到 loop,accept 自动派发
                create_protocol, sock=sock,
                ssl=config.ssl, backlog=config.backlog)
            self.servers.append(server)
    elif config.fd is not None: ...                          # L130  从 fd 接管
    elif config.uds is not None: ...                         # L138  UNIX domain socket
    else:                                                    # L152  host:port
        server = await loop.create_server(create_protocol,
            host=config.host, port=config.port,
            ssl=config.ssl, backlog=config.backlog)

    self._log_started_message(listeners)                    # L171  打 "Uvicorn running on ..."
    self.started = True                                      # L177
```

- **`lifespan.startup()` 是关键**:它会触发 FastAPI 应用上注册的所有 startup handler —— 包括用户写的 `lifespan="..."` async context manager、`@asynccontextmanager` lifespan、`router.lifespan_context` 等。vllm 通过这个机制**把 EngineCore 子进程启动、权重加载**等动作接入 HTTP server 启动流程。
- **`loop.create_server(protocol_factory, sock=...)`**:asyncio 核心 API。返回值是 `asyncio.base_events.Server`,**不是 FastAPI app**,不阻塞;它把 `sock` 注册到 loop,每次新连接 accept 后**自动回调 `create_protocol` 生成协议实例**处理该连接。
- **`protocol_factory` 设计**:每个连接一个独立 protocol 实例(H11/HttpTools/WebSocket),互不影响;协议实例负责解析 HTTP、调度到 FastAPI app、再把响应写回 socket。
- **vllm 为什么走 `sockets is not None` 分支**:因为 `setup_server` 已经 bind 好端口并把 `sock` 透传进来,这里不能再 bind 一次(对应 issue #8204 的 race condition)。

---

## 3. `main_loop` —— 心跳与退出检查(`uvicorn/server.py:221-254`)

```python
async def main_loop(self):                                  # L221
    counter = 0
    should_exit = await self.on_tick(counter)               # L223
    while not should_exit:                                  # L224
        counter += 1
        counter = counter % 864000                          # L227  防溢出(24h)
        await asyncio.sleep(0.1)                            # L228  100 ms 一跳
        should_exit = await self.on_tick(counter)           # L229

async def on_tick(self, counter):                           # L232
    if counter % 10 == 0:                                   # L234  每 1 秒一次
        self.server_state.default_headers = ...             # L235  更新 Date header(每 1 秒过期)
        if self.config.callback_notify is not None:         # L240
            if current_time - self.last_notified > ...:     # L241
                await self.config.callback_notify()         # L243

    if self.should_exit: return True                        # L247  signal handler 设的退出位
    if self.limit_max_requests is not None and \
       self.server_state.total_requests >= max_requests:    # L249  限流:满 N 个请求自杀
        return True                                         # L252
    return False
```

- **真正的 HTTP 处理不在这**:本循环只是 100 ms 心跳,HTTP 收发完全交给 `loop.create_server` 注册的 `asyncio.Server` 异步驱动;两个 while 在**同一个 uvloop 上协作,互不阻塞**。
- **`on_tick` 三件事**:① 每秒刷新 `Date` header(响应里 `Date: Tue, 15 Jul 2026 ...`);② 给 `callback_notify` 注入点留接口;③ 检查 `should_exit` 和 `limit_max_requests`。
- **`limit_max_requests`** 是优雅重启用:配合外部 supervisor,处理 N 个请求后让进程退出,supervisor 拉起新进程避免内存泄漏。
- **"卡死"的本质**:vllm 的 `await server_task` 在 `await self.main_loop()` 里;主循环一直在跑,直到 `should_exit = True` → 循环退出 → `_serve` 走到 `shutdown()`。

---

## 4. `shutdown` —— 关闭 asyncio.Server 与 lifespan(`uvicorn/server.py:271-305`)

```python
async def shutdown(self, sockets=None):                     # L271
    logger.info("Shutting down")
    for server in self.servers: server.close()              # L275  停止接受新连接
    for sock in sockets or []: sock.close()                 # L277  vllm 这里会关 setup_server 的 sock

    for connection in list(self.server_state.connections):  # L280
        connection.shutdown()                               # L281  请求各连接 graceful 关闭
    await asyncio.sleep(0.1)                                # L282  给 in-flight 请求 100 ms 收尾

    try:
        await asyncio.wait_for(                            # L285
            self._wait_tasks_to_complete(),
            timeout=self.config.timeout_graceful_shutdown,
        )
    except asyncio.TimeoutError:                            # L290
        for t in self.server_state.tasks:
            t.cancel(msg="Task cancelled, ...")             # L293  超时则暴力 cancel

    if not self.force_exit:                                 # L297
        await self.lifespan.shutdown()                      # L298  ★ FastAPI shutdown 钩子触发点
```

- **三段关闭**:
  1. **L275-277**:关 `server.close()` + 关原始 socket → 不再 accept 新连接;
  2. **L280-282**:请求所有 in-flight 连接 graceful 关闭(写完响应再 close),等 100 ms;
  3. **L285-294**:`_wait_tasks_to_complete()` 等所有 tasks 完成;超时(`timeout_graceful_shutdown`)则强制 cancel。
- **`lifespan.shutdown()` 最后调用**:触发 FastAPI 所有 shutdown handler / asynccontextmanager 退出。vllm 通过这个机制**让 EngineCore 子进程 graceful shutdown**(收尾日志、上报监控)。
- **与 `launcher.py` 的 `handle_shutdown` 对应**:launcher 的 `engine_client.shutdown()` 是在 uvicorn `lifespan.shutdown()` **之前**跑的 —— vllm 自己先关引擎再让 uvicorn 收尾,这样 in-flight 请求里的 `await engine_client.generate(...)` 能拿到明确错误,而不是被静默 cancel。

---

## 5. 整体生命周期图

```
[launcher.py: await server_task 卡住]
   │
   ▼  Server.serve() → _serve()
   │
   ├─ startup():
   │     ├─ lifespan.startup()                  ← FastAPI @app.on_event("startup")
   │     │     └─ AsyncLLM / EngineCore 启动
   │     └─ loop.create_server(create_protocol) ← accept 自动派发
   │
   ├─ main_loop():
   │     └─ while not should_exit:              ← 100ms 心跳
   │           ├─ on_tick: 更新 Date header
   │           └─ on_tick: 检查 should_exit / limit_max_requests
   │
   ├─ 收到 SIGTERM/SIGINT → handle_exit → should_exit = True
   │
   ├─ main_loop 退出
   │
   └─ shutdown():
         ├─ server.close() + sock.close()       ← 不再 accept
         ├─ connection.shutdown() × N           ← in-flight 请求 graceful 关闭
         ├─ _wait_tasks_to_complete()            ← 等所有 task(超时则 cancel)
         └─ lifespan.shutdown()                 ← FastAPI @app.on_event("shutdown")
                                                    └─ AsyncLLM / EngineCore 关闭
   │
   ▼
[_serve 返回 → serve 返回 → server_task 完成 → launcher.py except/finally]
```

---

## 6. 与 FastAPI / vllm 的衔接点

| uvicorn 阶段 | 触发 | vllm 侧响应 |
|---|---|---|
| `lifespan.startup()` | FastAPI startup | AsyncLLM 构造 + EngineCore 子进程拉起 |
| `loop.create_server(...)` | accept 新连接 | 注册协议工厂,每连接产 H11 协议实例 |
| `main_loop` 心跳 | 100ms tick | (无显式 hook;watchdog 由 vllm 自己在 launcher.py 跑) |
| `server.should_exit=True` | handle_exit (信号) | launcher 的 `handle_shutdown` 触发关引擎 |
| `lifespan.shutdown()` | FastAPI shutdown | AsyncLLM 析构 + EngineCore 子进程回收 |

---

## 7. 文件路径速查

| 步骤 | 文件 | 关键行 |
|---|---|---|
| `Server.serve` / `_serve` | `uvicorn/server.py` | 71 / 76 |
| `startup` | 同上 | 94 |
| `loop.create_server` | 同上 | 126 / 153 |
| `main_loop` | 同上 | 221 |
| `on_tick` | 同上 | 232 |
| `shutdown` | 同上 | 271 |
| `_wait_tasks_to_complete` | 同上 | 307 |
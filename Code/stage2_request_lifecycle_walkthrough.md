# vLLM 请求生命周期走读(从 HTTP 到 EngineCore)

> 接续 [`stage1_vllm_bootstrap_walkthrough.md`](./stage1_vllm_bootstrap_walkthrough.md):在 `await server_task` 把控制权交给 uvicorn 之后,一次 `POST /v1/chat/completions` 请求在 vllm 内部到底经历了什么?最后再看 vllm-ascend 是怎么用 monkey-patch 把 NPU 适配塞进这条链路的。
>
> 走读范围:`api_server.py` → `chat_completion/api_router.py` → `chat_completion/serving.py` → `v1/engine/async_llm.py` → `v1/engine/core_client.py`,最后顺路走 `vllm_ascend/patch/` 的 monkey-patch 机制。
>
> 阅读时间约 10 分钟。

---

## 0. uvicorn 在干什么(一段带过)

`await server_task` 把控制权交给 uvicorn 的 `Server.serve()`,它做的事只有三件:
1. 启动 lifespan 钩子 → 触发 FastAPI `@asynccontextmanager` lifespan(详见 `stage1`);
2. `loop.create_server(protocol_factory, sock=...)` 注册 accept 回调;
3. 进入 100 ms 心跳 `main_loop`,把 socket accept / HTTP 解析 / 协议分发全部交给 asyncio.Server 异步驱动。

**所以从 vllm 的视角看**:uvicorn 只是"接 socket → 把 HTTP 字节流喂给 FastAPI → 把 handler 的响应字节流写回 socket"的黑盒,**vllm 自己写的代码从 FastAPI 的 handler 才开始**。下面直接跳到 vllm 自己的入口。

---

## 1. `build_app` 装配路由(`api_server.py:157-307`)

```python
def build_app(args, supported_tasks, model_config):             # L157
    app = FastAPI(lifespan=lifespan)                            # L179
    app.state.args = args

    register_vllm_serve_api_routers(app)                        # L184  base 路由(tokenize/render 等)
    register_models_api_router(app)                             # L190  /v1/models
    register_sagemaker_api_router(...)                          # L196

    if "generate" in supported_tasks:
        register_generate_api_routers(app)                      # L203  ★ /v1/chat/completions 等
        attach_disagg_router(app); attach_rlhf_router(app)
        elastic_ep_attach_router(app)
        register_generative_scoring_api_router(app)             # L227

    if ...render...:  attach_render_router(app)                 # L234
    if ...transcription...: register_speech_to_text_api_routers(app)
    if ...POOLING...: register_pooling_api_routers(app)

    app.add_middleware(CORSMiddleware, ...)                     # L249
    app.add_middleware(AuthenticationMiddleware, tokens=...)    # L268  --api-key
    app.add_middleware(ScalingMiddleware)                       # L276
    for m in args.middleware: app.add_middleware(...)           # L294  用户自定义 middleware
    return app
```

- **每个 router 都是一个独立子包**(chat_completion / completion / responses / generative_scoring / models / sagemaker 等),通过 `attach_router(app)` 自挂;`build_app` 的核心就是按 `supported_tasks` 把对应 router **import + 注册**。
- **路由注册发生在 EngineCore 启动之前**:所以这里 `app.state.engine_client` 还没设;真正的注入在 `init_app_state(engine_client, app.state, ...)`(`api_server.py:310-430`)里完成,它把 `engine_client`、各个 `OpenAIServing*` handler 实例挂到 `app.state`。
- **router 全部走 `app.include_router(router)`**(`chat_completion/api_router.py:106`):FastAPI 标准做法,无 vllm 黑魔法。

---

## 2. `/v1/chat/completions` 路由(`chat_completion/api_router.py`)

```python
router = APIRouter()                                            # L28

@router.post("/v1/chat/completions", dependencies=[Depends(validate_json_request)], ...)  # L40
@with_cancellation                                              # L51  客户端断开时 abort request
@load_aware_call                                                # L52  按 server load 触发 load-aware 调度
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):  # L53
    handler = chat(raw_request)                                 # L57  从 app.state 取 handler
    if handler is None: raise NotImplementedError(...)
    generator = await handler.create_chat_completion(request, raw_request)  # L61  ★
    if isinstance(generator, ErrorResponse): return JSONResponse(...)        # L64
    if isinstance(generator, ChatCompletionResponse): return JSONResponse(...)# L68
    return StreamingResponse(content=generator, media_type="text/event-stream")  # L74
```

- **`handler = chat(raw_request)`**:`chat` 是 L32 定义的 `request.app.state.openai_serving_chat`,即上一节 `init_app_state` 注入的 `OpenAIServingChat` 单例。**这是 vllm 把"单例 handler + per-request state"分得很清的关键**。
- **三个装饰器都是 vllm 自研**:
  - `@with_cancellation`:检测到 client TCP 断开时,自动 `abort` 当前 request;
  - `@load_aware_call`:读 `app.state.server_load_metrics`,在过载时延迟或拒绝;
  - `Depends(validate_json_request)`:FastAPI 原生依赖,校验请求体大小、JSON 合法性。
- **三种返回形态**:① `ErrorResponse` → `JSONResponse`;② `ChatCompletionResponse`(非流式)→ `JSONResponse`;③ `AsyncGenerator`(流式 SSE)→ `StreamingResponse`。**流式走 `text/event-stream`**,这是 OpenAI 兼容的关键。

---

## 3. `OpenAIServingChat.create_chat_completion`(`chat_completion/serving.py:228-`)

```python
async def create_chat_completion(self, request, raw_request=None):     # L228
    return await self._with_kv_transfer_rejection_cleanup(             # L240  KV-transfer 清理钩子
        self._create_chat_completion(request, raw_request), request, raw_request)

async def _create_chat_completion(self, request, raw_request=None):   # L244
    tokenizer = self.renderer.tokenizer                                # L250
    chat_template_kwargs = ...                                          # L252
    reasoning_parser = ...                                              # L253
    # ... chat template 渲染 / 多模态处理 / tool_call 解析 ...
    generator = self.engine_client.generate(                          # ★ 调引擎
        request, raw_request,
        ...
    )
```

- **`_with_kv_transfer_rejection_cleanup`**:vllm 的 KV-transfer connector(用于 PD 分离 / disagg)钩子;请求结束时清理由 connector 加的 reject 标记,避免 KV cache 状态污染。**vllm-ascend 在 patch 里替换过这块**(见第 7 节)。
- **`self.engine_client.generate(...)`**:这一行是**整个 HTTP → 引擎的接口边界**;`engine_client` 是 `app.state.engine_client`,即 `AsyncLLM` 实例。
- **chat template 渲染**:`self.renderer` 在 `init_app_state` 里初始化,负责把 `messages: list[dict]` 渲染成 prompt 字符串 + 多模态占位;返回 `EngineCoreRequest` 给引擎。

---

## 4. `AsyncLLM.generate` 与后台 output_handler(`v1/engine/async_llm.py:524-`)

```python
async def generate(self, prompt, sampling_params, request_id, ...) -> AsyncGenerator[RequestOutput, None]:  # L524
    """
    1) Making an AsyncStream corresponding to the Request.
    2) Processing the Input.
    3) Adding the Request to the Detokenizer.
    4) Adding the Request to the EngineCore (separate process).
    """
    q: RequestOutputCollector | None = None
    try:
        q = await self.add_request(...)                                # L559  ★ 见下
        finished = False
        while not finished:
            out = q.get_nowait() or await q.get()                      # L579  drain queue
            finished = out.finished
            if out is not STREAM_FINISHED: yield out                   # L586
    except (asyncio.CancelledError, GeneratorExit):
        if q is not None: await self.abort(q.request_id, internal=True)  # L593  客户端断开 → abort
```

- **返回 `AsyncGenerator`**:不是一次性结果,而是 per-token 流;handler 拿到这个 generator 后**直接喂给 `StreamingResponse`**,实现 SSE。
- **`q = RequestOutputCollector`**:每个 request 一个 asyncio.Queue;`output_handler` 后台 task 把 EngineCore 推回的 `RequestOutput` 塞进 q,generator 从 q 里 drain。
- **`get_nowait() or await q.get()`**:先 try 非阻塞 drain,**避免 task switch**,对 TTFT / token-to-token 延迟至关重要。
- **`generate` 自己就是生成器,没有后台循环** —— 后台循环是 `_run_output_handler`(L637),所有 request 共用一个 output_handler task。

```python
async def add_request(self, request_id, prompt, params, ...):          # L280
    if self.errored: raise EngineDeadError()
    ...
    request = self.input_processor.process_inputs(                     # L349  tokenize + chat-template
        request_id, prompt, params, supported_tasks=..., ...)
    q = RequestOutputCollector(...)                                    # L379  per-request queue
    await self.detokenizer.add(request)                                #  ★ 注册 detokenizer
    if self.output_handler is None:                                    # L370
        self._run_output_handler()                                     # L373  首次启动后台 task
    await self.engine_core.add_request_async(request)                  # L412  ★★ 推到 EngineCore 子进程
    return q
```

- **`input_processor.process_inputs`**:把 prompt(prompt text / tokens / multimodal / AsyncGenerator)统一转为 `EngineCoreRequest`;这是 vllm 抽象的"输入层"。
- **`detokenizer.add`**:把 request 注册到 detokenizer 子模块(token id → 字符串的解码流水线),后续 output_handler 收到 token id 时直接 detokenize。
- **`engine_core.add_request_async`**:**跨进程边界** —— `engine_core` 是 `EngineCoreClient` 实例,通过 **ZMQ** 和 EngineCore 子进程通信。

---

## 5. 跨进程:EngineCore 子进程(`v1/engine/core_client.py`)

```python
# core_client.py: 异步客户端(API server 进程)
async def add_request_async(self, request: EngineCoreRequest) -> None:  # L217 / L1090 / L1328
    ...
    await self._send_input(EngineCoreRequestType.ADD, request)           # ZMQ PUSH

async def get_output_async(self):                                       # 持续 poll
    msg = await self._recv_output()                                     # ZMQ PULL
    return msg
```

```python
# core.py: 子进程入口
def run_engine_core(*args, dp_rank=0, local_dp_rank=0, **kwargs):       # L1093
    """EngineCore main loop (in subprocess)."""
    engine_core = EngineCoreProc(*args, **kwargs)
    while True:
        engine_core.step()                                              # ★★★ 调度主循环
```

- **架构**:`AsyncLLM` 在 API server 进程;`EngineCoreProc` 在独立子进程;两者通过 **ZMQ PUSH/PULL**(input queue + output queue)通信。**好处**:EngineCore OOM 不会拖死 API server,且子进程崩溃时 API server 可以 graceful 报错(`EngineDeadError`)。
- **`EngineCoreProc.step()`**:`while True` 调一次 step,每次做 ① 从 input queue 拿新 request;② `scheduler.schedule()` 选出要跑的 batch;③ `worker.execute_model(...)` 实际跑模型;④ 把 outputs 塞回 output queue。**这是 vllm v1 架构的核心循环**。
- **vllm-ascend 在 step 路径上有大量 patch**:调度、attention backend、kv_cache_coordinator、multiproc_executor、cudagraph 等都换成了 NPU 版本(见第 6 节)。

---

## 6. `output_handler` 后台循环(`v1/engine/async_llm.py:637-707`)

```python
def _run_output_handler(self):                                          # L637
    if self.output_handler is not None: return                         # 已启动则跳过

    async def output_handler():
        try:
            while True:
                outputs = await self.engine_core.get_output_async()     # L??  ZMQ PULL
                ...
                for output in outputs.outputs:
                    q = self.request_outputs[output.request_id]
                    q.put(RequestOutput(...))                           # 塞 per-request queue
                    if output.finished:
                        self.detokenizer.abort(...)                    # 清理
        except EngineDeadError:
            self._engine_dead = True

    self.output_handler = asyncio.create_task(output_handler())        # L707
```

- **单 output_handler 服务所有 request**:它从 EngineCore ZMQ PULL 所有 `RequestOutput`,按 `request_id` 分发到对应 per-request queue。**这就是为什么 `AsyncLLM.generate` 能用 `AsyncGenerator` 而不阻塞其他 request**。
- **`asyncio.create_task`**:output_handler 和 generate 共用 uvloop,**互不阻塞**;一个 client 慢不会卡其他 client。
- **`output_handler` vs `watchdog_loop`**:前者是"把引擎输出喂回给 HTTP",后者(`launcher.py:144`)是"定期检查引擎是否健康,死了就让 uvicorn 退出"。两个 task 角色完全不同,容易混淆。

---

## 7. vllm-ascend 的 monkey-patch 注入点(`vllm_ascend/__init__.py` + `vllm_ascend/patch/`)

```python
# vllm_ascend/__init__.py
_GLOBAL_PATCH_APPLIED = False

def _ensure_global_patch():                                            # L23
    global _GLOBAL_PATCH_APPLIED
    if _GLOBAL_PATCH_APPLIED: return
    from vllm_ascend.utils import adapt_patch
    adapt_patch(is_global_patch=True)                                  # L36  拉起 platform/* patches
    _GLOBAL_PATCH_APPLIED = True

def register():           return "vllm_ascend.platform.NPUPlatform"     # L40  vllm 通过这个 string 找到 Platform 类
def register_connector():                              _ensure_global_patch(); ...  # L46
def register_model_loader():                           _ensure_global_patch(); ...  # L56
```

```python
# vllm_ascend/utils.py:511
def adapt_patch(is_global_patch: bool = False):
    if is_global_patch:
        from vllm_ascend.patch import platform                        # 触发 platform/__init__.py
    else:
        from vllm_ascend.patch import worker                          # worker 进程启动时触发
```

```python
# vllm_ascend/patch/platform/__init__.py
import vllm_ascend.patch.platform.patch_camem_allocator               # L19
import vllm_ascend.patch.platform.patch_distributed                    # L20
import vllm_ascend.patch.platform.patch_kv_cache_interface             # L21
import vllm_ascend.patch.platform.patch_balance_schedule              # L46
...                                                                      # 25+ 个 patch_xxx.py
```

- **三阶段触发**:
  1. **Platform 注册**:`register()` 返回 `NPUPlatform` 类,被 vllm 的 `general_plugins` 机制调用(`vllm.plugins.load_general_plugins`);
  2. **Connector 注册**:`register_connector()` 拉起 `_ensure_global_patch()` → 触发 platform/* patches;
  3. **Worker 注册**:每个 worker 进程启动时(`vllm_ascend/worker/worker.py:107`)调 `adapt_patch()`(非 global)→ 触发 worker/* patches。
- **`is_global_patch=True/False` 的区别**:
  - `platform/*` 的 patch 是 **API server / EngineCore 主进程**层级生效(monkey-patch `torch.distributed`、`Scheduler`、`KVCacheInterface` 等);
  - `worker/*` 的 patch 是 **worker 子进程**层级生效(monkey-patch `MultiprocExecutor`、`patch_cudagraph` 等);**worker 子进程不能继承 platform patch 的运行时状态**,必须自己重新 import。
- **典型 patch 例子**(`patch_balance_schedule.py`):重写 `vllm.v1.engine.core.EngineCoreProc.run_engine_core` 和 `vllm.v1.core.sched.scheduler.Scheduler`,关闭 vllm 默认的 chunked-prefill,改用 ascend 的 balance scheduling。
- **patch 的"单文件一个改动"约定**:每个 `patch_xxx.py` 只动一个目标方法,文件头注释里要写清 "Why / How / Related PR / Future Plan"(`patch/__init__.py:29-1007` 全是这种自描述)。这是 vllm-ascend 跟上游同步的核心工程纪律 —— 上游某个 PR 合并了,就把对应 patch 删掉。

---

## 8. 完整请求链路图

```
[客户端 POST /v1/chat/completions]
   │
   ▼  uvicorn Server.serve()  accept → H11 protocol → FastAPI
   │
   ▼  FastAPI 路由匹配 (chat_completion/api_router.py)
   │   @with_cancellation @load_aware_call
   │   create_chat_completion(request, raw_request)
   ▼
   ▼  OpenAIServingChat.create_chat_completion (serving.py:228)
   │   ├─ chat template 渲染 (self.renderer)
   │   ├─ tool_call / reasoning_parser 处理
   │   └─ self.engine_client.generate(request, raw_request)  ← AsyncLLM
   ▼
   ▼  AsyncLLM.generate (async_llm.py:524)
   │   ├─ add_request (L280)
   │   │   ├─ input_processor.process_inputs → EngineCoreRequest
   │   │   ├─ detokenizer.add(request)               注册 detokenizer
   │   │   ├─ _run_output_handler() (L637, lazy)     首次启动后台 task
   │   │   └─ engine_core.add_request_async(request) ZMQ PUSH → EngineCore 子进程
   │   └─ 循环: yield RequestOutput from q            per-request AsyncGenerator
   ▼
   ▼  StreamingResponse(content=generator, media_type="text/event-stream")
   │   边算边把 SSE 字节流写回 socket
   ▼
   ▼  EngineCoreProc.run_engine_core (core.py:1093, 子进程)
   │   while True:
   │       ├─ engine_core.step()        scheduler.schedule() + worker.execute_model()
   │       │                              ↑
   │       │                              └─ vllm-ascend patch_balance_schedule / patch_attention
   │       └─ outputs → ZMQ PULL → AsyncLLM output_handler
   ▼
   ▼  EngineCore 子进程 OOM / 崩 → EngineDeadError → API 返回 503
```

---

## 9. vllm 关键类速查

| 角色 | 文件 | 关键方法 / 行 |
|---|---|---|
| FastAPI app 装配 | `vllm/entrypoints/openai/api_server.py` | `build_app` L157 / `init_app_state` L310 |
| Chat 路由 | `vllm/entrypoints/openai/chat_completion/api_router.py` | `create_chat_completion` L53 |
| Chat handler | `vllm/entrypoints/openai/chat_completion/serving.py` | `OpenAIServingChat.create_chat_completion` L228 |
| 引擎入口 | `vllm/v1/engine/async_llm.py` | `AsyncLLM.generate` L524 / `add_request` L280 |
| 输出后台循环 | 同上 | `_run_output_handler` L637 |
| 跨进程客户端 | `vllm/v1/engine/core_client.py` | `add_request_async` L217 |
| 引擎子进程 | `vllm/v1/engine/core.py` | `EngineCoreProc.run_engine_core` L1093 |
| ascend 入口 | `vllm_ascend/__init__.py` | `register` L40 / `_ensure_global_patch` L23 |
| ascend patch 加载 | `vllm_ascend/utils.py` | `adapt_patch` L511 |
| ascend platform patches | `vllm_ascend/patch/platform/__init__.py` | 25+ 个 `patch_xxx.py` |
| ascend worker patches | `vllm_ascend/patch/worker/__init__.py` | `patch_cudagraph` / `patch_distributed` 等 |

---

## 10. 下一步可以深入的入口

- **`InputProcessor`(`v1/engine/input_processor.py`)**:把 multimodal / prompt / token / async generator 全部归一化到 `EngineCoreRequest`;里面大概率有 ascend patch。
- **`OutputProcessor` + `Detokenizer`(`v1/engine/output_processor.py`)**:token id → 字符串 + logprob/tool_call 后处理;Chat handler 拿到 `RequestOutput` 后还会再走一次 SSE 格式化。
- **`EngineCoreProc.step()`**:调度主循环 —— scheduler、kv_cache_coordinator、multiproc_executor 的所有 ascend patch 都汇聚在 step 路径里。
- **`patch_balance_schedule.py` / `patch_multiproc_executor.py`**:两个最常改的 ascend patch,跟上游 PR 跟进最频繁,值得熟读。
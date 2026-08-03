# vLLM 请求生命周期走读(从 HTTP 到 EngineCore)

> 接续 [`stage1_vllm_bootstrap_walkthrough.md`](./stage1_vllm_bootstrap_walkthrough.md):在 `await server_task` 把控制权交给 uvicorn 之后,一次 `POST /v1/chat/completions` 请求在 vllm 内部到底经历了什么?最后再看 vllm-ascend 是怎么用 monkey-patch 把 NPU 适配塞进这条链路的。
>
> 走读范围:`api_server.py` → `chat_completion/api_router.py` → `chat_completion/serving.py` → `v1/engine/async_llm.py` → `v1/engine/core_client.py`,最后顺路走 `vllm_ascend/patch/` 的 monkey-patch 机制。
>
> 阅读时间约 10 分钟。

---

## 0. uvicorn 在干什么(一段带过)

```python
async def serve(self, sockets=None):                  # uvicorn/server.py:72
    with self.capture_signals():                     # 装 SIGINT/SIGTERM handler
        await self._serve(sockets)

async def _serve(self, sockets=None):                 # uvicorn/server.py:76
    if not config.loaded: config.load()              # 兜底再 load 一次
    self.lifespan = config.lifespan_class(config)    # FastAPI lifespan 适配器
    await self.startup(sockets=sockets)              # ★ 触发 FastAPI startup → AsyncLLM/EngineCore 启动
    if not self.should_exit:
        await self.main_loop()                       # ★★★ 100ms 心跳(SIGTERM 后退出)
    if self.started:
        await self.shutdown(sockets=sockets)         # ★ 触发 FastAPI shutdown
```

**关键**:uvicorn 只做 `lifespan.startup → accept socket → main_loop → lifespan.shutdown` 四件事;真正的 HTTP 字节流由 `loop.create_server(protocol_factory, sock=...)` 注册的回调自动驱动。**vllm 自己写的代码从 FastAPI handler 才开始**,下面直接跳到 vllm 入口。

---

## 1. `build_app` 装配路由(`api_server.py:157-307`)

```python
def build_app(args, supported_tasks, model_config):             # L157  装配 FastAPI app,按 supported_tasks 选择挂路由
    app = FastAPI(lifespan=lifespan)                            # L179  lifespan 钩子见 stage1
    app.state.args = args

    # 基础路由(无条件挂):tokenize / render / /v1/models / sagemaker
    register_vllm_serve_api_routers(app)                        # L184
    register_models_api_router(app)                             # L190
    register_sagemaker_api_router(...)                          # L196

    # 文本生成任务专属路由: /v1/chat/completions / /v1/completions / /v1/responses 都在这
    if "generate" in supported_tasks:
        register_generate_api_routers(app)                      # L203  ★
        attach_disagg_router(app); attach_rlhf_router(app)      # PD 分离 / RLHF
        elastic_ep_attach_router(app)                           # 弹性 EP
        register_generative_scoring_api_router(app)             # L227

    # 其他任务类型路由
    if ...render...:        attach_render_router(app)           # L234
    if ...transcription...: register_speech_to_text_api_routers(app)
    if ...POOLING...:       register_pooling_api_routers(app)

    # middleware 链:CORS → API key 鉴权 → 弹性扩缩 → 用户自定义
    app.add_middleware(CORSMiddleware, ...)                     # L249
    app.add_middleware(AuthenticationMiddleware, tokens=...)    # L268  --api-key
    app.add_middleware(ScalingMiddleware)                       # L276
    for m in args.middleware: app.add_middleware(...)           # L294  --middleware CLI 传入
    return app
```

**设计要点**:router 按 `supported_tasks` 条件 import 是为了**延迟加载** —— `import vllm.entrypoints.openai.chat_completion.serving` 连带 import 大半个 vllm,池化/语音类 API server 没必要为它付代价。`engine_client` 此时还没设;真正的注入在 `init_app_state(engine_client, app.state, ...)`(L310),需要等 `build_async_engine_client` 退出之后才能拿到。

---

## 2. `/v1/chat/completions` 路由(`chat_completion/api_router.py`)

```python
router = APIRouter()                                            # L28

@router.post("/v1/chat/completions",                            # L40  OpenAI 兼容路径
             dependencies=[Depends(validate_json_request)])     #       FastAPI 原生依赖:校验请求体大小/JSON
@with_cancellation                                              # L51  客户端断开时自动 abort request
@load_aware_call                                                # L52  按 server load 触发延迟或拒绝
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):  # L53
    handler = chat(raw_request)                                 # L57  从 app.state 取单例 handler
    if handler is None: raise NotImplementedError(...)

    generator = await handler.create_chat_completion(request, raw_request)  # L61  ★ 进 handler

    # 三种返回形态:ErrorResponse / 非流式 ChatCompletionResponse / 流式 AsyncGenerator
    if isinstance(generator, ErrorResponse):                                          # L64
        return JSONResponse(content=generator.model_dump(), status_code=generator.error.code)
    elif isinstance(generator, ChatCompletionResponse):                               # L68
        return JSONResponse(content=generator.model_dump(), headers=metrics_header(...))
    return StreamingResponse(content=generator, media_type="text/event-stream")       # L74  SS

def chat(request: Request) -> OpenAIServingChat | None:         # L32  app.state.openai_serving_chat 单例
    return request.app.state.openai_serving_chat
```

**设计要点**:`handler` 在 `init_app_state` 阶段构造一次,所有请求共享 —— 因为 handler 内部全是 stateless 的(tokenizer/request_logger/renderer 都在 init 时 freeze),per-request 状态都走 `request`/`raw_request` 显式传入。流式响应 `text/event-stream` 是 OpenAI 兼容协议的关键,客户端按 SSE 解析 `data: {...}\n\n` 帧。

---

## 3. `OpenAIServingChat.create_chat_completion`(`chat_completion/serving.py:228-`)

```python
async def create_chat_completion(self, request, raw_request=None):      # L228
    return await self._with_kv_transfer_rejection_cleanup(              # L240  KV-transfer 清理钩子
        self._create_chat_completion(request, raw_request), request, raw_request)

async def _create_chat_completion(self, request, raw_request=None):    # L244
    tokenizer = self.renderer.tokenizer                                 # L250
    chat_template_kwargs = self._effective_chat_template_kwargs(request)  # L252
    reasoning_parser = ...                                              # L253  Qwen3 / DeepSeek 思维链解析

    # ... chat template 渲染 / 多模态处理 / tool_call 解析 / prompt tokenize ...
    # 最后一步:调引擎
    generator = self.engine_client.generate(                            # ★ 进入 AsyncLLM
        request, raw_request,
        ...
    )
    return generator                                                    # AsyncGenerator 或 ChatCompletionResponse
```

**设计要点**:`_with_kv_transfer_rejection_cleanup` 是 vllm 给 KV-transfer connector(PD 分离)留的钩子 —— 请求结束时清理由 connector 标记的 reject,避免 KV cache 状态污染下一个 request。**vllm-ascend 的 patch_kv_cache_coordinator 会替换这块**。`engine_client.generate` 是 HTTP → 引擎的唯一接口边界;handler 内部把 chat 文本转成 `EngineCoreRequest` 之后,剩下的全是引擎的事。

---

## 4. `AsyncLLM.generate` 与后台 `output_handler`(`v1/engine/async_llm.py`)

```python
async def generate(self, prompt, sampling_params, request_id, ...):     # L524
    """
    1) Making an AsyncStream corresponding to the Request.
    2) Processing the Input.
    3) Adding the Request to the Detokenizer.
    4) Adding the Request to the EngineCore (separate process).
    """
    q: RequestOutputCollector | None = None
    try:
        q = await self.add_request(...)                                # L559  ★ 注册到引擎
        finished = False
        while not finished:
            # drain queue without await if possible (避免 task switch,TTFT 友好)
            out = q.get_nowait() or await q.get()                      # L579  per-request queue
            finished = out.finished
            if out is not STREAM_FINISHED: yield out                   # L586  流式吐出
    except (asyncio.CancelledError, GeneratorExit):                     # L591  客户端断开 → abort
        if q is not None: await self.abort(q.request_id, internal=True)  # L593
        raise
```

```python
async def add_request(self, request_id, prompt, params, ...):          # L280
    if self.errored: raise EngineDeadError()                           # L300  引擎死了直接抛

    # 1) 输入层:统一归一化成 EngineCoreRequest
    request = self.input_processor.process_inputs(                     # L349  tokenize + chat template + 多模态
        request_id, prompt, params, supported_tasks=..., ...)

    # 2) Detokenizer:注册 token-id → 字符串 解码流水线
    q = RequestOutputCollector(request_id)                             # L379  per-request asyncio.Queue
    await self.detokenizer.add(request)                                # ★ detokenizer.add

    # 3) 懒启动 output_handler(所有 request 共享一个后台 task)
    if self.output_handler is None:                                    # L370
        self._run_output_handler()                                     # L373  首次启动

    # 4) 跨进程:推到 EngineCore 子进程
    await self.engine_core.add_request_async(request)                  # L412  ★★ ZMQ PUSH
    return q
```

```python
def _run_output_handler(self):                                          # L637
    if self.output_handler is not None: return                         # 幂等:已启动则跳过

    async def output_handler():
        try:
            while True:
                outputs = await self.engine_core.get_output_async()    # ZMQ PULL EngineCore 输出
                for output in outputs.outputs:
                    q = self.request_outputs[output.request_id]        # 找到 per-request queue
                    q.put(RequestOutput(...))                          # 塞进 q,generate() 那边即可 drain
                    if output.finished:
                        self.detokenizer.abort(...)                    # 请求结束,清理 detokenizer
        except EngineDeadError:
            self._engine_dead = True                                    # 引擎崩了,后续 request 全部 503

    self.output_handler = asyncio.create_task(output_handler())        # L707  后台 task,与 generate 共用 uvloop
```

**设计要点**:`generate` 返回 `AsyncGenerator` 而不是一次性结果 —— 所以 handler 可以直接 `return StreamingResponse(content=generator, ...)`,**SSE 字节流边算边写**。`q.get_nowait() or await q.get()` 是 vllm 优化 trick:优先非阻塞 drain 避免 task switch,对 TTFT / token-to-token 延迟至关重要。**单个 `output_handler` 服务所有 request**(按 `request_id` hash 分发),这是 vllm 1 个 API server 进程能扛上千 qps 的关键 —— 所有 request 共享同一个 ZMQ PULL 循环。

---

## 5. 跨进程:EngineCore 子进程(`v1/engine/core_client.py` + `v1/engine/core.py`)

```python
# API server 进程侧:core_client.py
async def add_request_async(self, request: EngineCoreRequest) -> None:  # L217 / L1090 / L1328
    """(inproc / ZMQ / HTTP) 三种 client 都有同名方法"""
    await self._send_input(EngineCoreRequestType.ADD, request)           # ZMQ PUSH input queue

async def get_output_async(self):                                       # 持续 poll
    msg = await self._recv_output()                                     # ZMQ PULL output queue
    return msg
```

```python
# 子进程侧:core.py
def run_engine_core(*args, dp_rank=0, local_dp_rank=0, **kwargs):       # L1093
    """EngineCore main loop (in subprocess)."""
    engine_core = EngineCoreProc(*args, **kwargs)
    while True:
        engine_core.step()                                              # ★★★ 调度 + 跑模型 主循环
```

**设计要点**:`AsyncLLM` 在 API server 进程,`EngineCoreProc` 在独立子进程,两者通过 **ZMQ PUSH/PULL** 通信。**好处**:① EngineCore OOM 不会拖死 API server;② 子进程崩溃时 API server 抛 `EngineDeadError` → 503 优雅失败;③ Data Parallel 时一个 API server 进程可以管理多个 EngineCore 子进程。**`EngineCoreProc.step()` 是 vllm v1 架构的核心循环** —— 里面三件事:① 从 input queue 拿新 request;② `scheduler.schedule()` 选 batch;③ `worker.execute_model(...)` 跑模型;④ outputs 塞回 output queue。**vllm-ascend 的 patch_balance_schedule / patch_attention / patch_kv_cache_coordinator / patch_multiproc_executor 都汇聚在 step 路径上**。

---

## 6. vllm-ascend 的 monkey-patch 注入点(`vllm_ascend/__init__.py` + `vllm_ascend/patch/`)

```python
# vllm_ascend/__init__.py
_GLOBAL_PATCH_APPLIED = False

def _ensure_global_patch():                                            # L23  幂等:每个进程只跑一次
    global _GLOBAL_PATCH_APPLIED
    if _GLOBAL_PATCH_APPLIED: return
    from vllm_ascend.utils import adapt_patch
    adapt_patch(is_global_patch=True)                                  # L36  拉起 platform/* patches
    _GLOBAL_PATCH_APPLIED = True

def register():           return "vllm_ascend.platform.NPUPlatform"     # L40  vllm 通过这个 string 找到 Platform 类
def register_connector():                _ensure_global_patch(); ...    # L46  KV-transfer connector 注册
def register_model_loader():             _ensure_global_patch(); ...    # L56  weight loader 注册
```

```python
# vllm_ascend/utils.py:511
def adapt_patch(is_global_patch: bool = False):
    if is_global_patch:
        from vllm_ascend.patch import platform                        # 触发 platform/__init__.py → 25+ patch_xxx.py
    else:
        from vllm_ascend.patch import worker                          # worker 进程启动时触发
```

```python
# vllm_ascend/patch/platform/__init__.py
import vllm_ascend.patch.platform.patch_camem_allocator               # L19  内存分配器
import vllm_ascend.patch.platform.patch_distributed                    # L20  torch.distributed
import vllm_ascend.patch.platform.patch_kv_cache_interface             # L21  KV cache 抽象
import vllm_ascend.patch.platform.patch_kv_cache_utils                 # L22
import vllm_ascend.patch.platform.patch_mla_prefill_backend            # L23  MLA prefill
import vllm_ascend.patch.platform.patch_pp_mtp                         # L24  PP+MTP
# ↑ 几个无条件 import 的 platform patches

if not is_310p():                                                     # L27  按芯片型号选择性 patch
    import vllm_ascend.patch.platform.patch_mamba_config
else:
    import vllm_ascend.patch.platform.patch_mamba_config_310

# ... 十几个 model-specific / 通用 patches ...
import vllm_ascend.patch.platform.patch_balance_schedule              # L46  Scheduler 调度策略
import vllm_ascend.patch.platform.patch_kv_cache_coordinator           # L48  KV cache 协调器
import vllm_ascend.patch.platform.patch_speculative_config             # L49  投机解码配置
```

**三阶段触发**:
1. **Platform 注册**:`register()` 返回 `NPUPlatform` 类,被 vllm 的 `general_plugins` 机制调用;
2. **Connector 注册**(`register_connector`):拉起 `_ensure_global_patch()` → 触发 platform/* patches;
3. **Worker 注册**(`vllm_ascend/worker/worker.py:107`):每个 worker 子进程启动时调 `adapt_patch()`(非 global)→ 触发 worker/* patches。

**`is_global_patch=True/False` 的区别**:
- `platform/*` 在 **EngineCore 主进程** 生效(monkey-patch `Scheduler`、`KVCacheInterface`、`torch.distributed` 等);
- `worker/*` 在 **worker 子进程** 生效(monkey-patch `MultiprocExecutor`、`patch_cudagraph` 等);
- **worker 子进程不能继承 platform patch 的运行时状态**,必须自己重新 import。

**patch 的"单文件一个改动"约定**:每个 `patch_xxx.py` 只动一个目标方法,文件头注释里写清 `Why / How / Related PR / Future Plan`(`patch/__init__.py:29-1007` 全是这种自描述)。这是 vllm-ascend 跟上游同步的核心工程纪律 —— 上游某个 PR 合并了,就把对应 patch 删掉。**典型例子** `patch_balance_schedule.py`:重写 `vllm.v1.engine.core.EngineCoreProc.run_engine_core` 和 `vllm.v1.core.sched.scheduler.Scheduler`,关闭 vllm 默认的 chunked-prefill,改用 ascend 的 balance scheduling。

---

## 7. 完整请求链路图

```
[客户端 POST /v1/chat/completions]
   │  HTTP 字节流
   ▼  uvicorn Server.serve()  accept → H11 protocol → FastAPI
   │
   │  FastAPI 路由匹配 (chat_completion/api_router.py)
   │  @with_cancellation @load_aware_call
   ▼  create_chat_completion(request, raw_request)
   │
   │  OpenAIServingChat.create_chat_completion (serving.py:228)
   │  ├─ chat template 渲染 (self.renderer)
   │  ├─ tool_call / reasoning_parser 处理
   │  └─ self.engine_client.generate(request, raw_request)
   ▼
   │  AsyncLLM.generate (async_llm.py:524)
   │  ├─ add_request (L280)
   │  │  ├─ input_processor.process_inputs → EngineCoreRequest
   │  │  ├─ detokenizer.add(request)               注册 detokenizer
   │  │  ├─ _run_output_handler() (L637, lazy)     首次启动后台 task
   │  │  └─ engine_core.add_request_async(request) ZMQ PUSH → EngineCore 子进程
   │  └─ 循环: yield RequestOutput from q            per-request AsyncGenerator
   ▼
   │  StreamingResponse(content=generator, media_type="text/event-stream")
   │  边算边把 SSE 字节流写回 socket
   ▼
   │  EngineCoreProc.run_engine_core (core.py:1093, 子进程)
   │  while True:
   │      ├─ engine_core.step()        scheduler.schedule() + worker.execute_model()
   │      │                              ↑
   │      │                              └─ vllm-ascend patch: balance_schedule / attention / kv_cache
   │      └─ outputs → ZMQ PULL → AsyncLLM output_handler
   ▼
   │  EngineCore 子进程 OOM / 崩 → EngineDeadError → API 返回 503
   ▼
[客户端收到 SSE 流 data: {...}\n\n 帧]
```

---

## 8. vllm 关键类速查

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

## 9. 下一步可以深入的入口

- **`InputProcessor`(`v1/engine/input_processor.py`)**:把 multimodal / prompt / token / async generator 全部归一化到 `EngineCoreRequest`;里面大概率有 ascend patch。
- **`OutputProcessor` + `Detokenizer`(`v1/engine/output_processor.py`)**:token id → 字符串 + logprob/tool_call 后处理;Chat handler 拿到 `RequestOutput` 后还会再走一次 SSE 格式化。
- **`EngineCoreProc.step()`**:调度主循环 —— scheduler、kv_cache_coordinator、multiproc_executor 的所有 ascend patch 都汇聚在 step 路径里。
- **`patch_balance_schedule.py` / `patch_multiproc_executor.py`**:两个最常改的 ascend patch,跟上游 PR 跟进最频繁,值得熟读。
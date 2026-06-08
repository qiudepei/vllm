# vllm-ascend 请求处理全生命周期分析

> 基于 vLLM v1 架构 + vllm-ascend 插件，分析一个 HTTP 请求从进入到输出 Token 的完整路径。
> 涉及三进程架构：**主进程(API Server + AsyncLLM)** ↔ **EngineCore 子进程(调度)** ↔ **Worker 子进程(模型推理)**

---

## 目录

1. [总体架构概览](#1-总体架构概览)
2. [Phase 1: 服务启动与 API 路由注册](#2-phase-1-服务启动与-api-路由注册)
3. [Phase 2: AsyncLLM 引擎前端](#3-phase-2-asyncllm-引擎前端)
4. [Phase 3: EngineCore 调度循环](#4-phase-3-enginecore-调度循环)
5. [Phase 4: Scheduler.schedule() 调度决策](#5-phase-4-schedulerschedule-调度决策)
6. [Phase 5: Worker 模型执行](#6-phase-5-worker-模型执行)
7. [Phase 6: AscendSampler 采样](#7-phase-6-ascendsampler-采样)
8. [Phase 7: update_from_output 输出处理](#8-phase-7-update_from_output-输出处理)
9. [Phase 8: 输出回传与流式响应](#9-phase-8-输出回传与流式响应)
10. [Ascend 特有调度器补丁](#10-ascend-特有调度器补丁)
11. [完整流程图](#11-完整流程图)

---

## 1. 总体架构概览

vLLM v1 使用 **三进程架构**，通过 ZMQ 进行进程间通信：

```
┌──────────────────────────────────────────────────────────────────┐
│                        主进程 (API Server)                        │
│  ┌──────────────┐    ┌────────────────┐    ┌──────────────────┐  │
│  │ FastAPI       │───▶│ OpenAIServing  │───▶│   AsyncLLM       │  │
│  │ (api_server)  │    │ Chat/Completion│    │ (EngineClient)   │  │
│  └──────────────┘    └────────────────┘    └────────┬─────────┘  │
│                                                      │ ZMQ        │
├──────────────────────────────────────────────────────┼───────────┤
│                   EngineCore 子进程                    │           │
│  ┌───────────────────────────────────────────────────▼────────┐  │
│  │                 EngineCoreProc                              │  │
│  │  ┌──────────────┐  ┌────────────┐  ┌───────────────────┐   │  │
│  │  │ input_queue   │  │ Scheduler  │  │ model_executor    │   │  │
│  │  │ processing    │─▶│ .schedule()│─▶│ (MultiprocExec)   │   │  │
│  │  └──────────────┘  └────────────┘  └─────────┬─────────┘   │  │
│  └──────────────────────────────────────────────┼──────────────┘  │
│                                                   │               │
├───────────────────────────────────────────────────┼───────────────┤
│                 Worker 子进程                       │               │
│  ┌────────────────────────────────────────────────▼──────────┐   │
│  │                     NPUWorker                              │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌─────────────┐  │   │
│  │  │ NPUModelRunner  │─▶│ AscendAttention│─▶│ Ascend       │  │   │
│  │  │ .execute_model() │  │ Backend        │  │ Sampler     │  │   │
│  │  └────────────────┘  └────────────────┘  └─────────────┘  │   │
│  └────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

### 模块功能总览

| 模块 | 进程 | 职责 |
|------|------|------|
| **FastAPI / APIRouter** | 主进程 | HTTP 服务器，接收请求、按 URL 路由分发、中间件处理（CORS/auth）、返回响应 |
| **OpenAIServingChat** | 主进程 | Chat Completion 接口处理器：解析请求、渲染 chat template、调引擎、拼 SSE 流 |
| **OpenAIServingCompletion** | 主进程 | Completion 接口处理器：纯文本补全（无多轮对话） |
| **OpenAIServingRender** | 主进程 | 渲染器：prompt tokenize、chat template 拼接、多模态输入处理 |
| **OpenAIServingModels** | 主进程 | 模型注册中心：管理模型列表、LoRA 适配器、返回 `/v1/models` 数据 |
| **InputProcessor** | 主进程 | 输入处理器：把 prompt + 参数转成引擎内部标准格式 `EngineCoreRequest` |
| **OutputProcessor** | 主进程 | 输出处理器：把引擎输出 token IDs 还原为文字（detokenize），包装为 `RequestOutput` |
| **AsyncLLM** | 主进程 | 引擎客户端：实现 `EngineClient` 接口，对外提供 `generate()` / `abort()` / `add_request()`，对内协调 InputProcessor / OutputProcessor / EngineCoreClient |
| **EngineCoreClient** | 主进程 | ZMQ 通信客户端：通过 ZMQ ROUTER-DEALER 与 EngineCore 子进程通信，发送请求/接收输出 |
| **EngineCore / EngineCoreProc** | 子进程 | 引擎核心：运行 `run_busy_loop()` 主循环，驱动调度→执行→输出流水线 |
| **Scheduler** | 子进程 | 调度器：管理请求队列（waiting/running/skipped），KV cache 分配与释放，preemption，每步产生 `SchedulerOutput` |
| **KVCacheManager** | 子进程 | KV cache 管理器：block 分配/释放、prefix cache 哈希匹配、block 池管理 |
| **MultiprocExecutor** | 子进程 | 模型执行器：把调度输出分发给 Worker 多进程，收集模型输出（支持 PP/TP 通信） |
| **NPUWorker** | 子进程 | Ascend Worker：持有一块 NPU 设备，调用 ModelRunner 执行模型 forward，处理 PP 通信 |
| **NPUModelRunner** | 子进程 | Ascend 模型运行器：准备输入 batch、构建 attention metadata、调 model forward、返回 logits |
| **AscendAttentionBackend** | 子进程 | Ascend Attention 后端：实现 PagedAttention / MLA / SFA / DSA 等注意力算法 |
| **AscendSampler** | 子进程 | Ascend 采样器：logits → token IDs，支持贪心/随机/top-k/top-p，避免 NPU-CPU 同步 |
| **Request** | 子进程 | 请求内部状态：记录 prompt/output tokens、computed tokens 数、状态（WAITING/RUNNING/PREEMPTED 等） |
| **RequestOutputCollector** | 主进程 | 单请求输出队列：asyncio.Event 驱动的生产者-消费者队列，output_handler 写入、generate() 读取 |
| **IncrementalDetokenizer** | 主进程 | 增量解码器：逐 token 解码，维护 byte 缓冲区，支持跨 token 的 utf-8 字符拼接 |

---

## 2. Phase 1: 服务启动与 API 路由注册

### 2.1 CLI 入口：`vllm serve`

> **`main.py`**：`vllm` 命令的总入口。使用 `argparse` 注册多个子命令（serve、benchmark、launch、collect_env 等），每个子命令对应一个模块。`main()` 解析命令行参数后调用对应子命令的 `cmd()` 函数。
>
> **`serve.py`**：`vllm serve` 子命令的实现。`ServeSubcommand.cmd()` 处理各种部署模式（单 API server、多 API server、headless、gRPC 等），最终在单 server 模式下调用 `uvloop.run(run_server(args))` 启动服务。

```python
# main.py:17 — CLI 入口
def main():
    import vllm.entrypoints.cli.serve          # 延迟加载
    CMD_MODULES = [
        vllm.entrypoints.cli.serve,            # serve 子命令
        # ...
    ]
    # 注册所有子命令的 subparser
    for cmd_module in CMD_MODULES:
        new_cmds = cmd_module.cmd_init()       # ServeSubcommand()
        for cmd in new_cmds:
            cmd.subparser_init(subparsers)
            cmds[cmd.name] = cmd

    args = parser.parse_args()
    args.dispatch_function(args)               # → ServeSubcommand.cmd(args)

# serve.py:50 — ServeSubcommand
class ServeSubcommand(CLISubcommand):
    name = "serve"
    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        # ... 单 API server 模式 ...
        uvloop.run(run_server(args))           # ← 最终入口
```

### 2.2 `run_server()` → `run_server_worker()` → 构建引擎

> **`api_server.py`**：整个服务的总控。`run_server()` 先调用 `setup_server()` 创建 TCP socket（或 UDS），避免端口竞争。然后 `run_server_worker()` 内部调用 `build_async_engine_client()` 构建引擎，再调用 `build_and_serve()` 启动 HTTP 服务。
>
> **`build_async_engine_client()`**：异步上下文管理器，创建 `AsyncLLM` 实例（含 EngineCore 子进程的启动）。使用 `AsyncEngineArgs.from_cli_args()` 解析配置，然后调用 `AsyncLLM.from_vllm_config()` 创建引擎。
>
> **`AsyncLLM` 构建过程**：`from_vllm_config()` → `__init__()` 创建 InputProcessor、OutputProcessor，然后调用 `EngineCoreClient.make_async_mp_client()` 启动 EngineCore 子进程（通过 ZMQ + multiprocessing.Process），并建立 ZMQ 通信管道。

```python
# api_server.py:650
async def run_server(args, **uvicorn_kwargs):
    listen_address, sock = setup_server(args)          # 先创建 socket 避免端口竞争
    await run_server_worker(listen_address, sock, args)

# api_server.py:666
async def run_server_worker(listen_address, sock, args, client_config=None):
    async with build_async_engine_client(args) as engine_client:   # ← 构建引擎
        shutdown_task = await build_and_serve(
            engine_client, listen_address, sock, args
        )
        await shutdown_task
```

**构建引擎的完整链路：**

```
build_async_engine_client(args)
  └─ build_async_engine_client_from_engine_args(engine_args)
      └─ engine_args.create_engine_config() → VllmConfig
      └─ AsyncLLM.from_vllm_config(vllm_config)
          └─ AsyncLLM.__init__()
              ├─ InputProcessor(self.vllm_config, renderer)
              ├─ OutputProcessor(tokenizer, ...)
              └─ EngineCoreClient.make_async_mp_client(...)
                  └─ MPClient.__init__()
                      ├─ ZMQ socket 创建 (ROUTER + PULL)
                      ├─ launch_core_engines()
                      │   └─ CoreEngineProcManager()
                      │       └─ multiprocessing.Process(
                      │            target=EngineCoreProc.run_engine_core
                      │          ).start()
                      │           └─ EngineCoreProc.__init__()
                      │               ├─ ZMQ handshake w/ frontend
                      │               └─ EngineCore.__init__()
                      │                   ├─ model_executor
                      │                   ├─ _initialize_kv_caches()
                      │                   └─ Scheduler(...)
                      ├─ 等待所有 EngineCore 发送 READY
                      └─ start_engine_core_monitor()
```

**`core_client.py` —— EngineCoreClient 实现**：

`MPClient.__init__()` 详细流程：
1. 创建 ZMQ context（同步 + 异步）
2. 绑定 ROUTER socket（input，收前端请求）和 PULL socket（output，收引擎输出）
3. 调用 `launch_core_engines()`：创建 `CoreEngineProcManager`，启动 `multiprocessing.Process` 运行 `EngineCoreProc.run_engine_core`
4. 等待所有 EngineCore 子进程发送 READY 握手消息（包含 max_model_len、num_gpu_blocks 等同步）
5. 启动后台监控线程 `start_engine_core_monitor()` 监控子进程存活

**`utils.py` —— `launch_core_engines()` 和 `CoreEngineProcManager`**：

`CoreEngineProcManager.__init__()` 创建 N 个 `multiprocessing.Process`，每个进程执行 `EngineCoreProc.run_engine_core()`（带 dp_rank / local_dp_rank 参数）。`wait_for_engine_startup()` 等待所有引擎发回 READY。

### 2.3 `build_and_serve()` → 构建 FastAPI app + 初始化状态

> **`build_and_serve()`**：引擎已就绪，开始搭建 HTTP 服务。先调用 `engine_client.get_supported_tasks()` 获取引擎支持的任务类型（generate/pooling/embedding 等），然后调 `build_app()` 创建 FastAPI 应用并注册路由，再调 `init_app_state()` 初始化 Serving 层组件，最后 `serve_http()` 启动 uvicorn。

```python
# api_server.py:557
async def build_and_serve(engine_client, listen_address, sock, args, **uvicorn_kwargs):
    supported_tasks = await engine_client.get_supported_tasks()
    model_config = engine_client.model_config

    app = build_app(args, supported_tasks, model_config)       # ← 构建 FastAPI + 路由
    await init_app_state(engine_client, app.state, args, supported_tasks)  # ← 初始化 serving 组件

    return await serve_http(app, sock=sock, ...)               # ← 启动 uvicorn
```

### 2.4 `build_app()` — 路由注册

> **`FastAPI`**：Python ASGI Web 框架，vLLM 用它构建 REST API。`build_app()` 创建 `FastAPI()` 实例，然后通过多组 `app.include_router(router)` 注册路由。支持 lifespan（后台日志统计）、CORS 中间件、认证中间件等。
>
> **`APIRouter`**：FastAPI 的路由分组机制。vLLM 把不同类别的 API 放在不同 router 中（chat、completion、models、管理类等），最后统一挂载到主 app。每个 router 上通过 `@router.post()` / `@router.get()` 装饰器注册具体端点。
>
> **`register_vllm_serve_api_routers()`**：注册服务管理类路由，包括 Prometheus metrics、LoRA 管理（CRUD）、Profiling 启停、Tokenize/Detokenize。
>
> **`register_generate_api_routers()`**：注册生成类 API 路由，是核心入口。内部注册 chat、completion、responses、Anthropic 兼容接口、评分接口的子 router。
>
> **`register_models_api_router()`**：注册 `GET /v1/models` 端点，用于查询可用模型列表。

**文件**: `vllm/entrypoints/openai/api_server.py:155`

```python
def build_app(args, supported_tasks=None, model_config=None) -> FastAPI:
    app = FastAPI(lifespan=lifespan)

    # ─── 服务管理路由 ───
    register_vllm_serve_api_routers(app)
    #   register_instrumentator_api_routers()  → Prometheus /metrics
    #   attach_lora_router()                   → LoRA CRUD
    #   attach_profile_router()                → profile start/stop
    #   attach_tokenize_router()               → /tokenize, /detokenize

    # ─── 模型列表 ───
    register_models_api_router(app)
    #   GET /v1/models  → show_available_models()

    # ─── 生成类 API (核心) ───
    register_generate_api_routers(app)
    #   register_chat_api_router(app)
    #     → POST /v1/chat/completions
    #     → POST /v1/chat/completions/batch
    #   register_responses_api_router(app)
    #     → POST /v1/responses
    #   register_completion_api_router(app)
    #     → POST /v1/completions
    #   register_anthropic_api_router(app)
    #     → 兼容 Anthropic 格式
    #   register_generative_scoring_api_router(app)

    # ─── 其他 ───
    attach_disagg_router(app)
    elastic_ep_attach_router(app)
    attach_render_router(app)
    register_speech_to_text_api_routers(app)
    register_pooling_api_routers(app)
```

**Chat Completion 路由器的具体定义：**

> **`api_router.py`（chat_completion）**：定义 `POST /v1/chat/completions` 端点。`create_chat_completion()` 从 `app.state` 获取 `OpenAIServingChat` 实例，调 `handler.create_chat_completion()` 处理。`attach_router()` 将 router 挂载到 FastAPI app。

```python
# vllm/entrypoints/openai/chat_completion/api_router.py:28
router = APIRouter()

# line 40
@router.post("/v1/chat/completions", ...)
async def create_chat_completion(
    request: ChatCompletionRequest,
    raw_request: Request,
):
    handler = chat(raw_request)  # 从 app.state.openai_serving_chat 获取
    generator = await handler.create_chat_completion(request, raw_request)

# line 105
def attach_router(app: FastAPI):
    app.include_router(router)
```

### 2.5 `init_app_state()` — Serving 层初始化

> **`OpenAIServingModels`**：模型注册中心。管理 served model 列表、LoRA 适配器（init_static_loras）、为其他 Serving 组件提供模型信息（model_name、model_config 等）。`GET /v1/models` 的查询由此提供数据。
>
> **`OpenAIServingRender`**：渲染器，将用户请求转换为引擎输入。核心功能：① tokenize（文本 → token IDs）；② chat template 渲染（多轮对话 + 模板拼接）；③ 多模态输入处理（图片→image feature）；④ prompt logprobs 计算。内部持有 `Renderer` 实例（内含 tokenizer、chat template、多模态处理器）。
>
> **`OpenAIServingTokenization`**：独立的 tokenize/detokenize 服务，用于 `/tokenize` 和 `/detokenize` 端点。
>
> **`OpenAIServingChat`**：Chat Completion 接口的完整实现。`_create_chat_completion()` 方法的工作流：① `render_chat_request()` → 调用 OpenAIServingRender 渲染请求；② 构造 `SamplingParams`；③ `engine_client.generate()` → 调用 AsyncLLM 生成；④ 根据 stream 模式选择流式或非流式响应生成器。内部持有 engine_client、models、render 等依赖。
>
> **`OpenAIServingChatBatch`**：Chat Completion Batch 接口，处理批量请求（`/v1/chat/completions/batch`）。
>
> **`OpenAIServingCompletion`**：Completion 接口处理器，用于 `/v1/completions` 端点。与 Chat 不同，它处理纯文本（无多轮对话结构和 tool calls），直接对 prompt 做补全。

```python
# api_server.py:301
async def init_app_state(engine_client, state, args, supported_tasks=None):
    state.engine_client = engine_client

    # 1. 模型注册
    state.openai_serving_models = OpenAIServingModels(
        engine_client, base_model_paths, lora_modules, ...
    )

    # 2. 渲染器（chat template, tokenizer）
    state.openai_serving_render = OpenAIServingRender(
        model_config, renderer=engine_client.renderer, ...
    )

    # 3. Tokenization 服务
    state.openai_serving_tokenization = OpenAIServingTokenization(
        engine_client, models, render, ...
    )

    # 4. 生成类服务
    await init_generate_state(engine_client, state, args, ...)
    # 内部创建:
    #   state.openai_serving_chat     = OpenAIServingChat(engine_client, ...)
    #   state.openai_serving_completion = OpenAIServingCompletion(engine_client, ...)
    #   state.anthropic_serving_messages = AnthropicServingMessages(...)
    #   state.serving_tokens = ServingTokens(...)
    #   state.serving_generative_scoring = ServingGenerativeScoring(...)
```

### 2.6 启动 uvicorn

> **`uvicorn`**：Python ASGI HTTP 服务器，监听 TCP 端口、接收 HTTP 请求、解析后转给 FastAPI 应用。`serve_http(app, sock=sock)` 使用已创建的 socket 启动，避免端口竞争。

```python
# api_server.py:602
return await serve_http(app, sock=sock, ...)
```

此时完整的调用链为：

```
vllm serve <model>
  └─ main() → ServeSubcommand.cmd()
      └─ run_server(args)
          ├─ setup_server()               # 创建 socket
          └─ run_server_worker()
              ├─ build_async_engine_client()
              │   └─ AsyncLLM.from_vllm_config()
              │       └─ EngineCoreClient → 启动 EngineCore 子进程
              └─ build_and_serve(engine_client)
                  ├─ build_app()           # 注册所有路由
                  ├─ init_app_state()      # 创建 Serving 组件
                  └─ serve_http()          # 启动 uvicorn
```

### 2.7 HTTP 请求进来后的处理

> **`ChatCompletionRequest`**：OpenAI 兼容的请求协议类（Pydantic 模型），字段包括 `messages`（对话历史）、`model`、`temperature`、`max_tokens` / `max_completion_tokens`、`stream`、`top_p`、`frequency_penalty`、`presence_penalty`、`tools`、`tool_choice`、`stop`、`priority` 等。FastAPI 自动将 JSON body 反序列化为该对象。
>
> **`SamplingParams`**：引擎内部使用的采样参数，从 `ChatCompletionRequest` 转换而来。关键字段：`temperature`（温度，控制随机性）、`top_p`（核采样累积概率阈值）、`top_k`（只从前 k 个 token 采样）、`max_tokens`（最大生成数）、`stop`（停止字符串列表）、`frequency_penalty` / `presence_penalty` / `repetition_penalty`（三种惩罚系数）、`n`（每个 prompt 生成几个候选）、`best_of`、`seed`、`output_kind`（流式模式：逐 token / 累积 / 仅最终）。
>
> **`chat_completion_stream_generator()`**：流式响应生成器。接收 `AsyncGenerator[RequestOutput]`，将每次产出的 `RequestOutput` 转为 OpenAI 格式的 SSE 事件。首个 chunk 包含 `role` delta，中间 chunk 为 `delta.content`（含 reasoning_content 思考链），末位 chunk 含 `finish_reason`（stop/length）和 `usage` 统计。事件格式：`data: {json}\n\n`，结束时发送 `data: [DONE]\n\n`。
>
> **`chat_completion_full_generator()`**：非流式响应生成器。等待引擎生成全部 token 后，组装成完整的 `ChatCompletionResponse` JSON，包含 `choices[].message.content`、`usage.prompt_tokens`、`usage.completion_tokens`、`usage.total_tokens` 等。

**文件**: `vllm/entrypoints/openai/chat_completion/serving.py`

```python
# line 235
async def _create_chat_completion(self, request, raw_request=None):
    # 1. 渲染 chat request: 分词 + chat template
    result = await self.render_chat_request(request)  # line 250
    conversation, engine_inputs = result

    # 2. 构造 SamplingParams
    sampling_params = request.to_sampling_params(
        max_tokens, self.default_sampling_params
    )  # line 300

    # 3. 调用 engine_client.generate()  → AsyncLLM.generate()
    generator = self.engine_client.generate(
        engine_input, sampling_params, sub_request_id, ...
    )  # line 341

    # 4. 根据 stream 模式返回流式或完整响应
    if request.stream:
        return self.chat_completion_stream_generator(...)  # line 371
    return await self.chat_completion_full_generator(...)   # line 383
```

---

## 3. Phase 2: AsyncLLM 引擎前端

> **`AsyncLLM`**：实现 `EngineClient` 接口，是主进程与 EngineCore 子进程之间的桥梁。对外提供 `generate()`、`add_request()`、`abort()` 等 API。内部协调三个核心组件：`InputProcessor`（输入处理）、`OutputProcessor`（输出处理）、`engine_core`（EngineCoreClient，ZMQ 通信）。

### 3.1 初始化

> **`InputProcessor`**：输入处理器，将 Serving 层传来的原始 prompt（文本/token IDs/多模态）和采样参数转换为引擎内部的 `EngineCoreRequest`。核心方法 `process_inputs()` 执行 tokenize、多模态特征提取（图像→image features 等）、参数校验、请求 ID 分配。输出是标准化的 `EngineCoreRequest`，可通过 ZMQ 序列化传输。
>
> **`OutputProcessor`**：输出处理器，与 InputProcessor 互为逆过程。将 EngineCore 返回的 `EngineCoreOutput`（token IDs）还原为 `RequestOutput`（文字 + logprobs）。内部维护每个请求的 `RequestState`（含 `IncrementalDetokenizer` 和 `LogprobsProcessor`），支持流式增量解码。核心方法 `process_outputs()` 被 `output_handler` 后台任务持续调用。
>
> **`EngineCoreClient`**：主进程与 EngineCore 子进程的 IPC 通信。`make_async_mp_client()` 工厂方法根据配置创建不同实现：`AsyncMPClient`（单引擎 ZMQ）、`DPLBAsyncMPClient`（DP 内部负载均衡）、`DPAsyncMPClient`（DP 外部负载均衡）。内部使用 ZMQ ROUTER-DEALER 模式，通过 `msgspec` 做序列化。

```python
# line 73
class AsyncLLM(EngineClient):
    def __init__(self, vllm_config, executor_class, ...):
        # 输入处理器：EngineInput → EngineCoreRequest
        self.input_processor = InputProcessor(self.vllm_config, renderer)  # line 135

        # 输出处理器：EngineCoreOutputs → RequestOutput
        self.output_processor = OutputProcessor(...)  # line 138

        # EngineCore 客户端：通过 ZMQ 与子进程通信
        self.engine_core = EngineCoreClient.make_async_mp_client(...)  # line 146
```

### 3.2 generate() — 请求入口

> **`generate()`**：Serving 层调用的主入口。它是一个异步生成器（`AsyncGenerator[RequestOutput]`），调用方通过 `async for` 逐条消费输出。方法内部先调 `add_request()` 将请求加入引擎，然后进入 while 循环从 `RequestOutputCollector` 中取结果，每次 `await q.get()` 阻塞直到新输出到达。请求被客户端取消时（`CancelledError` / `GeneratorExit`），自动调用 `abort()` 终止引擎侧请求。

```python
# line 524
async def generate(self, prompt, sampling_params, request_id, ...):
    q = await self.add_request(request_id, prompt, sampling_params, ...)  # line 559

    finished = False
    while not finished:
        out = q.get_nowait() or await q.get()  # line 579
        finished = out.finished
        if out is not STREAM_FINISHED:
            yield out
```

### 3.3 add_request() — 请求入队

> **`add_request()`**：完整的请求入队流程。先调 `InputProcessor.process_inputs()` 将 prompt 转为 `EngineCoreRequest`。然后创建 `RequestOutputCollector`（asyncio.Event 驱动的单请求队列）。最后调 `_add_request()` 同时注册到 OutputProcessor（本进程）和 EngineCore（ZMQ 发到子进程）。
>
> **`EngineCoreRequest`**：通过 ZMQ 传输的请求体（msgspec.Struct），包含 `request_id`、`prompt_token_ids`、`sampling_params`、`mm_features`（多模态特征）、`lora_request`、`priority`、`arrival_time` 等字段。`InputProcessor.process_inputs()` 生产它。
>
> **`RequestOutputCollector`**：每个请求一个实例，是 `output_handler`（生产者）和 `generate()` 协程（消费者）之间的队列。通过 `asyncio.Event` 实现非阻塞 put / 阻塞 get。支持 DELTA 模式（增量聚合多个输出）和 FINAL_ONLY 模式。

```python
# line 280
async def add_request(self, request_id, prompt, params, ...):
    # 1. 处理输入：prompt + params → EngineCoreRequest
    request = self.input_processor.process_inputs(...)  # line 349

    # 2. 创建输出收集器
    queue = RequestOutputCollector(params.output_kind, request.request_id)  # line 376

    # 3. 添加请求
    await self._add_request(request, prompt_text, None, 0, queue)  # line 382
    return queue

async def _add_request(self, request, prompt, parent_req, index, queue):
    # 3a. 注册到 OutputProcessor（本进程）
    self.output_processor.add_request(request, prompt, parent_req, index, queue)  # line 409
    # 3b. 通过 ZMQ 发送到 EngineCore 子进程
    await self.engine_core.add_request_async(request)  # line 412
```

### 3.4 output_handler — 后台输出循环

> **`output_handler`**：AsyncLLM 启动的一个后台 asyncio Task，持续运行直到引擎关闭。循环工作：① `engine_core.get_output_async()` —— 从 ZMQ PULL socket 接收 EngineCore 产出的 `EngineCoreOutputs`；② `output_processor.process_outputs()` —— 将 token IDs 还原为文字，包装成 `RequestOutput`，推送到对应请求的 `RequestOutputCollector`；③ 处理 stop string 触发的 abort（通过 `engine_core.abort_requests_async()`）；④ 日志统计。
>
> **`EngineCoreOutputs`**：引擎一次 step 产生的所有输出集合。包含 `outputs: list[EngineCoreOutput]`（每个被调度请求的 output）、`scheduler_stats: SchedulerStats`（调度统计）、`finished_requests: set[str]`（已完成的请求 ID）。

```python
# line 637
def _run_output_handler(self):
    async def output_handler():
        while True:
            # 1) 从 EngineCore 拉取输出
            outputs = await engine_core.get_output_async()  # line 660

            # 2) 处理 EngineCoreOutputs → RequestOutput
            processed_outputs = output_processor.process_outputs(
                outputs_slice, outputs.timestamp, iteration_stats
            )  # line 675

            # 3) 处理 stop string 触发的 abort
            if processed_outputs.reqs_to_abort:
                await engine_core.abort_requests_async(
                    processed_outputs.reqs_to_abort
                )

            # 4) 日志记录
            if logger_ref[0]:
                logger_ref[0].record(...)

    self.output_handler = asyncio.create_task(output_handler())
```

---

## 4. Phase 3: EngineCore 调度循环

> **`EngineCore`**：引擎核心类，运行在独立子进程中。管理 Scheduler（调度器）、model_executor（模型执行器）、KV cache 初始化、结构化输出管理器等。提供 `step()` / `step_with_batch_queue()` 方法驱动每次调度-执行的迭代。
>
> **`EngineCoreProc`**：EngineCore 的 ZMQ 包装类，继承自 EngineCore。增加了 ZMQ socket 通信层，通过 `run_busy_loop()` 主循环驱动引擎。包含 `input_queue`（ZMQ 输入线程写入，busy loop 消费）和 `output_queue`（busy loop 写入，ZMQ 输出线程发送）。

### 4.1 EngineCoreProc.run_busy_loop() — 主循环

> **`run_busy_loop()`**：EngineCore 子进程的主循环。无限重复：① `_process_input_queue()` —— 处理主进程发来的请求（ADD / ABORT / UTILITY）；② `_process_engine_step()` —— 执行一次引擎迭代（调度+执行+输出）。通过 `_handle_shutdown()` 检查退出信号。当无请求时，`_process_input_queue()` 会阻塞等待，不空转 CPU。

```python
# line 1216
def run_busy_loop(self):
    while self._handle_shutdown():
        # 1) 处理输入队列（add/abort/utility 请求）
        self._process_input_queue()
        # 2) 执行引擎 step
        self._process_engine_step()
```

### 4.2 _process_input_queue() — 输入队列处理

> **`_process_input_queue()`**：处理主进程发来的各类 IPC 请求。实际分发在 `_handle_client_request()`：`ADD` 类型调用 `EngineCore.add_request()`（预处理 Request 后交给 Scheduler），`ABORT` 类型调用 `abort_requests()`，`UTILITY` 类型根据方法名动态调用（如 reset_prefix_cache、pause_scheduler 等）。
>
> **`_handle_client_request()`**：根据 `EngineCoreRequestType` 枚举分发请求：`ADD`（新请求）、`ABORT`（取消请求）、`UTILITY`（管理命令如 pause/resume）、`WAKEUP`（唤醒空闲引擎）、`EXECUTOR_FAILED`（Worker 进程崩溃）。

```python
# line 1226
def _process_input_queue(self):
    while not self.has_work() and self.is_running():
        self._notify_idle_state_callbacks()
        # ... drain aborts queue ...
        req = self.input_queue.get(block=block)
        self._handle_client_request(*req)

# line 1317
def _handle_client_request(self, request_type, request):
    if request_type == EngineCoreRequestType.ADD:
        req, request_wave = request
        self.add_request(req, request_wave)       # → scheduler.add_request()
    elif request_type == EngineCoreRequestType.ABORT:
        self.abort_requests(request)               # → scheduler.finish_requests()
    elif request_type == EngineCoreRequestType.UTILITY:
        self._invoke_utility_method(...)
```

### 4.3 EngineCore.add_request() — 请求入调度器

> **`preprocess_add_request()`**：将 ZMQ 反序列化的 `EngineCoreRequest` 转为 Scheduler 内部使用的 `Request` 对象（含 block hash 初始化）。如果需要结构化输出（grammar/JSON schema），在此初始化 grammar 编译器。返回 `(Request, request_wave)`，后者用于 DP 场景标识请求批次。

```python
# line 341
def add_request(self, request, request_wave=0):
    req, request_wave = self.preprocess_add_request(request)
    self.scheduler.add_request(req)  # → 进入 waiting 队列
```

### 4.4 core.step() — 调度执行核心

> **`step()`**：EngineCore 每次迭代的核心。串行执行：① `scheduler.schedule()` 产生 `SchedulerOutput`（决定本轮谁跑多少 token）；② `model_executor.execute_model()` 将调度输出发给 Worker 进程执行（异步 non_block，返回 Future）；③ `scheduler.get_grammar_bitmask()` 获取结构化输出的 grammar 约束（如果有）；④ 等待模型执行完毕 `future.result()`，若输出为 None 则调用 `sample_tokens()` 采样；⑤ 处理 abort 队列；⑥ `scheduler.update_from_output()` 处理模型输出，生成 `EngineCoreOutputs` 返回给主进程。

```python
# line 443
def step(self):
    # 1) 调度：产生 SchedulerOutput
    scheduler_output = self.scheduler.schedule()  # line 454

    # 2) 执行模型（异步，返回 Future）
    future = self.model_executor.execute_model(
        scheduler_output, non_block=True
    )  # line 455

    # 3) 获取 grammar bitmask（结构化输出约束）
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)  # line 456

    # 4) 获取模型输出（如果为 None，需要采样）
    model_output = future.result()  # line 461
    if model_output is None:
        model_output = self.model_executor.sample_tokens(grammar_output)  # line 463

    # 5) 处理 abort 队列
    self._process_aborts_queue()  # line 467

    # 6) 处理模型输出 → EngineCoreOutputs
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )  # line 468

    return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

### 4.5 _process_engine_step() — 输出队列写入

> **`_process_engine_step()`**：EngineCoreProc 调用 step_fn() 执行一次迭代，然后将返回的 `dict[int, EngineCoreOutputs]` 逐个放入 `output_queue`（由后台 output socket 线程发送到主进程）。最后调用 `post_step()` 处理 spec decode 的 draft token 更新（如果有）。
>
> **`post_step()`**：step 后处理。在异步调度场景下，从 Worker 获取 draft token IDs 并更新到 Scheduler 中，供下一轮 spec decode 使用。

```python
# line 1257
def _process_engine_step(self):
    outputs, model_executed = self.step_fn()
    for output in outputs.items() if outputs else ():
        self.output_queue.put_nowait(output)
    self.post_step(model_executed)
```

### 4.6 step_with_batch_queue() — 流水线并行批处理

> **`step_with_batch_queue()`**：Pipeline Parallelism（PP）场景下的 step 实现。维护一个定长 deque `batch_queue`，提前调度下一个批次（`scheduler.schedule()` + `execute_model()`）放入队列，同时异步采样（`sample_tokens(non_block=True)`）。当队列满时阻塞等待最早批次完成。这种设计消除 PP 的流水线气泡（pipeline bubble），提高吞吐。
>
> **`batch_queue`**：存储 `(Future[ModelRunnerOutput], SchedulerOutput, Future[Any])` 三元组的有界双端队列。容量由 `vllm_config.max_concurrent_batches` 控制。

```python
# line 484
def step_with_batch_queue(self):
    batch_queue = self.batch_queue
    # 1) 队列未满时，提前调度下一批
    if len(batch_queue) < self.batch_queue_size:
        if self.scheduler.has_requests():
            scheduler_output = self.scheduler.schedule()
            exec_future = self.model_executor.execute_model(scheduler_output, non_block=True)
            # 异步采样
            if not scheduler_output.pending_structured_output_tokens:
                grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
                future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            batch_queue.appendleft((future, scheduler_output, exec_future))
            if len(batch_queue) < self.batch_queue_size ...:
                return None, model_executed  # 不等输出，继续填充队列

    # 2) 队列已满，阻塞等最早批次完成
    future, scheduler_output, exec_model_fut = batch_queue.pop()
    model_output = future.result()
    engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
    return engine_core_outputs, model_executed
```

---

## 5. Phase 4: Scheduler.schedule() 调度决策

> **`Scheduler`**：调度器是整个系统的决策核心。维护三种请求队列（waiting / running / skipped_waiting），管理 KV cache 的分配与释放，实现 prefix caching、preemption、chunked prefill 等高级功能。每个 EngineCore step 调用一次 `schedule()` 输出 `SchedulerOutput`。
>
> **`Request`**：单个请求的内部状态表示。关键字段：`prompt_token_ids`（原始 prompt），`_all_token_ids`（所有 token，含已生成的 output），`_output_token_ids`（已生成的 output），`num_computed_tokens`（已计算的 token 数），`status`（WAITING / RUNNING / PREEMPTED / FINISHED_*），`sampling_params`，`arrival_time`，`priority`，`num_preemptions`（被抢占次数），`mm_features`（多模态特征）。调度决策的核心依据是 `num_computed_tokens` 与 `num_tokens_with_spec` 的关系。
>
> **`KVCacheManager`**：KV cache 内存管理器。管理：block 的分配/释放/复用、prefix cache 的哈希匹配、block 哈希表维护、block 池管理。`allocate_slots()` 是调度中最关键的调用——请求需要新的 KV 缓存 block 时通过它分配，分配失败则触发抢占。
>
> **`waiting / running / skipped_waiting`**：三种请求队列。`waiting` 存放尚未开始生成的新请求；`running` 存放已经进入生成阶段的请求（正在 decode 或 chunked prefill）；`skipped_waiting` 存放暂时不能调度的请求（如等待 grammar 编译完成、等待远程 KV 加载完成、等待 streaming input 下一段）。
>
> **`SchedulingPolicy`**：调度策略枚举，支持 `FCFS`（先来先服务）和 `PRIORITY`（优先级）。`PRIORITY` 模式下，高优先级请求（priority 值小）先于低优先级请求调度，抢占时也优先保留高优先级请求。

### 5.1 调度数据结构

```python
# line 65
class Scheduler(SchedulerInterface):
    def __init__(self, vllm_config, kv_cache_config, ...):
        # 调度约束
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = (
            self.scheduler_config.max_num_scheduled_tokens
            if ...
            else self.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len

        # 请求队列
        self.waiting = create_request_queue(self.policy)        # 等待队列
        self.skipped_waiting = create_request_queue(self.policy) # 跳过队列
        self.running: list[Request] = []                         # 运行队列
        self.requests: dict[str, Request] = {}                   # 全量请求

        # KV Cache 管理器
        self.kv_cache_manager = KVCacheManager(...)
```

### 5.2 schedule() — 主调度逻辑

> **`SchedulerOutput`**：调度步骤的输出，描述本轮"谁该干什么"。关键字段：`scheduled_new_reqs: list[NewRequestData]`（新 prefill 请求）、`scheduled_cached_reqs: CachedRequestData`（续推 decode 请求）、`num_scheduled_tokens: dict[str, int]`（每个请求分配的 token 数）、`total_num_scheduled_tokens: int`（总 token 数）、`scheduled_spec_decode_tokens`（spec decode 的 draft token）、`scheduled_encoder_inputs`（多模态 encoder 输入）、`preempted_req_ids`（被抢占的请求 ID）、`finished_req_ids`（已完成的请求 ID）。

```python
# line 336
def schedule(self) -> SchedulerOutput:
    self.current_step += 1
    token_budget = self.max_num_scheduled_tokens

    # === Step 1: 调度 RUNNING 请求（decode 续推）===
    req_index = 0
    while req_index < len(self.running) and token_budget > 0:
        request = self.running[req_index]

        # 计算需要调度多少新 token
        num_new_tokens = (
            request.num_tokens_with_spec
            + request.num_output_placeholders
            - request.num_computed_tokens
        )
        if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
            num_new_tokens = self.scheduler_config.long_prefill_token_threshold
        num_new_tokens = min(num_new_tokens, token_budget)

        # 分配 KV cache slot
        new_blocks = self.kv_cache_manager.allocate_slots(
            request, num_new_tokens,
            num_lookahead_tokens=self.num_lookahead_tokens,
        )

        if new_blocks is None:
            # KV cache 不足 → 抢占最低优先级请求
            if self.policy == SchedulingPolicy.PRIORITY:
                preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
            else:
                preempted_req = self.running.pop()
            self._preempt_request(preempted_req, timestamp)  # → 回 waiting

        # 调度成功
        scheduled_running_reqs.append(request)
        num_scheduled_tokens[request_id] = num_new_tokens
        token_budget -= num_new_tokens

    # === Step 2: 调度 WAITING 请求（新 prefill）===
    while (self.waiting or self.skipped_waiting) and token_budget > 0:
        if len(self.running) == self.max_num_running_reqs:
            break

        request = request_queue.peek_request()

        # 检查 prefix cache 命中（本地）
        new_computed_blocks, num_new_local_computed_tokens = (
            self.kv_cache_manager.get_computed_blocks(request)
        )
        # 检查外部 KV cache 命中（KVConnector，PD 分离场景）
        if self.connector is not None:
            ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(
                request, num_new_local_computed_tokens
            )

        num_new_tokens = request.num_tokens - num_computed_tokens
        num_new_tokens = min(num_new_tokens, token_budget)

        # 分配 KV cache block
        new_blocks = self.kv_cache_manager.allocate_slots(
            request, num_new_tokens,
            num_new_computed_tokens=num_new_local_computed_tokens,
            new_computed_blocks=new_computed_blocks,
            ...
        )

        if new_blocks is None:
            break  # 无可抢占，停止调度

        # 调度成功 → 移入 running
        self.running.append(request)
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = num_computed_tokens
        scheduled_new_reqs.append(request)
        req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(request_id)
        num_scheduled_tokens[request_id] = num_new_tokens
        token_budget -= num_new_tokens

    # === Step 3: 构造 SchedulerOutput ===
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=new_reqs_data,        # 新的 prefill 请求
        scheduled_cached_reqs=cached_reqs_data,   # 续推的 decode 请求
        num_scheduled_tokens=num_scheduled_tokens, # 每个请求的 token 数
        total_num_scheduled_tokens=...,            # 总 token 数
        scheduled_spec_decode_tokens=...,          # spec decode token
        scheduled_encoder_inputs=...,              # 多模态 encoder 输入
        preempted_req_ids=...,                     # 被抢占的请求
        finished_req_ids=self.finished_req_ids,    # 已完成的请求
        num_common_prefix_blocks=...,              # 公共前缀 block（用于 cascade attention）
    )
    return scheduler_output
```

### 5.3 _update_after_schedule() — 调度后更新

> **`_update_after_schedule()`**：调度完成后立即调用的后处理。乐观地推进每个被调度请求的 `num_computed_tokens`（标记为已计算，即使模型尚未开始执行）。设置 `is_prefill_chunk` 标志（区分 prefill 中 vs decode 中）。清空 `finished_req_ids`（已复制到 SchedulerOutput）。如果启用了 `enable_return_routed_experts`，同时快照 block ID 供后续 routed experts 查询。

```python
# line 979
def _update_after_schedule(self, scheduler_output):
    for req_id, num_scheduled_token in num_scheduled_tokens.items():
        request = self.requests[req_id]
        request.num_computed_tokens += num_scheduled_token
        request.is_prefill_chunk = request.num_computed_tokens < (
            request.num_tokens + request.num_output_placeholders
        )
    self.finished_req_ids = set()
```

---

## 6. Phase 5: Worker 模型执行

### 6.1 NPUWorker.execute_model()

> **`NPUWorker`**：vllm-ascend 的 Worker 实现（继承自 vLLM `WorkerBase`），每个 Worker 进程持有一块 NPU 设备。`execute_model()` 的核心流程：① 内存 profiling（`profile_memory()`）；② PP 非首 rank 从上一 rank 接收 `IntermediateTensors`（`irecv_tensor_dict()`）；③ 调 `model_runner.execute_model()` 执行 forward；④ PP 非末 rank 将结果传到下一 rank（`isend_tensor_dict()`）；⑤ 末 rank 返回 `ModelRunnerOutput` 或 `AsyncModelRunnerOutput`。

```python
# vllm_ascend/worker/worker.py:417
def execute_model(self, scheduler_output):
    self.profile_memory()

    # PP: 非首 rank 从上一 rank 接收中间张量
    if forward_pass and not get_pp_group().is_first_rank:
        tensor_dict, comm_handles, comm_postprocess = get_pp_group().irecv_tensor_dict(
            all_gather_group=get_tp_group()
        )
        intermediate_tensors = AsyncIntermediateTensors(
            tensor_dict, comm_handles=comm_handles, comm_postprocess=comm_postprocess,
        )

    # 执行模型 forward
    output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)

    # PP: 非末 rank 发送中间张量到下一 rank
    if isinstance(output, IntermediateTensors):
        self._pp_send_work = get_pp_group().isend_tensor_dict(
            output.tensors, all_gather_group=get_tp_group(),
        )
        return None

    return output
```

### 6.2 NPUWorker.init_device() — 设备初始化

> **`init_device()`**：Worker 进程的 NPU 初始化入口。依次执行：① `torch.npu.set_device()` 绑定 NPU 设备；② `init_distributed_environment()` 初始化 HCCL（华为集合通信库）分布式环境；③ `init_ascend_model_parallel()` 初始化 Ascend 模型并行（TP/PP/DP 通信组）；④ 创建 `NPUModelRunner` 实例（v1 或 v2 版本取决于 `VLLM_USE_V2_MODEL_RUNNER`）。

```python
# vllm_ascend/worker/worker.py:317
def init_device(self):
    torch.npu.set_device(self.local_rank)         # 设置 NPU 设备
    init_distributed_environment(...)              # 初始化 HCCL 分布式
    init_ascend_model_parallel(...)                # 初始化 Ascend 模型并行

    if self.use_v2_model_runner:
        from vllm_ascend.worker.v2.model_runner import NPUModelRunner
    else:
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    self.model_runner = NPUModelRunner(...)
```

### 6.3 NPUModelRunner.execute_model() — 模型执行

> **`NPUModelRunner`**：Ascend NPU 上的模型运行器，继承自 vLLM 的 `GPUModelRunner`。`execute_model()` 的完整流程：① `_update_states(scheduler_output)` —— 根据调度输出更新输入 batch，添加新请求的 token IDs、更新续推请求的位置和 KV cache block 映射；② `_prepare_inputs(scheduler_output)` —— 准备 logits indices、attention metadata、sampling metadata；③ 构建 `AttentionMetadata`（通过 `AscendAttentionBackend.make_metadata()`）；④ `self.model(**kwargs)` —— 执行模型 forward，产出 hidden_states；⑤ `_get_logits()` —— 从 hidden_states 中提取 logits；⑥ 返回 `ModelRunnerOutput`（含 sampled_token_ids、logprobs 等）。
>
> **`InputBatch`**：ModelRunner 内部的输入缓冲区。管理当前 step 所有请求的 input_ids、positions、block_tables（KV cache block 映射表）、sampling_metadata 等。`_update_states()` 根据 `SchedulerOutput` 动态更新 batch 内容（添加新请求、移除已完成请求、更新续推请求等）。
>
> **`IntermediateTensors`**：PP 场景中在模型各 stage 之间传递的中间结果。每个 PP rank 只计算一部分 transformer layer，然后将 hidden_states + 其他辅助 tensors 打包为 `IntermediateTensors` 传给下一 rank。NPUWorker 使用 `AsyncIntermediateTensors` 支持异步通信。
>
> **`AscendAttentionBackend`**：Ascend NPU 上的注意力机制后端抽象。vLLM 的 attention backend 接口允许不同硬件实现不同的注意力算法：
>   - `AscendAttentionBackend`：通用 PagedAttention 实现，支持动态分页 KV cache。
>   - `AscendMLABackend`：Multi-head Latent Attention，用于 DeepSeek 系列模型，通过低秩压缩减少 KV cache 显存占用。
>   - `AscendSFABackend`：Sparse Flash Attention，利用稀疏性跳过不重要的注意力计算。
>   - `AscendDSABackend`：DSA（Decode-context Sparse Attention），用于 decode-context parallel 场景。
>   - `AscendFABackend`：Flash Attention 3 实现 (FA3)。

```python
# vllm_ascend/worker/model_runner_v1.py（继承自 GPUModelRunner）
def execute_model(self, scheduler_output, intermediate_tensors):
    # 1) 更新状态（新 token、KV block、位置信息）
    self._update_states(scheduler_output)

    # 2) 准备输入（logits indices, attention metadata）
    self._prepare_inputs(scheduler_output)

    # 3) 构建 AttentionMetadata
    attn_metadata = self.attn_backend.make_metadata(...)

    # 4) 模型 forward
    hidden_states = self.model(
        input_ids=self.input_batch.input_ids,
        positions=self.input_batch.positions,
        kv_caches=self.kv_caches,
        attn_metadata=attn_metadata,
        ...
    )

    # 5) 获取 logits
    logits = self._get_logits(hidden_states, ...)

    # 6) 返回 ModelRunnerOutput
    return ModelRunnerOutput(
        req_id_to_index=...,
        sampled_token_ids=...,   # 采样后的 token IDs
        logprobs=...,
        ...
    )
```

---

## 7. Phase 6: AscendSampler 采样

### 7.1 采样流程

> **`AscendSampler`**：Ascend NPU 上的采样器，继承自 vLLM 的 `Sampler`。核心职责是将模型输出的 logits（词汇表概率分布）转换为具体的 token ID。针对 Ascend 硬件的特性做了多项优化。
>
> **`apply_penalties()`**：静态方法，在采样前对 logits 施加三种惩罚。使用 Triton-Ascend 核函数在 NPU 上完成，避免将 logits/历史 token IDs 传输到 CPU。三种惩罚：`presence_penalty`（已出现的 token 减分）、`frequency_penalty`（出现越频繁减分越多）、`repetition_penalty`（已出现的 token 评分压低）。
>
> **`greedy_sample()`**：贪心采样，直接取 logits 最大值的索引（`argmax`）。当 `enable_reduce_sample` 配置启用时，通过 TP all_gather 通信在所有 GPU 间取全局 argmax（适用于并行词表场景）。
>
> **`SamplingMetadata`**：采样所需的请求级元数据。`AscendSampler` 的 `SamplingMetadata` 包含每个请求的 `temperature`、`top_p`、`top_k`、`presence_penalty`、`frequency_penalty`、`repetition_penalty`、`seed` 等参数，以及 `prompt_token_ids`（用于惩罚计算）。由 ModelRunner 在 `_prepare_inputs()` 阶段构建。

```python
# vllm_ascend/sample/sampler.py:44
class AscendSampler(Sampler):
    @staticmethod
    def apply_penalties(logits, sampling_metadata, output_token_ids):
        if HAS_TRITON and not sampling_metadata.no_penalties:
            return apply_all_penalties(
                logits,
                sampling_metadata.prompt_token_ids,
                sampling_metadata.presence_penalties,
                sampling_metadata.frequency_penalties,
                sampling_metadata.repetition_penalties,
                output_token_ids,
            )
        return Sampler.apply_penalties(...)

    @staticmethod
    def greedy_sample(logits):
        if get_ascend_config().enable_reduce_sample:
            # TP 通信后取全局 argmax
            local_max_logits, local_max_indices = logits.max(dim=-1)
            gathered_logits = tp_group.all_gather(...)
            global_max_rank = gathered_logits.argmax(dim=-1)
            return gathered_global_idx.gather(...)
        return logits.argmax(dim=-1).view(-1)
```

### 7.2 随机采样的 Ascend 优化

> **`random_sample()`**：随机采样的 Ascend 优化版本。不使用 `torch.multinomial`（该函数会引起 NPU→CPU 的同步阻塞），而是使用"指数分布噪声 + argmax"的等效算法：对概率分布 p 生成指数分布随机数 e，取 `argmax(p / e)`。所有计算在 NPU 上完成，避免跨设备同步。
>
> **`do_async_exponential()`**：在独立的 NPU stream（`global_stream()`）上预先生成下一轮的指数分布随机数，与当前轮次的模型执行并行。通过 `torch.npu.Event` 记录完成时机，采样时通过 `wait_stream()` 同步。这种重叠技术隐藏了随机数生成的开销。
>
> **`AscendTopKTopPSampler`**：top-k 和 top-p（nucleus）采样的 Ascend 实现。`forward_native()` 重写了 PyTorch 原生实现以使用 Ascend 算子。`prepare_sampling()` 根据模型的 top-k 配置记录当前 top-k 值。在 batch_invariant 模式下回退到 vLLM 默认实现。

```python
# vllm_ascend/sample/sampler.py:18
def random_sample(probs, generators):
    """
    使用指数分布 + argmax 替代 torch.multinomial
    原因：torch.multinomial 会引起 CPU-NPU 同步
    """
    with npu_stream_switch(global_stream()):
        q = torch.empty_like(probs)
        q.exponential_()                         # 在 NPU 上生成指数分布随机数
        if generators:
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)
    torch.npu.current_stream().wait_stream(global_stream())
    return probs.div_(q).argmax(dim=-1).view(-1)  # argmax(p/q) 等效于按概率采样
```

---

## 8. Phase 7: update_from_output 输出处理

> **`update_from_output()`**：Scheduler 处理模型输出的核心方法，在 `step()` 末尾调用。输入为 `SchedulerOutput`（本轮调度决定）和 `ModelRunnerOutput`（模型推理结果），输出为 `dict[int, EngineCoreOutputs]`（按 client_index 分组的输出集合）。
>
> **`EngineCoreOutput`**：单个请求的单步输出。字段：`request_id`、`new_token_ids: list[int]`（新生成的 token IDs）、`finish_reason: FinishReason`（停止原因：STOPPED / LENGTH / ABORTED / ERROR）、`stop_reason: int | str`（具体停止位置）、`new_logprobs`（token 级别的 logprob 值）、`new_prompt_logprobs_tensors`（prompt logprobs，仅在 prefill 阶段返回）、`pooling_output`（pooling 模型的输出）、`kv_transfer_params`（PD 分离场景的 KV 传输参数）、`routed_experts`（MoE 路由专家信息）、`events`（事件日志，用于统计）。

**文件**: `vllm/v1/core/sched/scheduler.py`

```python
# line 1308
def update_from_output(self, scheduler_output, model_runner_output):
    sampled_token_ids = model_runner_output.sampled_token_ids
    logprobs = model_runner_output.logprobs
    num_scheduled_tokens = scheduler_output.num_scheduled_tokens

    outputs = defaultdict(list)

    # 遍历所有被调度的请求
    for req_id, num_tokens in num_scheduled_tokens.items():
        request = self.requests.get(req_id)
        if request is None or request.is_finished():
            continue

        # 获取该请求的采样结果
        req_index = model_runner_output.req_id_to_index[req_id]
        generated_token_ids = sampled_token_ids[req_index]

        # --- 处理 spec decode 拒绝 ---
        if scheduled_spec_token_ids and generated_token_ids:
            num_draft_tokens = len(scheduled_spec_token_ids)
            num_accepted = len(generated_token_ids) - 1
            num_rejected = num_draft_tokens - num_accepted
            request.num_computed_tokens -= num_rejected

        # --- 追加 output token 并检查 stop ---
        new_token_ids, stopped = self._update_request_with_output(
            request, new_token_ids
        )
        # check_stop() 内部检查: max_tokens, stop strings, stop token ids, eos

        # --- 处理已完成请求 ---
        if stopped:
            finish_reason = request.get_finished_reason()
            self._handle_stopped_request(request)
            self._free_request(request)

        # --- 构建 EngineCoreOutput ---
        if new_token_ids or stopped:
            outputs[request.client_index].append(
                EngineCoreOutput(
                    request_id=req_id,
                    new_token_ids=new_token_ids,
                    finish_reason=finish_reason,
                    new_logprobs=new_logprobs,
                    ...
                )
            )

    # 移除已停止的请求
    if stopped_running_reqs:
        self.running = remove_all(self.running, stopped_running_reqs)

    return {
        client_index: EngineCoreOutputs(outputs=outs)
        for client_index, outs in outputs.items()
    }
```

---

## 9. Phase 8: 输出回传与流式响应

### 9.1 输出回传路径

> **`IncrementalDetokenizer`**：增量式解码器。每次收到新 token IDs，逐步解码出对应文本。维护每个请求的 byte 级别解码状态（utf-8 编码的中间缓冲区），支持跨多个 token 的字符拼接。核心方法 `decode(token_ids)` 返回新解码出的文本段。`num_output_tokens()` 追踪已输出的 token 数量，用于 stream interval 控制。
>
> **`LogprobsProcessor`**：logprobs 格式化处理器。将引擎返回的原始 logprobs tensor 转换为 OpenAI API 格式的 `Logprob` 对象列表。每个 `Logprob` 包含 `token`（token ID）、`token_logprob`（对数概率值）、`bytes`（token 的 utf-8 字节表示）、`top_logprobs`（top-k 候选 token 的 logprob）。
>
> **`RequestState`**：OutputProcessor 内部维护的每个请求的完整状态。字段：`request_id`、`detokenizer: IncrementalDetokenizer`、`logprobs_processor: LogprobsProcessor`、`queue: RequestOutputCollector`、`output_kind`（输出模式：逐 token/累积/仅最终）、`stream_interval`（多少 token 输出一次）、`sent_tokens_offset`（已发送的 token 偏移量）、`is_prefilling`（是否仍在 prefill 阶段）、`stats: RequestStateStats`（请求级统计）。`make_request_output()` 方法根据输出模式和间隔决定是否产出 `RequestOutput`。
>
> **`RequestOutputCollector`**：单请求的输出收集器。`put(RequestOutput)` 由 `output_handler` 调用（非阻塞），`get()` / `get_nowait()` 由 `generate()` 的 async generator 消费（阻塞使用 asyncio.Event）。`aggregate` 标志控制 DELTA 模式下是否增量合并输出。
>
> **`OutputProcessor`**：完整的输出处理流水线。`process_outputs()` 接收一批 `EngineCoreOutput`，对每个 output：① 查找对应的 `RequestState`；② `detokenizer.decode()` 将 token IDs 转为文本；③ `logprobs_processor.process()` 格式化 logprobs；④ `make_request_output()` 决定是否产出 `RequestOutput`（受 stream interval 控制）；⑤ 若有输出，`queue.put(request_output)` 推送给 generate() 消费者。同时收集 stats 和需要 abort 的请求。

```
EngineCore 子进程                   主进程
┌──────────────┐   ZMQ PUB   ┌──────────────────┐
│ output_queue  │───────────▶│ get_output_async() │
│ (socket线程)  │   Socket   │ (output_handler)   │
└──────────────┘             └────────┬─────────┘
                                      │
                            OutputProcessor
                            .process_outputs()
                              ├─ IncrementalDetokenizer.decode()
                              ├─ LogprobsProcessor.process()
                              └─ make_request_output() → queue.put()
                                      │
                                     ▼
                            RequestOutputCollector
                            .put(RequestOutput)
                                      │
                            AsyncLLM.generate()
                            while 循环 yield
                                      │
                                     ▼
                            OpenAIServingChat
                            chat_completion_stream_generator()
                                      │
                            SSE: data: {...}\n\n
```

### 9.2 流式响应生成

> **`chat_completion_stream_generator()`**：将 `AsyncGenerator[RequestOutput]` 转换为 OpenAI 格式 SSE 流。首个 chunk 发送 `role` delta（如 `{"role":"assistant"}`），后续每个 chunk 包含 `delta.content`（新生成的文本段，可能为空）、`delta.reasoning_content`（思考链片段）。最终 chunk 包含 `finish_reason`（"stop"/"length"）和 `usage` 统计（prompt_tokens、completion_tokens、total_tokens）。通过 `stream_harmony` 模块处理 DeltaMessage 的合并逻辑。
>
> **`chat_completion_full_generator()`**：非流式响应的生成器。等待所有 token 生成完毕后，一次组装完整的 `ChatCompletionResponse`，包含 `choices[0].message.content`（完整文本）、`usage` 统计、`model` 名称等。
>
> **`DeltaMessage`**：流式响应中每次增量数据的数据结构。字段：`role`（首个 chunk 含 "assistant"）、`content`（新文本、可选）、`reasoning_content`（思考链文本）。
>
> **`RequestOutput`**：OutputProcessor 产出的最终输出格式。字段：`request_id`、`prompt_token_ids`、`prompt_text`、`outputs: list[CompletionOutput]`（每个包含 token_ids、text、logprobs、finish_reason）、`finished` 标志。

**文件**: `vllm/entrypoints/openai/chat_completion/serving.py`

```python
# line 399
async def chat_completion_stream_generator(self, request, result_generator, ...):
    # 第一个 chunk: role
    yield f"data: {ChatCompletionStreamResponse(
        choices=[ChatCompletionResponseStreamChoice(
            delta=DeltaMessage(role=response_role),
        )]
    ).model_dump_json()}\n\n"

    async for res in result_generator:
        for output in res.outputs:
            delta_text = output.text  # 新生成的文本
            yield f"data: {ChatCompletionStreamResponse(
                choices=[ChatCompletionResponseStreamChoice(
                    delta=DeltaMessage(content=delta_text),
                )]
            ).model_dump_json()}\n\n"

    # 最后一个 chunk: finish_reason + usage
    yield f"data: {ChatCompletionStreamResponse(
        choices=[ChatCompletionResponseStreamChoice(
            finish_reason=finish_reason, delta=DeltaMessage(),
        )],
        usage=usage_info,
    ).model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"
```

---

## 10. Ascend 特有调度器补丁

> vllm-ascend 通过 monkey-patch 机制修改 vLLM 核心类的行为以适应 Ascend NPU 特性。补丁分为两类：**全局补丁**（EngineCore 进程启动时打一次，影响 Scheduler、Executor 等核心组件）和 **Worker 补丁**（每个 Worker 进程各打一次，替换 Triton op、模型 forward 等）。补丁分类存放在 `patch/platform/`（全局）和 `patch/worker/`（Worker 级）目录下。

### 10.1 BalanceScheduler — DP 负载均衡

> **`BalanceScheduler`**：继承自 vLLM 的 `Scheduler`，增加了 DP（Data Parallel）rank 间的负载均衡功能。当 `enable_balance_scheduling=True` 时，`patch_balance_schedule.py` 全局替换 `Scheduler` 为 `BalanceScheduler`。
>
> **`balance_gather()`**：每步调度前，通过 `dist.all_gather` 收集所有 DP rank 的当前 running 请求数。如果某个 rank 已达 `max_num_running_reqs` 上限，暂停调度新请求，等待其他 rank 处理完当前请求跟上进度。避免多 DP rank 间负载不均导致某些 rank 堆积、其他 rank 空闲。
>
> 该补丁同时替换了 `EngineCoreProc.run_engine_core`，在 DP 场景下创建 `BalanceDPEngineCoreProc` 替代标准 `EngineCoreProc`，支持 DP 负载均衡的 busy loop。

```python
# vllm_ascend/patch/platform/patch_balance_schedule.py:36
class BalanceScheduler(Scheduler):
    def __init__(self, ...):
        super().__init__(...)
        self._balance_enabled = _balance_scheduling_enabled(vllm_config)
        if self._balance_enabled:
            self.balance_queue = [
                torch.tensor([0], dtype=torch.int, device="cpu")
                for _ in range(self.vllm_config.parallel_config.data_parallel_size)
            ]

    def balance_gather(self, dp_group):
        """收集各 DP rank 的 running 请求数"""
        if not self._balance_enabled:
            return
        running_tensor = torch.tensor([len(self.running)], dtype=torch.int, device="cpu")
        dist.all_gather(self.balance_queue, running_tensor, group=dp_group)

    def schedule(self) -> SchedulerOutput:
        if not self._balance_enabled:
            return super().schedule()
        # ... 同标准调度 ...
        # 但调度 WAITING 请求前检查：
        while (self.waiting or self.skipped_waiting) and token_budget > 0:
            balance_flag = max(
                t.item() for t in self.balance_queue
            ) == self.max_num_running_reqs
            if balance_flag:
                break  # 暂停，等其它 rank 追上
            ...
```

### 10.2 其他调度相关补丁

> **`patch_profiling_chunk.py`**：Profiling chunk scheduler 补丁。在 EngineCore 初始化后运行业 profiling（跑少量请求测量预填充延迟），拟合二次模型预测最优 chunk 大小。Hook `scheduler.update_from_output()` 将实际执行时间反馈给 ProfilingChunkManager 调整预测模型。
>
> **`patch_scheduler.py`**：调度器小补丁，修复 mamba model 的 block-aligned split 断言在外部 KV connector 场景失败的问题。
>
> **`patch_kv_cache_interface.py`**：扩展 `MLAAttentionSpec` 以支持 DSA（Decode-context Sparse Attention）和 Sparse C8 KV cache 格式。
>
> **`patch_kv_cache_utils.py`**：Hybrid KV cache 支持（不同 layer 使用不同 KV cache block 大小），以及 Context Parallel 场景下的 block size 修复。
>
> **`patch_multiproc_executor.py`**：替换 `MultiprocExecutor` 为 `AscendMultiprocExecutor`，将子进程创建参数 `daemon=True` 改为 `daemon=False`，确保进程正常退出。
>
> **`patch_mamba_manager.py`**：替换 `MambaManager` 为 `AscendMambaManager`，支持 hybrid prefix cache 与 PCP（Prefill Context Parallel）/ DCP（Decode Context Parallel）。
>
> **`adapt_patch()` 函数**：补丁分发入口（`vllm_ascend/utils.py`）。当 `is_global_patch=True` 时导入 `vllm_ascend.patch.platform`（全局补丁）；否则导入 `vllm_ascend.patch.worker`（Worker 级补丁）。全局补丁在 `NPUPlatform.pre_register_and_update()` 和 `vllm_ascend.__init__._ensure_global_patch()` 中调用。Worker 补丁在 `NPUWorker.__init__()` 中调用。

| 补丁 | 位置 | 作用 |
|------|------|------|
| `patch_balance_schedule.py` | platform | 替换 Scheduler 为 BalanceScheduler + 替换 EngineCoreProc.run_engine_core |
| `patch_profiling_chunk.py` | platform | EngineCore init 后运行 profiling，hook `update_from_output` 反馈延迟 |
| `patch_scheduler.py` | platform | 修复 Mamba block-aligned split 断言 |
| `patch_kv_cache_interface.py` | platform | 扩展 MLAAttentionSpec（DSA + Sparse C8） |
| `patch_kv_cache_utils.py` | platform | Hybrid KV cache + CP block size 修复 |
| `patch_multiproc_executor.py` | platform | AscendMultiprocExecutor（daemon=False） |
| `patch_mamba_manager.py` | platform | AscendMambaManager（prefix cache + PCP/DCP） |
| `patch_distributed.py` | platform | Tensor alignment 修复 |
| `patch_torch_accelerator.py` | platform | torch.accelerator → torch.npu 重定向 |
| `patch_triton.py` | worker | 替换 Triton op 为 Ascend 等价实现 |
| `patch_weight_utils.py` | worker | KV scale 参数重映射 |
| `patch_rejection_sampler.py` | worker | NPU 专用的 rejection sampling |
| `patch_cudagraph.py` | worker | CUDAGraph dispatch 修复 |

---

## 11. 完整流程图

```
┌────────────────────────────────────────────────────────────────────────────────┐
│  用户请求 POST /v1/chat/completions                                              │
├────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │         主进程 (uvicorn + AsyncLLM)                                      │   │
│  │                                                                         │   │
│  │  1. FastAPI Router → create_chat_completion()                           │   │
│  │     OpenAIServingChat._create_chat_completion()                          │   │
│  │     ├─ render_chat_request() → OpenAIServingRender(tokenize + template) │   │
│  │     └─ engine_client.generate() → AsyncLLM.generate()                   │   │
│  │                                                                         │   │
│  │  2. AsyncLLM.generate()                                                  │   │
│  │     ├─ add_request()                                                     │   │
│  │     │  ├─ InputProcessor.process_inputs() → EngineCoreRequest           │   │
│  │     │  ├─ OutputProcessor.add_request() → RequestState                  │   │
│  │     │  └─ ZMQ → EngineCore子进程                                         │   │
│  │     └─ while: q.get() → yield RequestOutput                             │   │
│  │                                                                         │   │
│  │  3. [后台] output_handler task                                           │   │
│  │     ├─ engine_core.get_output_async() → EngineCoreOutputs (ZMQ)         │   │
│  │     └─ OutputProcessor.process_outputs()                                │   │
│  │        ├─ IncrementalDetokenizer.decode()  (token IDs → text)           │   │
│  │        ├─ LogprobsProcessor.process()      (logprobs 格式化)             │   │
│  │        └─ RequestOutputCollector.put()     (推给 generate())            │   │
│  │                                                                         │   │
│  │  4. Serving 层响应                                                       │   │
│  │     ├─ [流式] chat_completion_stream_generator() → SSE chunks            │   │
│  │     └─ [非流式] chat_completion_full_generator() → JSON                  │   │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │ ZMQ (ROUTER-DEALER)
┌──────────────────────────────────────────────────────────────────────────────┐
│           EngineCore 子进程 (EngineCoreProc)                                  │
│                                                                              │
│  主循环: run_busy_loop()                                                     │
│  ┌─────────────────────────────────────────┐                                │
│  │ while _handle_shutdown():                │                                │
│  │   _process_input_queue()   (收请求)      │                                │
│  │   _process_engine_step()   (走一步)      │                                │
│  └─────────────────────────────────────────┘                                │
│                                                                              │
│  每步 step():                                                               │
│  ┌─ 1. scheduler.schedule() ──────────────────────────────────────────┐     │
│  │    ├─ Step 1: RUNNING 处理 (decode 续推)                            │     │
│  │    │  ├─ 计算 num_new_tokens                                       │     │
│  │    │  ├─ kv_cache_manager.allocate_slots()                         │     │
│  │    │  ├─ 分配失败 → _preempt_request()                             │     │
│  │    │  └─ 记录到 scheduled_running_reqs                             │     │
│  │    │                                                               │     │
│  │    ├─ Step 2: WAITING 处理 (新 prefill)                             │     │
│  │    │  ├─ prefix cache 检查 (本地hash + 外部KVConnector)              │     │
│  │    │  ├─ encoder input 调度                                        │     │
│  │    │  ├─ kv_cache_manager.allocate_slots()                         │     │
│  │    │  └─ → running, status = RUNNING                               │     │
│  │    │                                                               │     │
│  │    └─ 输出: SchedulerOutput                                         │     │
│  │       (new_reqs / cached_reqs / num_tokens / blocks / ...)         │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  ┌─ 2. model_executor.execute_model(scheduler_output) ────────────────┐     │
│  │    └─ ZMQ → Worker 子进程                                            │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  ┌─ 3. future.result() → ModelRunnerOutput                                  │     │
│  │    └─ 若 None → sample_tokens()                                         │     │
│  └──────────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  ┌─ 4. scheduler.update_from_output(scheduler_output, model_output) ────┐     │
│  │    ├─ 遍历 num_scheduled_tokens                                        │     │
│  │    ├─ spec decode 拒绝处理 (num_computed_tokens -= rejected)           │     │
│  │    ├─ _update_request_with_output() → check_stop()                     │     │
│  │    ├─ 处理已完成请求 (handle_stopped + free)                            │     │
│  │    └─ 输出: EngineCoreOutputs → ZMQ → 主进程                            │     │
│  └──────────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │ ZMQ (multiprocess)
┌──────────────────────────────────────────────────────────────────────────────┐
│           Worker 子进程 (NPUWorker)                                           │
│                                                                              │
│  execute_model(scheduler_output):                                            │
│  ┌─ 1. [PP] PP非首rank: irecv_tensor_dict() ← 上一rank                      │     │
│  │                                                                           │
│  ├─ 2. model_runner.execute_model(scheduler_output, intermediate_tensors)   │     │
│  │    ├─ _update_states()   → 更新 InputBatch                                │     │
│  │    │  ├─ scheduled_new_reqs: 添加新请求的 token IDs / positions / blocks  │     │
│  │    │  └─ scheduled_cached_reqs: 续推请求的位置/block 更新                 │     │
│  │    │                                                                      │     │
│  │    ├─ _prepare_inputs() → 准备 logits indices / sampling metadata         │     │
│  │    │                                                                      │     │
│  │    ├─ AscendAttentionBackend.make_metadata() → AttentionMetadata          │     │
│  │    │  ├─ AscendAttentionBackend (通用 PagedAttention)                      │     │
│  │    │  ├─ AscendMLABackend (DeepSeek MLA 低秩压缩注意力)                    │     │
│  │    │  ├─ AscendSFABackend (稀疏 Flash Attention)                           │     │
│  │    │  └─ AscendDSABackend (DSA Context Parallel)                           │     │
│  │    │                                                                      │     │
│  │    ├─ model(**kwargs) → forward pass → hidden_states                      │     │
│  │    └─ _get_logits() → ModelRunnerOutput (sampled_token_ids + logprobs)    │     │
│  │                                                                           │     │
│  ├─ 3. [PP] PP非末rank: isend_tensor_dict() → 下一rank                       │     │
│  │                                                                           │     │
│  └─ 4. 返回 ModelRunnerOutput → EngineCore                                    │     │
│                                                                              │
│  sample_tokens() → AscendSampler:                                            │
│  ├─ apply_penalties()      → Triton-Ascend 惩罚核函数                         │     │
│  ├─ topk_topp_sampler      → top-k / top-p 过滤                              │     │
│  ├─ random_sample()        → exponential + argmax (避免 CPU sync)            │     │
│  └─ greedy_sample()        → argmax (或 TP all_gather 全局 argmax)          │     │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 附录: 关键文件索引

| 组件 | 文件 | 关键类/方法 |
|------|------|-------------|
| **API 路由** | `vllm/entrypoints/openai/chat_completion/api_router.py` | `create_chat_completion()`, `attach_router()` |
| **Chat Serving** | `vllm/entrypoints/openai/chat_completion/serving.py` | `OpenAIServingChat._create_chat_completion()`, `chat_completion_stream_generator()` |
| **引擎客户端** | `vllm/v1/engine/async_llm.py` | `AsyncLLM.generate()`, `add_request()`, `_run_output_handler()` |
| **输入处理器** | `vllm/v1/engine/input_processor.py` | `InputProcessor.process_inputs()` |
| **输出处理器** | `vllm/v1/engine/output_processor.py` | `OutputProcessor.process_outputs()`, `RequestOutputCollector`, `RequestState`, `IncrementalDetokenizer` |
| **EngineCore** | `vllm/v1/engine/core.py` | `EngineCore.step()`, `EngineCoreProc.run_busy_loop()`, `EngineCoreProc.run_engine_core()` |
| **EngineCoreClient** | `vllm/v1/engine/core_client.py` | `MPClient.__init__()`, `launch_core_engines()` |
| **引擎工具** | `vllm/v1/engine/utils.py` | `CoreEngineProcManager`, `launch_core_engines()` |
| **CLI 入口** | `vllm/entrypoints/cli/main.py` | `main()` |
| **Serve 子命令** | `vllm/entrypoints/cli/serve.py` | `ServeSubcommand.cmd()` |
| **API Server** | `vllm/entrypoints/openai/api_server.py` | `build_app()`, `init_app_state()`, `run_server()`, `build_async_engine_client()` |
| **生成路由** | `vllm/entrypoints/generate/api_router.py` | `register_generate_api_routers()`, `init_generate_state()` |
| **Serving 模型** | `vllm/entrypoints/openai/models/serving.py` | `OpenAIServingModels` |
| **调度器** | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule()`, `update_from_output()`, `add_request()` |
| **调度器输出** | `vllm/v1/core/sched/output.py` | `SchedulerOutput`, `NewRequestData`, `CachedRequestData` |
| **KV Cache 管理器** | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager.allocate_slots()` |
| **请求对象** | `vllm/v1/request.py` | `Request`, `RequestStatus` |
| **Worker 基类** | `vllm_ascend/worker/worker.py` | `NPUWorker.execute_model()`, `init_device()` |
| **Model Runner v1** | `vllm_ascend/worker/model_runner_v1.py` | `NPUModelRunner.execute_model()` (extends GPUModelRunner) |
| **Model Runner v2** | `vllm_ascend/worker/v2/model_runner.py` | `NPUModelRunner` (v2) |
| **Ascend 采样器** | `vllm_ascend/sample/sampler.py` | `AscendSampler`, `random_sample()`, `greedy_sample()`, `do_async_exponential()` |
| **Ascend top-k/top-p** | `vllm_ascend/sample/sampler.py` | `AscendTopKTopPSampler.forward_native()` |
| **Ascend 惩罚** | `vllm_ascend/sample/penalties.py` | `apply_all_penalties()` (Triton) |
| **Attention 后端** | `vllm_ascend/attention/attention_v1.py` | `AscendAttentionBackend` |
| **MLA 后端** | `vllm_ascend/attention/mla_v1.py` | `AscendMLABackend` |
| **SFA 后端** | `vllm_ascend/attention/sfa_v1.py` | `AscendSFABackend` |
| **DSA 后端** | `vllm_ascend/attention/dsa_v1.py` | `AscendDSABackend` |
| **FA3 后端** | `vllm_ascend/attention/fa3_v1.py` | `AscendFABackend` |
| **Balance 调度补丁** | `vllm_ascend/patch/platform/patch_balance_schedule.py` | `BalanceScheduler`, `balance_gather()` |
| **Profiling 调度补丁** | `vllm_ascend/patch/platform/patch_profiling_chunk.py` | `ProfilingChunkManager` |
| **全局补丁入口** | `vllm_ascend/patch/platform/__init__.py` | 所有 platform 级补丁导入 |
| **Worker 补丁入口** | `vllm_ascend/patch/worker/__init__.py` | 所有 worker 级补丁导入 |
| **补丁分发** | `vllm_ascend/utils.py` | `adapt_patch()` |
| **NPU 平台** | `vllm_ascend/platform.py` | `NPUPlatform.check_and_update_config()` |
| **插件入口** | `vllm_ascend/__init__.py` | `register()` |
| **Ascend 配置** | `vllm_ascend/ascend_config.py` | `AscendConfig` (--additional-config) |

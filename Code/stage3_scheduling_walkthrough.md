# vLLM 请求调度与执行走读(EngineCore 主循环)

> 接续 [`stage2_request_lifecycle_walkthrough.md`](./stage2_request_lifecycle_walkthrough.md):一次 `POST /v1/chat/completions` 经过 `AsyncLLM.generate` 加工后,通过 ZMQ 推到 EngineCore 子进程。本节走读子进程这边的完整流程:`run_busy_loop` 主循环 → `Scheduler.schedule()` 选 batch → `KVCacheManager.allocate_slots` 分配物理块 → `Executor.execute_model` 跑模型 → `update_from_output` 把 token 推回 API server。
>
> 走读范围:`vllm/v1/engine/core.py` / `vllm/v1/core/sched/scheduler.py` / `vllm/v1/core/kv_cache_manager.py` / `vllm/v1/executor/`,最后顺路看 vllm-ascend 在每一步上的 patch。
>
> 阅读时间约 12 分钟。

---

## 0. 衔接:从 ZMQ 到 EngineCore 子进程

```python
# core_client.py: API server 进程侧
async def add_request_async(self, request: EngineCoreRequest) -> None:  # core_client.py:217
    await self._send_input(EngineCoreRequestType.ADD, request)           # ZMQ PUSH 到 input_queue

async def get_output_async(self):                                       # 持续 poll
    msg = await self._recv_output()                                     # ZMQ PULL output_queue
    return msg
```

```python
# core.py: 子进程入口
def run_engine_core(*args, dp_rank=0, local_dp_rank=0, **kwargs):       # core.py:1093
    """Launch EngineCore busy loop in background process."""
    engine_core = EngineCoreProc(*args, **kwargs)                       # 构造 EngineCore
    engine_core.run_busy_loop()                                         # ★ 永不返回(SystemExit 才退)
```

**关键设计**:`AsyncLLM` 在 API server 进程,`EngineCoreProc` 在独立子进程,两者通过 **ZMQ PUSH/PULL** 通信 —— input queue 收 request,output queue 推 `EngineCoreOutputs`。EngineCore OOM 不会拖死 API server,子进程崩了 API server 抛 `EngineDeadError` → 503。

---

## 1. `EngineCore` 三段式构造(`core.py:94-176`)

```python
class EngineCore:
    def __init__(self, vllm_config, executor_class, log_stats, ...):
        load_general_plugins()                                     # 加载 vllm-ascend 的 register/platform patches

        self.model_executor = executor_class(vllm_config)         # L121  Executor 创建(此处触发 worker 子进程拉起)
        kv_cache_config = self._initialize_kv_caches(vllm_config)  # L131  分配 KV cache block pool,占显存
        self.structured_output_manager = StructuredOutputManager(vllm_config)

        Scheduler = vllm_config.scheduler_config.get_scheduler_cls()  # L135  vllm-ascend 这里可换成 BalanceScheduler
        self.scheduler = Scheduler(vllm_config, kv_cache_config, ...)  # L148  构造 Scheduler
        ...
```

**关键设计**:构造阶段一次性做完 ① 加载 plugin → ② Executor/worker 拉起 → ③ KV cache block pool 分配 → ④ Scheduler 构造;后面 `run_busy_loop` 只做"接请求 → 调度 → 跑 → 回包"四件事。`get_scheduler_cls()` 是 vllm-ascend 的 `patch_balance_schedule` 的注入点 —— `enable_balance_scheduling=True` 时返回 `BalanceScheduler`。

---

## 2. `run_busy_loop` 主循环(`core.py:1193-1252`)

```python
def run_busy_loop(self):
    """Core busy loop of the EngineCore."""
    while self._handle_shutdown():                  # L1195  shutdown_state == RUNNING 才继续
        # 1) 阻塞等请求,直到有 work
        self._process_input_queue()                # L1197  处理 input_queue 的 ADD/ABORT/PAUSE 等
        # 2) 调度 + 跑模型 + 回包
        self._process_engine_step()                # L1199  调用 step_fn() + 把 outputs 塞 output_queue
    raise SystemExit

def _process_input_queue(self):
    while not self.has_work() and self.is_running():           # L1207  没活干就阻塞
        ...
        req = self.input_queue.get(block=process_input_queue_block)  # L1219  无新请求时阻塞
        self._handle_client_request(*req)                      # L1220  按 type 分派(ADD/ABORT/...)
    # 有活干了:把剩下的请求一次性 drain 完
    while not self.input_queue.empty():
        req = self.input_queue.get_nowait()
        self._handle_client_request(*req)                      # L1232

def _process_engine_step(self) -> bool:
    outputs, model_executed = self.step_fn()                    # L1238  ★ 调度+执行
    for output in outputs.items() if outputs else ():
        self.output_queue.put_nowait(output)                    # L1241  ZMQ PULL 给 API server
    self.post_step(model_executed)                             # L1243  spec decode draft token 处理
    return model_executed
```

**关键设计**:`_handle_client_request` 是按 `EngineCoreRequestType` 分派的 dispatch:ADD → 构造 `Request` → `self.scheduler.add_request`;ABORT → `self.scheduler.finish_requests`;PAUSE/RESUME → `set_pause_state`;WAKEUP → 唤醒 idle 引擎。**整个主循环只有一个 `while` + 两个 `_process_*`**,是 vllm v1 架构的核心节奏。

---

## 3. `step()` 调度 + 执行 + 更新(`core.py:428-457`)

```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    """Schedule, execute, and make output."""
    if not self.scheduler.has_requests():                                # L437  没活直接退出
        return {}, False

    # 1) 调度:选 batch + 分配 KV blocks
    scheduler_output = self.scheduler.schedule()                         # L439  ★ SchedulerOutput

    # 2) 执行:把 SchedulerOutput 交给 executor(非阻塞,future 模式)
    future = self.model_executor.execute_model(scheduler_output, non_block=True)  # L440

    # 3) 采样:grammar-guided bitmask + speculative sampling
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)        # L441
    with self.log_error_detail(scheduler_output), self.log_iteration_details(scheduler_output):
        model_output = future.result()                                            # L446  等待 GPU
        if model_output is None:                                                  # L447  decode-only 走非阻塞路径
            model_output = self.model_executor.sample_tokens(grammar_output)     # L448

    # 4) 处理 abort 队列(并发路径)
    self._process_aborts_queue()                                                 # L452

    # 5) 用 model output 更新 scheduler state,产出 EngineCoreOutputs
    engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)  # L453

    return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

**关键设计**:**`execute_model` + `sample_tokens` 是两条独立通道**。decode batch 走 `execute_model` 之后**不阻塞等待 GPU 算完**就能返回;可与下一轮的 `sample_tokens` 流水线重叠。`total_num_scheduled_tokens > 0` 是 step 是否"实际算了一轮模型"的标志,idle 状态下会跳过整个 step。

---

## 4. `Scheduler.schedule()` 调度算法(`scheduler.py:329-595`)

```python
def schedule(self) -> SchedulerOutput:
    """选 batch:running → waiting → prefixed;预 KV block;返回 SchedulerOutput"""
    scheduled_new_reqs: list[Request] = []
    scheduled_resumed_reqs: list[Request] = []
    scheduled_running_reqs: list[Request] = []
    preempted_reqs: list[Request] = []

    req_to_new_blocks: dict[str, KVCacheBlocks] = {}
    num_scheduled_tokens: dict[str, int] = {}
    token_budget = self.max_num_scheduled_tokens                      # L348  一次能跑的总 token 上限

    self.kv_cache_manager.new_step_starts()                            # L362  ★ 推进 block pool 的 cached/free 状态

    # Step 1: schedule RUNNING requests(优先)
    req_index = 0
    while req_index < len(self.running) and token_budget > 0:
        request = self.running[req_index]
        num_new_tokens = (request.num_tokens_with_spec
                          + request.num_output_placeholders
                          - request.num_computed_tokens)              # L385  还没算的 token 数
        if 0 < long_prefill_token_threshold < num_new_tokens:
            num_new_tokens = long_prefill_token_threshold             # chunked-prefill 上限
        num_new_tokens = min(num_new_tokens, token_budget, max_model_len - 1 - request.num_computed_tokens)

        # KV block 分配
        with record_function_or_nullcontext("schedule: allocate_slots"):  # L442  profiler hook
            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, num_lookahead_tokens=...)
                if new_blocks is not None: break                       # 拿到了
                # 拿不到:preempt(抢占)最低优先级 request
                if policy == PRIORITY: preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
                else:                 preempted_req = self.running.pop()
                self._preempt_request(preempted_req, ...)             # 释放 KV blocks,求重新 prefill
                preempted_reqs.append(preempted_req)
                if preempted_req == request: break                    # 已经 preempt 到自己,放弃本次

        scheduled_running_reqs.append(request)
        req_to_new_blocks[request_id] = new_blocks
        num_scheduled_tokens[request_id] = num_new_tokens
        token_budget -= num_new_tokens
        req_index += 1

    # Step 2: schedule WAITING requests(从 waiting 队列按调度策略拉)
    if not preempted_reqs:                                            # L690  有 preempt 就先不拉新
        while req_index < len(self.waiting) and token_budget > 0:
            ... 同上模式 ...

    # Step 3: 构造 SchedulerOutput(return 给 step())
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=...,                  # 本次首次调度(worker 缓存 prompt)
        scheduled_cached_reqs=...,               # request 状态 diff(working set)
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        scheduled_spec_decode_tokens=...,
        ...
    )
    return scheduler_output
```

**关键设计**:
- **"无 prefill / decode 之分"**:vllm v1 把每个 request 看成 `num_computed_tokens` 追赶 `num_tokens_with_spec` 的过程,**chunked prefill 是这套框架的自然结果** —— 不是开关,是一致算法下的特例。
- **三遍循环**:① RUNNING(已 prefill 的一部分,优先,decode 友好)→ ② WAITING(新来或被 preempt 重排队)→ ③ 收尾构造 SchedulerOutput。每遍都按 `token_budget` 截断,超出部分下轮再跑。
- **Preempt 机制**:KV block 拿不到时**抢占最低优先级 request**,释放它的 blocks 重新分配。preempted request 下一轮会重新 prefill(代价高),所以尽量避免 —— `policy == PRIORITY` 时挑 priority 最低的,否则弹队尾。
- **Prefix cache**:同一个 prompt 的若干个 request 共享 prefix blocks;`scheduler_output.num_common_prefix_blocks` 就是 prefix 重用的提示,worker 用它做 cascade attention。

---

## 5. `SchedulerOutput` 数据结构(`output.py:181-256`)

```python
@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]            # 本次首次调度(worker 缓存 prompt)
    scheduled_cached_reqs: CachedRequestData            # request 状态 diff(working set)
    num_scheduled_tokens: dict[str, int]                # req_id → 几个 token
    total_num_scheduled_tokens: int                     # sum(num_scheduled_tokens.values())
    scheduled_spec_decode_tokens: dict[str, list[int]]
    scheduled_encoder_inputs: dict[str, list[int]]
    num_common_prefix_blocks: list[int]                 # cascade attention 用
    finished_req_ids: set[str]                          # 上轮之间已 finished 的
    free_encoder_mm_hashes: list[str]
    preempted_req_ids: set[str] | None
    kv_connector_metadata: KVConnectorMetadata | None   # PD 分离 / KV-transfer
    new_block_ids_to_zero: list[int] | None             # 新分配的 block,worker 要 zero 显存
```

**关键设计**:`SchedulerOutput` 是 **CPU 侧的纯数据对象**,只描述"这一轮要算什么";**Worker 端要做的事在 model_runner 里有对偶的映射** —— `num_scheduled_tokens` → input_ids 切片、`req_to_new_blocks` → block table、`new_block_ids_to_zero` → zero GPU memory。这种"调度/执行严格解耦"是 vllm v1 跨平台/跨 backend 的关键。

---

## 6. `KVCacheManager.allocate_slots`(`kv_cache_manager.py:236`)

```python
def allocate_slots(self, request, num_tokens, num_lookahead_tokens=0) -> KVCacheBlocks | None:
    """给 request 分配 num_tokens 的 KV cache block;返回 None 表示分配失败(需要 preempt)"""
    # 1) 计算需要多少新 block
    new_blocks = self.coordinator.get_new_blocks(request, num_tokens, num_lookahead_tokens)

    # 2) 命中 prefix cache?(同 prompt 已有 block)
    #    命中就免费获得 prefix blocks,只分配 suffix blocks

    # 3) block pool 里按 LRU 找空 block
    #    没有空 block → 逐出空闲 block(eviction)或返回 None(让调度器 preempt)

    # 4) 记录每个 request 占用的 block_ids
    if success:
        return KVCacheBlocks(
            block_ids=[...],
            computed_blocks=...,
            new_blocks=...,
        )
    return None
```

**关键设计**:
- **Block pool** 是 vllm 物理 KV cache 的抽象:把 GPU 显存切成等大的 block(典型 16 token/block),所有 request 共享。`block_pool` 维护 free/cached 两套状态 —— `free` 是从未分配过的,`cached` 是 prefix cache 命中可复用的。
- **Prefix cache 命中** = `BlockHash` 命中:相同 prefix 的 prompt 复用 block,只在不同处补 block。**这是 vllm 长 system prompt 场景下吞吐大幅提升的关键**。
- **`new_block_ids_to_zero`**:新分配的 block 物理显存里是上一轮被 evict 的内容(可能含 NaN),worker 端要在 attention 计算前 zero 掉,避免污染。

---

## 7. `Executor.execute_model`(`vllm/v1/executor/`)

```python
# 抽象接口
class Executor(ABC):
    def execute_model(self, scheduler_output: SchedulerOutput, non_block: bool = False) -> Future[ModelRunnerOutput]:
        """把 SchedulerOutput 派发到 worker(s);返回 future,non_block=True 时不等"""
```

```python
# MultiprocExecutor (多进程 worker)
def execute_model(self, scheduler_output, non_block=False):
    # 1) SchedulerOutput msgpack 序列化(走 RDMA / NCCL / shared memory 取决于 backend)
    # 2) 通过 Pipe / SharedMemory 推到所有 worker 进程
    # 3) 每个 worker 跑 ModelRunner.execute_model(scheduler_output)
    # 4) 各 worker 跑完后 reduce(EPLB 场景下做 expert parallel all-to-all)
    # 5) sample_tokens 在 rank 0 上跑
    return future  # Future[ModelRunnerOutput]
```

**关键设计**:
- **`SchedulerOutput` 跨进程传递** —— msgpack 序列化;对于 token-heavy 数据用 `tensor_ipc.py` 共享 GPU tensor(避免拷贝)。
- **TP > 1 时,每个 GPU 跑一个 worker**;SchedulerOutput 在 rank 0 上构造,广播到所有 rank。
- **`sample_tokens` 只在 rank 0 上跑** —— logits 必须在最后一层 gather 后才能算,decode-only 路径允许 decode 模型把 sample 提前到下一轮 step 的并行通道(vllm-ascend 的 `patch_balance_schedule` 启用了这条优化路径)。
- **vllm-ascend 注入点**:`patch_multiproc_executor.py` 改 `daemon=False`(EPLB 需要);`patch_distributed.py` 改 `torch.distributed.all_reduce` / `broadcast` 给 310p 做 tensor alignment;`patch_attention.py` 替换 attention backend 选择(改成 ascend NPU 的 MLA 实现)。

---

## 8. `Scheduler.update_from_output` 把 token 推回(`scheduler.py:1283-`)

```python
def update_from_output(self, scheduler_output, model_runner_output) -> dict[int, EngineCoreOutputs]:
    """用 model output 更新 scheduler 状态,产出 EngineCoreOutputs 推回 API server"""
    sampled_token_ids = model_runner_output.sampled_token_ids
    num_scheduled_tokens = scheduler_output.num_scheduled_tokens

    for req_id, num_tokens_scheduled in num_scheduled_tokens.items():  # L1347  1K+ request 热路径
        request = self.requests.get(req_id)
        if request is None or request.is_finished(): continue                  # 并发 abort 路径

        req_index = model_runner_output.req_id_to_index[req_id]
        generated_token_ids = sampled_token_ids[req_index] if sampled_token_ids else []

        # 1) 拼 output
        outputs[client_index].append(EngineCoreOutput(
            request_id=req_id,
            new_token_ids=generated_token_ids,
            finish_reason=...,
            stop_reason=...,
        ))

        # 2) 更新 request 状态
        request.num_computed_tokens += num_tokens_scheduled
        if request.is_finished():
            self.finish_requests(req_id, ...)                          # 释放 KV blocks

    # 3) 处理 spec decode 接受/拒绝
    self._update_spec_decode_state(...)

    # 4) 处理 routed_experts(MoE 路由信息,用于下一轮 EPLB)
    if model_runner_output.routed_experts is not None:
        self.routed_experts_mgr.store_batch(...)

    # 5) 处理 KV connector metadata(PD 分离 / async transfer)
    if self.connector:
        self.connector.update_state_after_step(...)

    return outputs
```

**关键设计**:
- **`update_from_output` 是 CPU 侧的 bookkeeping** —— 拿 GPU 输出的 token ids,把它们分到每个 request,更新 `num_computed_tokens`,处理 finish/abort,处理 spec decode 接受率。
- **`outputs` 是 `dict[int, EngineCoreOutputs]`**(按 `client_index` 分桶,DP 多 client 场景),`run_busy_loop` 把它们推到 output_queue → API server 的 `output_handler` → per-request `q.put(RequestOutput)` → `AsyncLLM.generate` 的 `yield` → `StreamingResponse` SSE 帧。
- **routed_experts 持久化**:vllm-ascend 的 `patch_routed_experts_capture.py` 在 worker 端把每个 step 的 MoE 路由结果 D2H 到 CPU,scheduler 端 `store_batch` 存到 `routed_experts_by_slot` 槽位,供 EPLB(动态 expert load balance)下一轮调度决策用。

---

## 9. vllm-ascend 在调度/执行路径上的关键 patch

vllm-ascend 在 §1-§8 这条主链上替换了多处关键对象(全部在 `vllm_ascend/patch/` 下):

| 阶段 | vllm 上游 | vllm-ascend 替换 | 文件 |
|---|---|---|---|
| Scheduler 构造 | `Scheduler` | `BalanceScheduler`(DP 场景下 `balance_gather` 同步 running 数量,跨 rank 避免单引擎过载) | `patch_balance_schedule.py` |
| Scheduler 实例化 | `DPEngineCoreProc` | `BalanceDPEngineCoreProc`(`run_busy_loop` 加 `balance_gather` + `engines_running` 判断) | `patch_balance_schedule.py:600` |
| `run_engine_core` | `EngineCoreProc.run_engine_core` | 替换版(接管信号 + DP 引擎构造) | `patch_balance_schedule.py:646` |
| KVCacheCoordinator | `HybridKVCacheCoordinator` | `AscendHybridKVCacheCoordinator`(DeepSeek-V4 等 hybrid 模型走 ascend 特化路径) | `patch_kv_cache_coordinator.py:58` |
| KVCacheSpec | `MLAAttentionSpec` / `SlidingWindowMLASpec` | `AscendMLAAttentionSpec` / `AscendSlidingWindowMLASpec` | `patch_kv_cache_interface.py:30, 215` |
| block sizes | `resolve_kv_cache_block_sizes` | `_ascend_resolve_kv_cache_block_sizes` | `patch_kv_cache_utils.py:23` |
| `_initialize_kv_caches` | 上游默认 | ascend 重写 block pool 初始化(兼容 310p 的内存对齐) | `patch_kv_cache_utils.py` |
| MultiprocExecutor | `daemon=True` | `daemon=False`(EPLB 要求子进程可 fork) | `patch_multiproc_executor.py` |
| `torch.distributed.all_reduce` | NCCL 原生 | ascend 适配(310p tensor alignment) | `patch_distributed.py` |
| Attention backend | 上游默认选 FlashAttn / xformers | ascend NPU MLA/Prefill backend | `patch_mla_prefill_backend.py` |

**典型一段**(`patch_balance_schedule.py:646`)展示 vllm-ascend 怎么注入 `run_engine_core`:

```python
def run_engine_core(*args, dp_rank=0, local_dp_rank=0, **kwargs):
    vllm_config = kwargs.get("vllm_config")
    if not _balance_scheduling_enabled(vllm_config):
        return _ORIGINAL_RUN_ENGINE_CORE(*args, dp_rank=dp_rank, local_dp_rank=local_dp_rank, **kwargs)
    # 否则接管:重写 signal handler + EngineCoreProc 构造 + DP 引擎构造
    ...
```

**`patch_balance_schedule` 的本质**:vllm v1 默认调度策略是 **chunked prefill**(prefill 和 decode 混跑),在 NPU 上不同长度 chunk 性能差异大,跨 DP rank 同 shape 调度更友好 —— ascend 加了 `enable_balance_scheduling` 用户开关,开启后走 `BalanceScheduler` + `BalanceDPEngineCoreProc`,刻意控制每个 DP rank 的 running 数量相近,减少异构 chunk 带来的 SM 闲置。

---

## 10. 完整流程图(EngineCore 内)

```
[ API server 进程 ]                                    [ EngineCore 子进程 ]
                    ZMQ PUSH
AsyncLLM.generate ─────────────────────────────────────► input_queue
                                                            │
                                                            ▼
                                                    _process_input_queue
                                                    _handle_client_request(type=ADD)
                                                            │
                                                            ▼
                                                    scheduler.add_request(request)
                                                            │  进入 self.waiting
                                                            │
                                                       ┌────┴────┐
                                                       │ run_busy_loop while True
                                                       └────┬────┘
                                                            ▼
                                                    _process_engine_step()
                                                            │
                                                            ▼
                                            step() ──────────────────►
                                                1) scheduler.schedule()  ──► SchedulerOutput
                                                  ├─ 遍历 running/waiting
                                                  ├─ kv_cache_manager.allocate_slots()
                                                  ├─ preempt 必要时
                                                  └─ 构造 SchedulerOutput
                                                2) model_executor.execute_model(scheduler_output, non_block=True)
                                                  └─ workers(NPU) 执行 forward
                                                3) model_executor.sample_tokens(grammar_output)
                                                4) scheduler.update_from_output()
                                                  ├─ 更新每个 request 的 num_computed_tokens
                                                  ├─ 处理 finished / preempted
                                                  └─ 构造 EngineCoreOutputs
                                                            │
                                                            ▼
                                                    output_queue.put_nowait(outputs)
                                                            │
                    ZMQ PULL                              │
AsyncLLM.output_handler ◄────────────────────────────────┘
   q.put(RequestOutput)
                                                            │
                                                            ▼
AsyncLLM.generate ── yield ──► StreamingResponse ──► SSE 字节流到客户端
```

---

## 11. vllm 关键类速查

| 角色 | 文件 | 关键方法 / 行 |
|---|---|---|
| EngineCore 子进程入口 | `vllm/v1/engine/core.py` | `EngineCoreProc.run_engine_core` L1093 |
| 主循环 | 同上 | `EngineCoreProc.run_busy_loop` L1193 |
| 调度主入口 | 同上 | `EngineCore.step` L428 |
| Scheduler 接口 | `vllm/v1/core/sched/interface.py` | `SchedulerInterface` L36 |
| Scheduler 实现 | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` L329 / `add_request` L1755 / `update_from_output` L1283 |
| Scheduler 调度算法 | 同上 | `_try_schedule_encoder_inputs` / `_preempt_request` / `_update_after_schedule` |
| SchedulerOutput | `vllm/v1/core/sched/output.py` | `SchedulerOutput` L181 |
| KVCacheManager | `vllm/v1/core/kv_cache_manager.py` | `allocate_slots` L236 / `free` L429 / `new_step_starts` L568 |
| KVCacheCoordinator | `vllm/v1/core/kv_cache_coordinator.py` | `get_new_blocks` / `free` L211 / `new_step_starts` L270 |
| Executor 抽象 | `vllm/v1/executor/` | `Executor.execute_model` |
| Ascend Scheduler | `vllm_ascend/patch/platform/patch_balance_schedule.py` | `BalanceScheduler` L36 / `BalanceDPEngineCoreProc` L600 / `run_engine_core` L646 |
| Ascend KVCache | `vllm_ascend/patch/platform/patch_kv_cache_coordinator.py` | `AscendHybridKVCacheCoordinator` L58 |
| Ascend KVCacheSpec | `vllm_ascend/patch/platform/patch_kv_cache_interface.py` | `AscendMLAAttentionSpec` L30 |
| Ascend Executor | `vllm_ascend/patch/platform/patch_multiproc_executor.py` | `daemon=False` |
| Ascend Distributed | `vllm_ascend/patch/platform/patch_distributed.py` | `all_reduce` / `broadcast` 适配 |
| Ascend Attention | `vllm_ascend/patch/platform/patch_mla_prefill_backend.py` | NPU MLA backend |

---

## 12. 下一步可以深入的入口

- **`Scheduler` 的 spec decode 路径**(`scheduler.py:1283-1350`):投机解码的 draft token 接受/拒绝、`update_draft_token_ids`、prefix caching 兼容性 —— vllm-ascend 的 `patch_deepseek_mtp.py` / `patch_qwen3_next_mtp.py` 在此修改。
- **`KVCacheManager` 的 prefix cache 细节**(`kv_cache_manager.py`):`BlockHash` 算法、eviction 策略(默认 LRU);vllm-ascend 的 `patch_kv_cache_utils.py` 重点改这块。
- **`Executor` 子进程通信**:`vllm/v1/executor/multiproc_executor.py`,msgpack 序列化 + tensor_ipc 共享;vllm-ascend 的 `patch_distributed.py` 影响这条路径。
- **`OutputProcessor` 与 detokenizer**(`vllm/v1/engine/output_processor.py`):token id → 字符串、logprob、stop reason;与 stage2 §4 `output_handler` 的对应。
- **`patch_balance_schedule.py` 完整版**:推荐对照 vllm v1.0 的 Scheduler PR 一起读,看 ascend 改了哪几行 —— 是掌握 vllm-ascend 调度哲学最快的入口。
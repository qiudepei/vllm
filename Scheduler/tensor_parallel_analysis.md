# vllm-ascend Tensor Parallelism (TP) 实现分析

> 基于 vLLM + vllm-ascend 插件，分析 Tensor Parallelism 在 Ascend NPU 上的完整实现，包括组初始化、权重重分片、前向通信、采样器 TP 协同。

---

## 目录

1. [总体设计](#1-总体设计)
2. [TP 通信组初始化](#2-tp-通信组初始化)
3. [并行线性层与权重重分片](#3-并行线性层与权重重分片)
4. [Column + Row 前向通信模式](#4-column--row-前向通信模式)
5. [各模块自定义 TP Op](#5-各模块自定义-tp-op)
6. [采样器 TP 协同](#6-采样器-tp-协同)
7. [FlashComm2 OProj TP](#7-flashcomm2-oproj-tp)
8. [Fine-Grained TP 配置](#8-fine-grained-tp-配置)
9. [关键文件索引](#9-关键文件索引)

---

## 1. 总体设计

Tensor Parallelism 将单个 Transformer 层的计算和参数拆分到多个 NPU 设备上，每个设备持有权重分片并独立计算，通过 HCCL 通信聚合结果。vllm-ascend 的 TP 实现覆盖：

- **并行线性层**：ColumnParallel / RowParallel / Replicated
- **模块级细粒度 TP**：MLP / OProj / Embedding / LMHead 使用不同的 TP size
- **通信-计算融合**：`npu_mm_all_reduce_base` / `npu_mm_reduce_scatter_base` 等融合算子
- **FlashComm2**：OProj 专用两级 TP + DP 通信优化
- **采样器 TP**：all-gather 协同跨 TP rank 的 logits 合并

**核心文件**：
- `vllm_ascend/distributed/parallel_state.py` — TP 组初始化
- `vllm_ascend/ops/linear_op.py` — 自定义前向 Op 路由与实现
- `vllm_ascend/ops/register_custom_ops.py` — 通信融合算子注册
- `vllm_ascend/ops/linear.py` — 并行线性层类定义

---

## 2. TP 通信组初始化

### 2.1 启动入口

`NPUWorker.init_device()` → `_init_worker_distributed_environment()`：

```python
# vllm_ascend/worker/worker.py:755
def _init_worker_distributed_environment(self):
    init_distributed_environment(
        self.parallel_config.world_size, self.rank,
        self.distributed_init_method, self.local_rank, "hccl"
    )
    ensure_model_parallel_initialized(
        self.parallel_config.tensor_parallel_size,
        self.parallel_config.pipeline_parallel_size,
    )
    init_ascend_model_parallel(self.parallel_config)
```

调用链：`init_distributed_environment()`（vLLM 标准初始化，backend="hccl"）→ `init_ascend_model_parallel()`（创建 Ascend 专用通信组）。

### 2.2 Rank 空间布局

`vllm_ascend/distributed/parallel_state.py:46`：

```python
all_ranks = torch.arange(world_size).reshape(
    -1,                      # ExternalDP（外部 DP 空间）
    global_dp_size,          # DP: Data Parallel
    global_pp_size,          # PP: Pipeline Parallel
    global_pcp_size,         # PCP: Prefill Context Parallel
    global_tp_size,          # TP: Tensor Parallel
)
```

5 维 rank 空间，`ExternalDP` 维度用于 verl 等外部框架的 DP 分组。

### 2.3 创建的通信组

| 全局变量 | 组名 | 用途 | 依赖条件 |
|----------|------|------|----------|
| `_MC2` | `"mc2"` | MC2 op 通信（EP 级别） | 始终创建 |
| `_P_TP` | `"p_tp_{num}"` | Prefill TP（PD 分离） | `pd_tp_ratio > 1` |
| `_OTP` | `"otp"` | Attention OProj TP | `finegrained_tp_config.oproj_tensor_parallel_size > 0` |
| `_LMTP` | `"lmheadtp"` | LM Head TP | `lmhead_tensor_parallel_size > 0` |
| `_EMBED_TP` | `"emtp"` | Embedding TP | `embedding_tensor_parallel_size > 0` |
| `_MLP_TP` | `"mlptp"` | MLP TP | `mlp_tensor_parallel_size > 0` |
| `_FLASHCOMM2_OTP` | `"flashcomm2_otp"` | FlashComm2 输出 TP | `flashcomm2_enable()` |
| `_FLASHCOMM2_ODP` | `"flashcomm2_odp"` | FlashComm2 输出 DP | `flashcomm2_enable()` |
| `_SHARD_WEIGHT` | `"shard_weight"` | 跨 PP stage 权重重分片 | `layer_sharding` 启用 |
| `_FC3_QUANT_X` | `"fc3_quant_x"` | FC3 量化 all-gather | `multistream_overlap_gate` |
| `_DYNAMIC_EPLB` | `"dynamic_eplb"` | 动态 Expert 负载均衡 | `dynamic_eplb` |

### 2.4 MC2 组（EP 级别通信）

`vllm_ascend/distributed/parallel_state.py:84-96`：

```python
group_ranks = (
    all_ranks.transpose(1, 2)
    .reshape(-1, global_dp_size * global_pcp_size * global_tp_size)
    .unbind(0)
)
_MC2 = init_model_parallel_group(group_ranks, ..., group_name="mc2")
```

通过 `transpose(1, 2)` 将 PP 维度移到最前面，使每个组包含所有非 PP rank，在 MoE EP 场景中跨 DP 和 TP rank 通信。

### 2.5 细粒度 TP 组创建

`_create_or_get_group()`（`vllm_ascend/distributed/parallel_state.py:117-133`）：

```python
def _create_or_get_group(group_size, group_name):
    rank_grid = torch.arange(world_size).reshape(global_pp_size, global_dp_size, global_tp_size)
    num_chunks = global_dp_size // group_size
    for pp_idx in range(global_pp_size):
        stage_ranks = rank_grid[pp_idx]
        for chunk in range(num_chunks):
            for tp_idx in range(global_tp_size):
                group = stage_ranks[chunk * group_size : (chunk + 1) * group_size, tp_idx].tolist()
                group_ranks.append(group)
```

逻辑：在每个 PP stage 内，将 DP 维度按 `group_size` 切块，每块内部 + 相同 TP rank 构成一个通信组。当 `oproj_tp_size = DP_size` 时，组内包含所有 DP rank，退化为标准 TP 组。

---

## 3. 并行线性层与权重重分片

### 3.1 并行线性层继承体系

`vllm_ascend/ops/linear.py`：

| Ascend 类 | 父类（vLLM） | 含义 |
|-----------|-------------|------|
| `AscendColumnParallelLinear` | `ColumnParallelLinear` | 权重沿输出维度切分 |
| `AscendRowParallelLinear` | `RowParallelLinear` | 权重沿输入维度切分 |
| `AscendMergedColumnParallelLinear` | `MergedColumnParallelLinear` | 合并的 ColumnParallel（如 Gate + Up） |
| `AscendQKVParallelLinear` | `QKVParallelLinear` | QKV 投影（head 维度切分） |
| `AscendReplicatedLinear` | `ReplicatedLinear` | 不切分，各 rank 持有完整权重 |

### 3.2 权重初始化切分

`AscendColumnParallelLinear.__init__()`：

```python
self.custom_op, self.tp_rank, self.tp_size = get_parallel_op(disable_tp, prefix, self, "column")
self.output_size_per_partition = divide(output_size, self.tp_size)
```

每个 rank 只初始化 `output_size / tp_size` 列。`AscendRowParallelLinear` 类似，切分 `input_size`。

### 3.3 Op 路由 `get_parallel_op()`

`vllm_ascend/ops/linear_op.py:694-723`：

```python
def get_parallel_op(disable_tp, prefix, layer, direct):
    if disable_tp or shared_expert_dp_enabled():
        return None, 0, 1
    if direct == "row":
        custom_op = _get_row_parallel_op(prefix, layer)
    if direct == "column":
        custom_op = _get_column_parallel_op(prefix, layer)
    if custom_op is not None:
        return custom_op, custom_op.tp_rank, custom_op.tp_size
    return None, get_tp_group().rank_in_group, get_tp_group().world_size
```

根据 `prefix`（层名）路由到不同的自定义 Op，决定使用哪个通信组和通信策略。

---

## 4. Column + Row 前向通信模式

### 4.1 Column Parallel 模式

```python
# MLPColumnParallelOp.apply_impl()
input_parallel = self.comm_group.all_gather(input_, 0)   # all-gather 输入
output = self.quant_method.apply(self.layer, input_parallel, bias)
return output  # 输出自然 sharded，无需通信
```

**数据流**：All-gather 全量输入 → 独立计算 → 输出自然分片。

### 4.2 Row Parallel 模式

```python
# MLPRowParallelOp.apply_impl()
input_parallel = self.get_input_parallel(input_)          # split 输入
output_parallel = quant_method.apply(self.layer, input_parallel, bias)
output = self.comm_group.reduce_scatter(output_parallel, 0)  # reduce-scatter 聚合
return output
```

**数据流**：切分输入 → 独立计算 → Reduce-scatter 聚合输出。

### 4.3 Column+Row 串联

MLP 层的完整 TP 模式：

```
输入                       输出
  │                         ▲
  ▼                         │
[Column: all-gather]      [Row: reduce-scatter]
  │                         │
  ▼                         │
Gate/Up 投影 (sharded)   Down 投影 (sharded)
  │              │           ▲
  │              └──── Silu ─┘
  │                  (element-wise)
  └─────────────────→ SiLU(Gate) * Up
```

### 4.4 Attention OProj 的 All-to-All 模式

`OProjRowParallelOp.apply_impl()`（`vllm_ascend/ops/linear_op.py:231-264`）：

```python
# 1) All-to-all 重新分配数据（按 head 维度）
send_buf = input_parallel.reshape(-1, tp_size, chunk_size).transpose(0, 1).contiguous()
dist.all_to_all_single(recv_buf, send_buf, group=self.comm_group.device_group)

# 2) 矩阵乘
output_parallel = quant_method.apply(self.layer, input_parallel, bias)

# 3) Reduce-scatter 聚合
output = self.comm_group.reduce_scatter(output_parallel, dim=0)
```

**OProj 的三段模式**：All-to-all（head 重分配）→ Matmul → Reduce-scatter。

---

## 5. 各模块自定义 TP Op

### 5.1 Op 继承体系

```
CustomLinearOp
├── CustomColumnParallelOp
│   ├── MLPColumnParallelOp          (comm_group = get_mlp_tp_group())
│   ├── SequenceColumnParallelOp     (comm_group = get_tp_group(), FlashComm)
│   ├── Flashcomm2OshardQKVParallelOp
│   └── ShardedCPColumnParallelOp    (fake group, tp_size=1)
└── CustomRowParallelOp
    ├── MLPRowParallelOp              (comm_group = get_mlp_tp_group())
    ├── OProjRowParallelOp           (comm_group = get_otp_group(), all-to-all)
    ├── Flashcomm2OProjRowParallelOp (两级 otp + odp 通信)
    ├── MatmulAllreduceRowParallelOp (fused npu_mm_all_reduce_base)
    ├── SequenceRowParallelOp        (FlashComm + fusion)
    └── ShardedCPRowParallelOp       (fake group)
└── CustomReplicatedOp               (无 TP 通信)
```

### 5.2 Column Parallel 路由表

`_get_column_parallel_op()`（`vllm_ascend/ops/linear_op.py:628-652`）：

| 条件 | Op | 组 |
|------|-----|-----|
| DSA CP + q/kv_b_proj | `ShardedCPColumnParallelOp` | fake（tp_size=1） |
| gate_up_proj + MLP TP + 非 MoE | `MLPColumnParallelOp` | `get_mlp_tp_group()` |
| flashcomm2_oshard + qkv_proj | `Flashcomm2OshardQKVParallelOp` | `get_tp_group()` |
| 启用了 SP + gate_up_proj/qkv_proj | `SequenceColumnParallelOp` | `get_tp_group()` |
| 其他 | `None`（fallback 到 vLLM 默认） | `get_tp_group()` |

### 5.3 Row Parallel 路由表

`_get_row_parallel_op()`（`vllm_ascend/ops/linear_op.py:655-691`）：

| 条件 | Op | 组 |
|------|-----|-----|
| DSA CP + o_proj | `ShardedCPRowParallelOp` | fake（tp_size=1） |
| down_proj + MLP TP + 非 MoE | `MLPRowParallelOp` | `get_mlp_tp_group()` |
| o_proj + OProj TP | `OProjRowParallelOp` | `get_otp_group()` |
| matmul_allreduce 启用 | `MatmulAllreduceRowParallelOp` | `get_tp_group()` |
| flashcomm2 + o/out_proj | `Flashcomm2OProjRowParallelOp` | `get_flashcomm2_otp_group()` |
| 启用了 SP + o/down_proj | `SequenceRowParallelOp` | `get_tp_group()` |
| 其他 | `None` | `get_tp_group()` |

### 5.4 Matmul+AllReduce 融合

`MatmulAllreduceRowParallelOp`（`vllm_ascend/ops/linear_op.py:385-406`）：

```python
output = torch_npu.npu_mm_all_reduce_base(
    input_parallel, self.layer.weight.t(), self.hcomm_info, bias=bias_
)
```

使用 Ascend NPU 提供的 `npu_mm_all_reduce_base` 融合算子，将矩阵乘和 all-reduce 合并为一次硬件调用，减少 HCCL 通信开销。

### 5.5 Sequence Parallelism 的通信-计算融合

`SequenceRowParallelOp.matmul_and_reduce()`（`vllm_ascend/ops/linear_op.py:500-577`）：

```python
# FlashComm1 非量化场景
output = torch_npu.npu_mm_reduce_scatter_base(
    x, self.layer.weight.t(), hcom_name, world_size,
    reduce_op="sum", bias=None, comm_turn=0, comm_mode="aiv",
)

# W8A8 量化场景
output = torch_npu.npu_mm_reduce_scatter_base(
    x_quant, self.layer.weight, hcom_name, world_size,
    reduce_op="sum", bias=None, x2_scale=deq_scale, output_dtype=torch.bfloat16, comm_mode="aiv",
)
```

### 5.6 自定义 Op 注册（通信融合层）

`vllm_ascend/ops/register_custom_ops.py` 注册的 TP 相关自定义算子：

| Op 名 | 实现函数 | 行为 |
|-------|---------|------|
| `maybe_all_gather_and_maybe_unpad` | `_maybe_all_gather_and_maybe_unpad_impl` | 条件性 all-gather 输入，FlashComm 下 unpadding |
| `maybe_pad_and_reduce` | `_maybe_pad_and_reduce_impl` | 条件性 padding + reduce-scatter |
| `maybe_all_reduce_tensor_model_parallel` | `_maybe_all_reduce_tensor_model_parallel_impl` | 条件性 all-reduce（MoE 最终输出） |
| `matmul_and_reduce` | `_matmul_and_reduce_impl` | matmul + reduce-scatter 融合 |
| `maybe_chunk_residual` | `_maybe_chunk_residual_impl` | Sequence Parallelism 下对 residual 切分（TP 维度） |

所有 custom op 都提供 `_impl` 和 `_fake` 两套实现（用于 ACL graph capture 的 shape 推导）。

---

## 6. 采样器 TP 协同

### 6.1 Greedy Sample 跨 TP All-Gather

`AscendSampler.greedy_sample()`（`vllm_ascend/sample/sampler.py:95-110`）：

```python
def greedy_sample(logits):
    if get_ascend_config().enable_reduce_sample:
        tp_group = get_tp_group()
        B, V_local = logits.shape
        rank = tp_group.rank_in_group

        local_max_logits, local_max_indices = logits.max(dim=-1)
        local_global_idx = local_max_indices + rank * V_local

        gathered_logits = tp_group.all_gather(local_max_logits.unsqueeze(-1), dim=-1)
        gathered_global_idx = tp_group.all_gather(local_global_idx.unsqueeze(-1), dim=-1)
        global_max_rank = gathered_logits.argmax(dim=-1)
        target_argmax = gathered_global_idx.gather(dim=-1, index=global_max_rank.unsqueeze(-1))

        return target_argmax
    else:
        return logits.argmax(dim=-1).view(-1)
```

每个 TP rank 保存 `V_local = V_global / tp_size` 列 logits。`enable_reduce_sample=True` 时，各 rank 计算本地 argmax 并通过两次 all-gather 收集全局最大值，选择全局 argmax token。

### 6.2 Top-K/Top-P + TP All-Gather

`_apply_top_k_top_p_pytorch()`（`vllm_ascend/sample/sampler.py:167-199`）：

```python
B, V_local = logits.shape
world_size = tp_group.world_size
rank = tp_group.rank_in_group
V_global = V_local * world_size

local_vals, local_idx = torch.topk(logits, k=top_k, dim=-1)
local_global_idx = local_idx + rank * V_local

gathered_vals = tp_group.all_gather(local_vals, dim=-1)
gathered_idx = tp_group.all_gather(local_global_idx, dim=-1)

full_logits = logits.new_full((B, V_global), -float("inf"))
full_logits.scatter_(dim=-1, index=gathered_idx, src=gathered_vals)
```

各 rank 计算 `top_k` 后，all-gather 合并所有候选，在完整词表空间中做 top-p/nucleus 过滤。

### 6.3 Rejection Sampler TP

`vllm_ascend/sample/rejection_sampler.py:186` — 使用 `tp_group.all_gather()` 跨 TP rank 收集 draft 和 target 模型的 logits，执行 rejection 决策。

---

## 7. FlashComm2 OProj TP

### 7.1 两级通信架构

`Flashcomm2OProjRowParallelOp`（`vllm_ascend/ops/linear_op.py:272-375`）：

```python
self.odp_group = get_flashcomm2_odp_group()     # DP 级通信组
self.otp_size = flashcomm2_oproj_tp_size        # OTP TP size（<= 全局 TP size）
self.comm_group = get_flashcomm2_otp_group()     # OTP 通信组
```

- **OTP 组**：`flashcomm2_otp_size` 个 rank 一组，做 reduce-scatter
- **ODP 组**：不同 OTP 组中相同位置的 rank 组成，做 all-to-all

### 7.2 创建逻辑

`vllm_ascend/distributed/parallel_state.py:152-189`：

```python
for dp_group_index in range(global_dp_size):
    for pp_group_index in range(global_pp_size):
        dp_pp_serial_index = dp_group_index * global_pp_size + pp_group_index
        tp_base_rank = dp_pp_serial_index * global_tp_size
        for i in range(num_fc2_oproj_tp_groups):
            ranks = []
            for j in range(flashcomm2_otp_size):
                tp_local_rank = i + j * num_fc2_oproj_tp_groups
                ranks.append(tp_base_rank + tp_local_rank)
            flashcomm2_otp_group_ranks.append(ranks)
```

在 TP 维度内创建 `global_tp_size / flashcomm2_otp_size` 个子组，每个子组包含 `flashcomm2_otp_size` 个 rank（取模分布）。

### 7.3 前向流程

```python
# 1) 切分输入
input_parallel = self.get_input_parallel(input_)

# 2) All-to-all（按 ODP 组重新分配 batch）
send_buf = reorganize(input_parallel, reorgnized_batch_ids)
dist.all_to_all_single(recv_buf, send_buf, group=self.odp_group.device_group)

# 3) Matmul（OTP 组内）
output_parallel = quant_method.apply(self.layer, input_parallel, bias)

# 4) Reduce-scatter（OTP 组内）
if self.tp_size > 1:
    output = self.comm_group.reduce_scatter(output_parallel, dim=0)

# 5) All-gather（全局 TP 组，非 FlashComm1 模式）
output = get_tp_group().all_gather(output, 0)
```

---

## 8. Fine-Grained TP 配置

### 8.1 配置项

`vllm_ascend/ascend_config.py:418` — `FinegrainedTPConfig`：

| 字段 | 含义 | 约束 |
|------|------|------|
| `oproj_tensor_parallel_size` | Attention OProj 的 TP size | `dp_size % value == 0` |
| `lmhead_tensor_parallel_size` | LM Head 的 TP size | 同上 |
| `embedding_tensor_parallel_size` | Embedding 的 TP size | 同上 |
| `mlp_tensor_parallel_size` | MLP 的 TP size | 同上 |

### 8.2 组创建逻辑

各模块的 TP size 可以小于全局 TP size。`_create_or_get_group()` 通过将 DP 维度切块来创建子组：

```python
num_chunks = global_dp_size // group_size
for chunk in range(num_chunks):
    for tp_idx in range(global_tp_size):
        group = stage_ranks[chunk * group_size : (chunk + 1) * group_size, tp_idx]
```

当 `group_size == global_dp_size` 时，只有一个 chunk，组内包含所有 DP rank，等价于标准 TP 组。

---

## 9. 关键文件索引

| 文件 | 内容 |
|------|------|
| `vllm_ascend/distributed/parallel_state.py` | `init_ascend_model_parallel()`、所有通信组 getter、ShardWeight 组创建 |
| `vllm_ascend/distributed/utils.py` | `split_tensor_along_first_dim()`、`all_gather_async()`、`fc3_all_gather_and_maybe_unpad_impl()` |
| `vllm_ascend/ops/linear.py` | `AscendColumnParallelLinear`、`AscendRowParallelLinear`、`AscendQKVParallelLinear` |
| `vllm_ascend/ops/linear_op.py` | 所有 Custom Op 实现（MLP/OProj/SP/FlashComm2/MatmulAllreduce） |
| `vllm_ascend/ops/register_custom_ops.py` | 通信融合算子（all_gather_and_unpad / pad_and_reduce / matmul_and_reduce） |
| `vllm_ascend/ops/vocab_parallel_embedding.py` | `AscendVocabParallelEmbedding`、`AscendParallelLMHead`、`AscendLogitsProcessor` |
| `vllm_ascend/sample/sampler.py` | `AscendSampler.greedy_sample()`（TP all-gather argmax） |
| `vllm_ascend/sample/rejection_sampler.py` | TP all-gather rejection |
| `vllm_ascend/worker/worker.py` | `NPUWorker._init_worker_distributed_environment()`、PP send/recv with TP group |
| `vllm_ascend/worker/model_runner_v1.py` | `_all_gather_hidden_states()`、SP pad |
| `vllm_ascend/ascend_config.py` | `FinegrainedTPConfig` |
| `vllm_ascend/spec_decode/llm_base_proposer.py` | 自定义 draft 模型 TP 组 |
| `vllm_ascend/ops/fused_moe/fused_moe.py` | MoE 条件 all-reduce |
| `vllm_ascend/distributed/device_communicators/pyhccl.py` | `PyHcclCommunicator`（libhccl.so Python 封装） |
| `vllm_ascend/distributed/device_communicators/npu_communicator.py` | `NPUCommunicator`（all_to_all） |

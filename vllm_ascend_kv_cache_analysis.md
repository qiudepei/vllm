# vllm-ascend KV 缓存方案详解

本文档基于 vllm-ascend 和 vllm 源码，详细分析 vllm-ascend 中的 KV 缓存实现方案，并与 vllm 基线进行对比。

---

## 一、整体架构

vllm-ascend 基于 vllm 的 **PagedAttention** 框架，将 KV 缓存按固定大小 Block（页）管理，但针对昇腾 NPU 做了大量适配和优化。系统支持三种注意力后端：

| 后端 | 适用模型 | KV 缓存结构 |
|------|---------|------------|
| `AscendAttentionBackend` | 标准 Transformer | `[2, num_blocks, block_size, num_kv_heads, head_size]` |
| `AscendMLABackend` | DeepSeek-V2/V3 (MLA) | k_cache (nope) + v_cache (rope)，无前置 dim=2 |
| `AscendSFABackend` | DeepSeek-V3.2 (Sparse Flash Attention) | k_cache + v_cache + dsa_k_cache [+ dsa_k_scale_cache] |

**Block Size 固定为 128**（vllm 基线支持 16/32 等多种，NPU 固定 128）。

---

## 二、KV 缓存数据结构与形状

### 2.1 标准 Attention KV 缓存

**文件**: `vllm_ascend/attention/attention_v1.py:92-98`

```python
@staticmethod
def get_kv_cache_shape(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
) -> tuple[int, ...]:
    return (2, num_blocks, block_size, num_kv_heads, head_size)
```

- 前置维度 `2` 将 K 和 V 分为同一张量的两个通道
- 形状为 `[2, num_blocks, block_size, num_kv_heads, head_size]`
- 但在实际分配时，K 和 V 缓存被存储为**独立张量**（详见第三节），以支持 Prefill-Decode 分离部署

### 2.2 MLA Attention KV 缓存

**文件**: `vllm_ascend/attention/mla_v1.py:86-87`

```python
@staticmethod
def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
    return num_blocks, block_size, num_kv_heads, head_size
```

- 无前置 dim=2，因为 MLA 存储的是 `kv_lora`（nope/rope）而非独立的 K/V
- 实际缓存拆为两个张量：
  - `k_cache`：存储 kv_lora_rank 部分（nope）
  - `v_cache`：存储 qk_rope_head_dim 部分（rope）
- reshape 时（`_reshape_kv_cache_tensors`，第 2948-2962 行）：k_dim 设为 `kv_lora_rank`，v_dim 设为 `qk_rope_head_dim`

### 2.3 SFA（Sparse Flash Attention）KV 缓存

**文件**: `vllm_ascend/attention/sfa_v1.py:87-88`

```python
@staticmethod
def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
    return (num_blocks, block_size, num_kv_heads, head_size)
```

SFA 在 MLA 基础上增加了 **sparse indexer** 的额外 KV 缓存条目。KV 缓存元组可以是：

- `(k_cache, v_cache, dsa_k_cache)` — 标准稀疏模式（3 个张量）
- `(k_cache, v_cache, dsa_k_cache, dsa_k_scale_cache)` — Sparse C8 量化模式（4 个张量）

其中：
- `k_cache`：存储 kv_lora（nope）分量
- `v_cache`：存储 k_rope 分量
- `dsa_k_cache`：存储 indexer 模块的 key 张量
- `dsa_k_scale_cache`：存储 indexer key 的量化缩放因子（仅 Sparse C8 启用时）

---

## 三、KV 缓存内存分配（NPU 关键差异）

**核心文件**: `vllm_ascend/worker/model_runner_v1.py:2704-2844`（`_allocate_kv_cache_tensors`）

### 3.1 K/V 分离分配

vllm 基线分配一个包含 K 和 V 的统一张量（dim0=2），vllm-ascend **始终将 K 和 V 分为独立张量**，以支持 PD 分离部署（Prefill-Decode Disaggregation）：

```python
k_tensor = torch.zeros(k_tensor_size, dtype=torch.int8, device=self.device)
v_tensor = torch.zeros(v_tensor_size, dtype=torch.int8, device=self.device)
```

### 3.2 2MB 内存对齐

当启用 `kv_transfer_config`（PD 分离）时，每个张量按 2MB 对齐，确保跨节点 KV 传输时的内存地址对齐：

```python
alignment = 2 * 1024 * 1024
k_tensor = torch.zeros(k_tensor_size + alignment, dtype=torch.int8, device=self.device)
k_tensor = self._align_memory(k_tensor, alignment)[:k_tensor_size]
```

### 3.3 K/V 内存比例分配

根据 `k_dim` 和 `v_dim` 的比例分配 K/V 张量大小，MLA 场景下 nope 和 rope 维度不同时尤其重要：

```python
k_dim, v_dim = self._get_attention_kv_cache_dims(layer_name, current_kv_cache_spec)
k_tensor_split_factor, v_tensor_split_factor = calc_split_factor(kv_head_dim_list)
```

### 3.4 Sparse C8 量化分配

对于 SFA 模型，额外分配 indexer key 和 scale 的张量：

```python
if self.use_sparse_c8_indexer:
    kv_cache_raw_tensors[layer_name_inner] = (k_tensor, v_tensor, dsa_k_tensor, dsa_k_scale_tensor)
```

### 3.5 张量 Reshape

**文件**: `vllm_ascend/worker/model_runner_v1.py:2846-3038`（`_reshape_kv_cache_tensors`）

原始 int8 张量被 reshape 为目标 dtype 和形状：

```python
# 标准注意力
k_cache = raw_k_tensor.view(k_cache_dtype).view(k_shape)
v_cache = raw_v_tensor.view(v_cache_dtype).view(v_shape)

# MLA（第 2948-2962 行）
k_dim, v_dim = self._get_attention_kv_cache_dims(layer_name, current_kv_cache_spec)
k_shape = (mla_num_blocks, mla_block_size, num_kv_heads, k_dim)   # kv_lora_rank
v_shape = (mla_num_blocks, mla_block_size, num_kv_heads, v_dim)   # qk_rope_head_dim

# Sparse C8 indexer（第 2978-2994 行）
dsa_k_cache_shape = (num_blocks, block_size, num_kv_heads, index_head_dim)
dsa_k_cache = raw_dsa_k_tensor.view(self.c8_k_cache_dtype).view(dsa_k_cache_shape)  # int8
dsa_k_scale_cache_shape = (num_blocks, block_size, num_kv_heads, 1)
dsa_k_scale_cache = raw_dsa_k_scale_tensor.view(self.c8_k_scale_dtype).view(dsa_k_scale_cache_shape)  # float16
```

---

## 四、KV 缓存写入（NPU 算子适配）

**核心文件**: `vllm_ascend/device/device_op.py`

### 4.1 标准 Attention 写入

**文件**: `vllm_ascend/attention/attention_v1.py:882-909`（`reshape_and_cache`）

```python
def reshape_and_cache(self, query, key, value, kv_cache, attn_metadata, output):
    if len(kv_cache) > 1:
        if self.key_cache is None:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
        slots = attn_metadata.slot_mapping
        DeviceOperator.reshape_and_cache(
            key=key[:attn_metadata.num_actual_tokens],
            value=value[:attn_metadata.num_actual_tokens],
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            slot_mapping=slots[:attn_metadata.num_actual_tokens],
        )
```

底层 NPU 算子因硬件型号不同而不同：

| 硬件 | 算子 | 位置 |
|------|------|------|
| 910B (A2) | `_npu_reshape_and_cache` | `device_op.py:31-33` |
| 910C (A5) | `npu_scatter_pa_kv_cache` | `device_op.py:204-208` |

```python
# A2 实现
@classmethod
def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
    torch_npu._npu_reshape_and_cache(
        key=key, value=value, key_cache=key_cache,
        value_cache=value_cache, slot_indices=slot_mapping
    )

# A5 实现
@classmethod
def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
    torch_npu.npu_scatter_pa_kv_cache(
        key=key, value=value.contiguous(), key_cache=key_cache,
        value_cache=value_cache, slot_mapping=slot_mapping
    )
```

### 4.2 MLA Decode 写入 — 融合算子

**文件**: `vllm_ascend/attention/mla_v1.py:1200-1226`（`exec_kv_decode`）

MLA Decode 阶段使用融合算子 `npu_kv_rmsnorm_rope_cache`，将三个操作融合为一个 NPU 算子：

```python
k_pe, k_nope, _, _ = torch_npu.npu_kv_rmsnorm_rope_cache(
    kv_no_split,
    self.kv_a_layernorm.weight,
    cos, sin,
    slots.to(torch.int64),
    kv_cache[1],  # rope cache (k_pe)
    kv_cache[0],  # nope cache (kv_lora)
    epsilon=self.kv_a_layernorm.variance_epsilon,
    cache_mode="PA_NZ" if self.enable_kv_nz else "PA",
)
```

**融合操作**: RMSNorm -> RoPE -> Cache写入，避免中间结果的显存写入/读取，是关键性能优化点。

### 4.3 SFA KV 缓存写入

**文件**: `vllm_ascend/attention/sfa_v1.py:746-788`

nope/rope 部分复用 MLA 的融合算子：

```python
torch_npu.npu_kv_rmsnorm_rope_cache(
    kv_no_split, self.kv_a_layernorm.weight, cos, sin,
    slots.to(torch.int64), kv_cache[1], kv_cache[0],
    epsilon=..., cache_mode="PA",
)
```

indexer key 部分使用 scatter 写入：

```python
# 第 1197-1207 行
torch_npu.npu_scatter_nd_update_(
    kv_cache[2].view(-1, k_li.shape[-1]),
    slot_mapping.view(-1, 1),
    k_li.view(-1, k_li.shape[-1])
)
if self.use_sparse_c8_indexer:
    torch_npu.npu_scatter_nd_update_(
        kv_cache[3].view(-1, k_li_scale.shape[-1]),
        slot_mapping.view(-1, 1),
        k_li_scale.view(-1, k_li_scale.shape[-1])
    )
```

---

## 五、KV 缓存读取（双路径注意力）

### 5.1 标准 Attention 双路径

**文件**: `vllm_ascend/attention/attention_v1.py:911-929`

Decode 阶段有两条路径：

1. **FIA (Flash-In-Attention) 路径**：当 `DecodeOnly` 且使用 PagedAttention 且非滑动窗口时
2. **PagedAttention 路径**：其他场景走标准 PagedAttention

```python
def forward_impl(self, query, key, value, kv_cache, attn_metadata, output):
    if (
        attn_metadata.attn_state == AscendAttentionState.DecodeOnly
        and using_paged_attention(num_tokens, self.vllm_config)
        and self.sliding_window is None
    ):
        output = self.forward_paged_attention(query, attn_metadata, output)
    else:
        # FIA 路径或其他
```

### 5.2 MLA Cache Load（Chunked Prefill）

**文件**: `vllm_ascend/device/device_op.py`

| 硬件 | 算子 | 位置 |
|------|------|------|
| 910B (A2) | `npu_paged_cache_load` | 第 190-200 行 |
| 910C (A5) | `npu_gather_pa_kv_cache` | 第 443-453 行 |

```python
# A2 实现
@staticmethod
def mla_cache_load(cache_kv_c, cache_k_pe, block_table, context_seq_len_npu, seq_starts, key, value):
    torch_npu.atb.npu_paged_cache_load(
        cache_kv_c, cache_k_pe, block_table, context_seq_len_npu,
        seq_starts=seq_starts, key=key, value=value,
    )

# A5 实现
@staticmethod
def mla_cache_load(cache_kv_c, cache_k_pe, block_table, context_seq_len_npu, seq_offset, key, value):
    torch_npu.npu_gather_pa_kv_cache(
        cache_kv_c, cache_k_pe, block_table, context_seq_len_npu,
        seq_offset=seq_offset, key=key, value=value,
    )
```

---

## 六、Block 管理（PagedAttention 页管理）

### 6.1 BlockTable

**文件**: `vllm_ascend/worker/block_table.py`

`BlockTable` 类管理逻辑序列位置到物理 KV 缓存 Block 的映射：

- **物理/逻辑 Block 拆分**：当 `use_hybrid_blocks=True`（混合 Mamba+Attention 模型）时，物理 Block 可拆分为多个更小的 Kernel Block：
  ```python
  self.blocks_per_phys_block = self.physical_block_size // self.logical_block_size
  ```

- **Slot Mapping 计算**（第 120-186 行）：将每个 token 位置映射到 KV 缓存中的 slot：
  ```python
  block_numbers = self.block_table.np.ravel()[block_table_indices]
  block_offsets = positions % self.block_size
  slot_mapping = block_numbers * self.block_size + block_offsets
  ```

- **Context Parallelism 支持**（第 128-168 行）：启用 DCP（Decode Context Parallelism）或 PCP（Prefill Context Parallelism）时，slot mapping 考虑跨设备交错模式

### 6.2 MultiGroupBlockTable

**文件**: `vllm_ascend/worker/block_table.py:231-320`

支持具有多个 KV 缓存组的模型（如混合 Attention + Mamba），不同层可有不同 Block 大小。

### 6.3 Block 元数据管理（与 vllm 基线一致）

- **KVCacheBlock**：包含 `block_id`、`ref_cnt`（引用计数）、`_block_hash`（用于 Prefix Caching）和双向链表指针（用于空闲队列）
- **FreeKVCacheBlockQueue**：双向链表实现的 LRU 空闲 Block 队列
- **BlockPool**：中央 Block 池，提供分配、释放、缓存、淘汰功能
- **SingleTypeKVCacheManager**：管理单一注意力类型的 Block，支持 Full Attention、Sliding Window、Chunked Local、Mamba、Cross Attention、Sink Attention 等

---

## 七、跨层 KV 缓存共享

**文件**: `vllm_ascend/worker/model_runner_v1.py:2665-2668`

MTP（Multi-Token Prediction）等场景下，多个层共享同一份 KV 缓存张量：

```python
for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
    kv_caches[layer_name] = kv_caches[target_layer_name]
```

---

## 八、PD 分离部署的 KV 传输

### 8.1 KV 传输工具函数

**文件**: `vllm_ascend/attention/utils.py:264-292`

```python
def wait_for_kv_layer_from_connector(layer_name: str):
    connector = get_kv_transfer_group()
    connector.wait_for_layer_load(layer_name)

def maybe_save_kv_layer_to_connector(layer_name: str, kv_cache_layer: list[torch.Tensor]):
    connector = get_kv_transfer_group()
    connector.save_kv_layer(layer_name, kv_cache_layer, attn_metadata)
```

### 8.2 异步 KV 传输

**文件**: `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py`

`KVCacheStoreSendingThread` 和 `KVCacheStoreRecvingThread` 负责 Prefill 和 Decode 节点之间的 KV 缓存异步传输，支持：

- **Mooncake 后端**：基于 RDMA 的高性能传输
- **Memcache 后端**：昇腾原生方案

### 8.3 NPU Event 同步

**文件**: `vllm_ascend/attention/attention_v1.py:892-908`

写入 KV 缓存后记录 NPU Event，用于与 KV 传输同步：

```python
if self.is_kv_producer:
    attn_metadata.reshape_cache_event = torch.npu.Event()
    ...
    attn_metadata.reshape_cache_event.record()
```

### 8.4 CPU Offload（Prefix Caching 卸载）

**文件**: `vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_kv_cache_manager.py`

`CPUKVCacheManager` 管理 CPU 侧的 KV 缓存 Block，用于 Prefix Caching，采用 LRU 淘汰策略。

---

## 九、端到端数据流

```
请求到达
    │
    ▼
KVCacheManager.get_computed_blocks()
    │  在 BlockPool.cached_block_hash_to_block 中查找 Block 哈希
    │  返回 Prefix Cache 命中的 Block（或空）
    ▼
KVCacheManager.allocate_slots()
    │  1. remove_skipped_blocks() — 释放注意力窗口外的 Block
    │  2. 检查空闲 Block 数量
    │  3. allocate_new_computed_blocks() — 命中时追加已有 Block
    │  4. allocate_new_blocks() — 从 BlockPool 获取新 Block
    │  5. cache_blocks() — 完整 Block 哈希缓存供后续共享
    ▼
Block IDs 返回调度器
    │
    ▼
Worker: BlockTable.append_row(block_ids)
    │  写入 block_table[req_idx]
    │  compute_slot_mapping() — 每个 token 位置 → slot 索引
    ▼
commit_block_table() + commit_slot_mapping()
    │  CPU→GPU 异步拷贝
    ▼
GPU 前向推理:
    │  reshape_and_cache() — 通过 slot_mapping 写入 K/V 到 kv_cache
    │  attention() — 通过 block_table 读取 K/V 执行注意力
    ▼
请求完成:
    KVCacheManager.free() → BlockPool.free_blocks()
    Block 归还空闲队列（LRU），已缓存 Block 保留于哈希表直至被淘汰
```

---

## 十、vllm-ascend vs vllm 基线关键差异汇总

| 维度 | vllm (GPU) | vllm-ascend (NPU) |
|------|-----------|-------------------|
| Block Size | 多种（16/32 等） | 固定 128 |
| K/V 存储 | 合并张量 dim0=2 | 分离为两个独立张量 |
| Cache 写入算子 | `reshape_and_cache_flash` | `_npu_reshape_and_cache` / `npu_scatter_pa_kv_cache` |
| MLA 融合 | 无 | `npu_kv_rmsnorm_rope_cache`（Norm+RoPE+写缓存融合） |
| SFA/Indexer | 不支持 | 额外 dsa_k_cache + 量化 scale cache |
| PD 分离 | 通用实现 | 2MB 对齐 + Mooncake/Memcache 专用后端 |
| 硬件适配 | 统一 | A2/A5 不同算子路径 |
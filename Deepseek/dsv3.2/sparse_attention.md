# DeepSeek V3.2 Sparse Attention 源码详解

> 本文档基于 `/vllm-workspace/vllm` 仓库（origin/main @ 3da29aa4a），深入分析 DeepSeek V3.2 引入的 **Lightning Indexer / Sparse Attention** 机制，结合源码逐函数说明其数学原理、tensor 流转和 kernel 调用。

---

## 一、整体架构概览

DeepSeek V3.2 在 MLA 之上新增了一个**旁路索引器** (Lightning Indexer)，对每个 query 预测历史 token 的重要性分数，选 top-k 个 token 作为 MLA 实际 attend 的候选。

```
                     ┌──────────────────────────┐
                     │   Lightning Indexer      │
   x (T, H) ────────►│   W_q · q + W_k · k      │  head_dim=128, n_head=64
                     │   + per-head weight w    │  K: MQA 风格共享
                     │   → top-k indices        │  topk=2048 典型
                     └────────────┬─────────────┘
                                  │ topk_indices (T, 2048) int32
                                  ▼
                     ┌──────────────────────────┐
                     │   MLA Sparse Attention   │  只对 topk 指向的 K/V 算标准 attn
                     │   Q · K[topk] · V         │  其余 token 跳过
                     └──────────────────────────┘
```

**关键文件路径**（相对 `/vllm-workspace/vllm`）：

| 组件 | 文件 | 关键行 |
|---|---|---|
| Indexer 模块 | `vllm/model_executor/models/deepseek_v2.py` | 603-743 |
| Indexer Op（topk 计算） | `vllm/model_executor/layers/sparse_attn_indexer.py` | 81-373 |
| MLA 入口 | `vllm/model_executor/models/deepseek_v2.py` | 878-1081 |
| Indexer Backend | `vllm/v1/attention/backends/mla/indexer.py` | 118-774 |
| MLA Sparse Backend | `vllm/v1/attention/backends/mla/flashmla_sparse.py` | 60-1149 |
| MQA Logits Kernel | `vllm/utils/deep_gemm.py` | 342-384 |

---

## 二、Lightning Indexer 模块

### 2.1 类签名与构造（`deepseek_v2.py:603-674`）

```python
class Indexer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        prefix: str = "",
        is_inplace_rope: bool = False,
    ):
```

**关键参数**：
- `q_lora_rank=1536`：Q 来自 fused_qkv_a_proj 的低秩中间表示
- `hidden_size=H`：K 和 weights 来自 `hidden_states`（即 layer norm 之后的输入）
- `topk_indices_buffer`：模型级**预分配**的 `(max_num_batched_tokens, topk_tokens) int32` buffer，所有 layer 共享，避免每次重新分配

**子模块**：

| 名称 | 类型 | 输入 → 输出 | 说明 |
|---|---|---|---|
| `wq_b` | `ReplicatedLinear(1536, 128*64)` | qr → (T, 64, 128) | Q 投影，disable TP（所有 rank 都算全量） |
| `wk_weights_proj` | `MergedColumnParallelLinear(H, [128, 64], disable_tp=True)` | x → (T, 128+64) | **一次 GEMM** 算 K 和 weights（V3.2 优化点） |
| `k_norm` | `LayerNorm(128, eps=1e-6)` | (T, 128) → (T, 128) | K 端 RMS norm |
| `softmax_scale` | `head_dim**-0.5 = 128**-0.5` | 标量 | 折进 weights |
| `k_cache` | `DeepseekV32IndexerCache` | K 持久化 | 每 token 132B (128 fp8 + 4 fp32 scale) |
| `indexer_op` | `SparseAttnIndexer` | 核心 topk 计算 | 见第三节 |

### 2.2 完整 forward 流程（`deepseek_v2.py:678-743`）

```python
def forward(
    self, hidden_states: torch.Tensor, qr: torch.Tensor,
    positions, rotary_emb
) -> torch.Tensor:
    # ─── 1. Q 投影 ──────────────────────────────────────────
    q, _ = self.wq_b(qr)                          # (T, 8192) = (T, 64*128)
    q = q.view(-1, self.n_head, self.head_dim)    # (T, 64, 128)

    # ─── 2. K + weights 一次 GEMM（wk_weights_proj） ───────
    kw, _ = self.wk_weights_proj(hidden_states)  # (T, 128+64)
    k = kw[:, :self.head_dim]                     # (T, 128) 全量 K，BF16
    weights = kw[:, self.head_dim:]               # (T, 64)  per-head 标量权重

    # ─── 3. K-norm + RoPE ──────────────────────────────────
    k = self.k_norm(k)                            # (T, 128) BF16

    q_pe, q_nope = torch.split(q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
    # rope_dim=64, nope_dim=64

    k_pe, k_nope = torch.split(k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1)

    q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
    # q_pe: (T, 64, 64) → GPT-J RoPE 后形状不变
    # k_pe: (T, 1, 64)  MQA 风格

    q = torch.cat([q_pe, q_nope], dim=-1)         # (T, 64, 128)
    k = torch.cat([k_pe.squeeze(-2), k_nope], dim=-1)  # (T, 128)

    # ─── 4. Q 量化（K 量化 fuse 在 cache insert 中）────────
    q = q.view(-1, self.head_dim)                  # (T*64, 128) flat per-head
    q_fp8, q_scale = per_token_group_quant_fp8(
        q, quant_block_size=128, use_ue8m0=True,
    )
    q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)  # (T, 64, 128) fp8_e4m3
    q_scale = q_scale.view(-1, self.n_head, 1)           # (T, 64, 1) fp32

    # ─── 5. 把 weights / q_scale / softmax_scale 折成一个张量 ──
    weights = (
        weights.unsqueeze(-1)                      # (T, 64, 1)
        * q_scale                                  # (T, 64, 1)
        * self.softmax_scale                       # 1/sqrt(128)
        * self.n_head**-0.5                        # 1/sqrt(64)
    ).squeeze(-1)                                   # (T, 64) fp32

    # ─── 6. 委托给 SparseAttnIndexer 做 cache 写入 + topk ─
    return self.indexer_op(hidden_states, q_fp8, k, weights)
```

**关键设计**：
1. **第 5 步的折叠**：把 `q_scale` 折进 `weights` 后，**下游 `mqa_logits` kernel 不再需要单独处理 Q 端 scale**（FP8 路径），只需把 Q 当成"已补偿的 fp8"与 K 算 logits。这是 V3.2 性能优化的关键。
2. **MQA 风格 K**：所有 64 个 head 共享同一份 K（`k_pe` 形状 `(T, 1, 64)`，K cache 也只存一份），节省 64 倍 K cache 容量。
3. **逐 token 写 K cache**：每 token 现算 K → 写 132B，**没有 Compressor**（V4 才引入）。

### 2.3 V3 是否启用 Indexer

由 `is_v32` 标志决定（`deepseek_v2.py:999`）：

```python
self.is_v32 = hasattr(config, "index_topk")   # config 含 index_topk 字段即 V3.2

if self.is_v32:
    self.indexer_rope_emb = get_rope(...)     # indexer 自己的 RoPE
    self.indexer = Indexer(...)
    # IndexCache 优化：每隔 N 层才跑一次 indexer
    if getattr(config, "use_index_cache", False):
        _index_topk_freq = getattr(config, "index_topk_freq", 1)
        _index_topk_pattern = getattr(config, "index_topk_pattern", None)
        layer_id = extract_layer_index(prefix)
        if _index_topk_pattern is None:
            _skip_topk = max(layer_id - 1, 0) % _index_topk_freq != 0
        elif 0 <= layer_id < len(_index_topk_pattern):
            _skip_topk = _index_topk_pattern[layer_id] == "S"
else:
    self.indexer = None
```

V3.2-Exp 引入 `use_index_cache`（论文 arxiv:2603.12201）允许**每隔 N 层复用上一层 indexer 的 topk 结果**，减少 indexer 计算量。`_skip_topk=True` 时 `MultiHeadLatentAttentionWrapper` 复用前一层 indexer 的 indices。

---

## 三、SparseAttnIndexer：topk 计算核心

### 3.1 入口类（`sparse_attn_indexer.py:406-537`）

```python
@CustomOp.register("sparse_attn_indexer")
class SparseAttnIndexer(CustomOp):
    def __init__(
        self, k_cache, quant_block_size, scale_fmt, topk_tokens,
        head_dim, max_model_len, max_total_seq_len, topk_indices_buffer,
        skip_k_cache_insert=False, use_fp4_cache=False,
    ):
```

`CustomOp` 机制允许 `forward_native` / `forward_cuda` / `forward_hip` / `forward_xpu` 分发。`forward_cuda` 直接调用 `torch.ops.vllm.sparse_attn_indexer` 自定义 op。

### 3.2 自定义 op 函数（`sparse_attn_indexer.py:81-373`）

#### 3.2.1 K 端：写 cache

```python
# Line 163-173
if not skip_k_cache_insert:
    ops.indexer_k_quant_and_cache(
        k,                              # (T, 128) BF16
        kv_cache,                       # (num_blocks, 64, 132) uint8
        slot_mapping,                   # (T,) int64, 每个 token 的 cache slot
        quant_block_size=128,
        scale_fmt="ue8m0",
    )
```

`indexer_k_quant_and_cache` 是 C++/Triton kernel：每 128 元素一组做 UE8M0 FP8 量化，写入 `(num_blocks, 64, 132)` cache：
- 前 128B：fp8 NoPE
- 后 4B：float32 scale（每 128 elem 一字节，4 字节 packed）

#### 3.2.2 Prefill 路径

```python
# Line 176-256
if has_prefill:
    prefill_metadata = attn_metadata_narrowed.prefill
    # 1) 把每个 chunk 的 K 从 cache gather 到连续 workspace
    k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
        values_spec, scales_spec,  # FP8: (total_seq_lens, 128) + (total_seq_lens, 4)
    )
    for chunk in prefill_metadata.chunks:
        k_quant = k_quant_full[:chunk.total_seq_lens]
        k_scale = k_scale_full[:chunk.total_seq_lens]
        if not chunk.skip_kv_gather:
            ops.cp_gather_indexer_k_quant_cache(
                kv_cache, k_quant, k_scale,
                chunk.block_table, chunk.cu_seq_lens,
            )
        # 2) 算 MQA logits
        q_slice = q_quant[chunk.token_start:chunk.token_end]
        logits = fp8_fp4_mqa_logits(
            (q_slice, q_scale),              # FP8 路径 q_scale=None（已折进 weights）
            (k_quant, k_scale),
            weights[chunk.token_start:chunk.token_end],
            chunk.cu_seqlen_ks, chunk.cu_seqlen_ke,   # 见 §3.3
            clean_logits=False,
        )
        # 3) topk 选择
        topk_indices = topk_indices_buffer[
            chunk.token_start:chunk.token_end, :topk_tokens
        ]
        ops.top_k_per_row_prefill(
            logits, chunk.cu_seqlen_ks, chunk.cu_seqlen_ke,
            topk_indices, num_rows, ...,
        )
```

#### 3.2.3 Decode 路径

```python
# Line 258-371
if has_decode:
    decode_metadata = attn_metadata_narrowed.decode
    kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)  # 4D view

    # 1) Pad Q（处理 spec-decode 时的 padding）
    if decode_metadata.requires_padding:
        padded_q_quant = pack_seq_triton(q_quant[:num_decode_tokens], decode_lens, ...)
        # 把 spec-decode 的多 token decode 展开成每个 token 独立算
    else:
        padded_q_quant = q_quant[:num_decode_tokens].reshape(B, next_n, ...)

    # 2) Paged MQA logits
    logits = fp8_fp4_paged_mqa_logits(
        (padded_q_quant, padded_q_scale),
        kv_cache,                                    # paged 视图
        weights[:num_padded_tokens],
        seq_lens,                                    # (B,) 或 (B, next_n)
        decode_metadata.block_table,
        decode_metadata.schedule_metadata,
        max_model_len=max_model_len,
    )
    # logits: (num_padded_tokens, max_seq_len) fp32

    # 3) Persistent Top-K（Hopper 优化）
    if topk_tokens in (512, 1024, 2048):
        torch.ops._C.persistent_topk(
            logits, seq_lens, topk_indices,
            topk_workspace, topk_tokens,
            attn_metadata_narrowed.max_seq_len,
        )
    else:
        ops.top_k_per_row_decode(logits, next_n, seq_lens, topk_indices, ...)

    # 4) Unpad（spec-decode 路径）
    if decode_metadata.requires_padding:
        topk_indices = unpack_seq_triton(topk_indices, decode_lens)
        topk_indices_buffer[:topk_indices.shape[0], :topk_indices.shape[1]] = topk_indices

return topk_indices_buffer
```

### 3.3 cu_seqlen_ks / cu_seqlen_ke：per-query 上下文范围

这两个张量定义每个 query 要 scan 的 K 范围：

- `cu_seqlen_ks[i]`：query i 可访问的 K 起点（通常 = 0）
- `cu_seqlen_ke[i]`：query i 可访问的 K 终点（通常 = L_i，含 self）

`mqa_logits` kernel 按 `cu_seqlen_ke[i] - cu_seqlen_ks[i]` 决定每行 logits 的有效长度，未填充位置置 `-inf`（`clean_logits=True` 时）。

### 3.4 MQA Logits Kernel（`vllm/utils/deep_gemm.py:342-384`）

```python
def fp8_fp4_mqa_logits(
    q: tuple[torch.Tensor, torch.Tensor | None],   # (q_values, q_scale)
    kv: tuple[torch.Tensor, torch.Tensor],         # (k_packed, k_scales)
    weights: torch.Tensor,                         # (M, H) fp32
    cu_seqlen_ks: torch.Tensor,                    # (M,) int32
    cu_seqlen_ke: torch.Tensor,                    # (M,) int32
    clean_logits: bool,
) -> torch.Tensor:                                 # (M, N) fp32 logits
```

底层是 **DeepGEMM 的 fused MQA logits kernel**（Hopper / Blackwell 各有实现）：

```
对每个 query q_i (head_dim=128, n_head=64):
  对每个历史 token j ∈ [cu_seqlen_ks[i], cu_seqlen_ke[i]):
    logits[i, j] = sum_{h=0..63} weights[i, h] * <q_fp8[i, h, :], k_fp8[j, :]>
```

**单 kernel 算出所有 head 的 logits 总和**，等价于：

```
s(i, j) = sum_h w_h * <q_h, k>  =  ⟨q_stacked, k_replicated · w⟩  (数学上等价变形)
```

**为什么这样设计**：kernel 一次处理一整行 (MQA 风格 K × 多 head Q)，最大化 Tensor Core 利用率（128×1×128 矩阵乘）。

### 3.5 Top-K Kernels

| Kernel | 路径 | 何时用 |
|---|---|---|
| `top_k_per_row_prefill` | C++ / Triton | Prefill，每行独立选 topk |
| `top_k_per_row_decode` | C++ / Triton | Decode，next_n=1 |
| `persistent_topk` | CUDA C++ | Decode，next_n=1 且 topk ∈ {512, 1024, 2048}（Hopper 专用优化） |

**`persistent_topk`** 利用 SM 持续驻留特性：把 SM 分配给固定的 row 范围，每个 SM 持续处理直到完成，避免 SM 调度开销。`RADIX_TOPK_WORKSPACE_SIZE = 1MB` 是 radix-based selection 的临时 buffer。

---

## 四、Indexer Metadata Builder

### 4.1 元数据结构（`indexer.py:200-216`）

```python
@dataclass
class DeepseekV32IndexerMetadata:
    seq_lens: torch.Tensor        # (B,) int32, 每个 request 的总长
    max_seq_len: int              # batch 内最大长度（persistent_topk 用）
    slot_mapping: torch.Tensor    # (T,) int64, token → cache slot

    num_decodes: int
    num_decode_tokens: int        # = sum(decode_lens)，spec-decode 时 > B
    num_prefills: int
    num_prefill_tokens: int

    decode: DeepSeekV32IndexerDecodeMetadata | None
    prefill: DeepseekV32IndexerPrefillMetadata | None
```

### 4.2 Prefill Chunk Metadata（`indexer.py:168-184`）

```python
@dataclass
class DeepseekV32IndexerPrefillChunkMetadata:
    block_table: torch.Tensor           # (num_reqs, max_blocks) int32
    cu_seqlen_ks: torch.Tensor          # (num_tokens,) int32
    cu_seqlen_ke: torch.Tensor          # (num_tokens,) int32
    cu_seq_lens: torch.Tensor           # (num_reqs+1,) int32
    token_to_seq: torch.Tensor
    total_seq_lens: int
    token_start: int                    # 在 flat q 中的起始偏移
    token_end: int
    num_reqs: int
    skip_kv_gather: bool = False        # chunk 不需要 K gather（K 已在 workspace）
```

### 4.3 Decode Metadata（`indexer.py:187-197`）

```python
@dataclass
class DeepSeekV32IndexerDecodeMetadata:
    block_table: torch.Tensor
    seq_lens: torch.Tensor              # 1D (B,) 或 2D (B, next_n) for spec
    decode_lens: torch.Tensor           # 每个 decode request 包含几个 token
    requires_padding: bool              # 是否需要 pack/unpack
    schedule_metadata: torch.Tensor     # DeepGEMM 调度元数据
```

### 4.4 Chunk 划分（`indexer.py:69-114`）

`split_indexer_prefill_chunks` 根据两个约束划分 chunk：

1. **Workspace 约束**：`Σ total_seq_lens ≤ workspace_size`（约 825MB）
2. **Logits 约束**：`M × N × 4 ≤ max_logits_bytes`（防止单 chunk logits 太大）

当单 request 超出 logits 预算时，按 query 维度 sub-chunk。

```python
max_logits_elems = max_logits_bytes // 4
while end < n:
    q, s = query_lens_cpu[end].item(), seq_lens_cpu[end].item()
    new_m, new_n = chunk_m + q, chunk_n + s
    if new_n <= workspace_size and new_m * new_n <= max_logits_elems:
        chunk_m, chunk_n = new_m, new_n
        end += 1
```

### 4.5 Decode 准备（`indexer.py:347-414`）

`requires_padding` 为 True 时表示有 spec-decode，需要把 multi-token decode 展成单 token：

```python
# 假设 4 个 request, decode_lens = [3, 1, 4, 0]
# 实际 num_decode_tokens = 3 + 1 + 4 + 0 = 8
# 每个 token 的有效 seq_len = (L_b - decode_lens_b) + j + 1
# → 展开后 batch_size=8, 每个 next_n=1
```

`use_native=True`（SM100，next_n ∈ {1,2}）时不做展平，`seq_lens` 保留 2D 形状 `(B, next_n)`，直接喂给 paged_mqa_logits。

---

## 五、MLA Sparse Attention Backbone

### 5.1 入口（`deepseek_v2.py:999-1081`）

V3.2 触发条件：`is_v32 = hasattr(config, "index_topk")`。

```python
# 关键：把 indexer 和 indexer_rotary_emb 传给 MultiHeadLatentAttentionWrapper
mla_modules = MLAModules(
    kv_a_layernorm=self.kv_a_layernorm,
    kv_b_proj=self.kv_b_proj,
    rotary_emb=self.rotary_emb,
    o_proj=self.o_proj,
    fused_qkv_a_proj=self.fused_qkv_a_proj if self.q_lora_rank is not None else None,
    kv_a_proj_with_mqa=self.kv_a_proj_with_mqa if self.q_lora_rank is None else None,
    q_a_layernorm=self.q_a_layernorm if self.q_lora_rank is not None else None,
    q_b_proj=self.q_b_proj if self.q_lora_rank is not None else None,
    q_proj=self.q_proj if self.q_lora_rank is None else None,
    indexer=self.indexer,
    indexer_rotary_emb=self.indexer_rope_emb,
    is_sparse=self.is_v32,            # ★ 关键：让 wrapper 走 sparse 路径
    topk_indices_buffer=topk_indices_buffer,
)

self.mla_attn = MultiHeadLatentAttentionWrapper(
    self.hidden_size, self.num_local_heads, self.scaling,
    self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim,
    self.q_lora_rank, self.kv_lora_rank, mla_modules, ...
    skip_topk=_skip_topk,             # IndexCache 优化
)
```

### 5.2 KV Cache 布局（`flashmla_sparse.py:68-90`）

**V3.2 FP8 layout**（`fp8_ds_mla`，**每 token 656B**）：

| 区段 | 字节 | dtype | 内容 |
|---|---|---|---|
| NoPE 量化 | 512 | fp8_e4m3 | 128 elem × 4（512=4×128） |
| Scale | 16 | float32 | 4 个 scale（每 128 fp8 一字节） |
| RoPE | 128 | bfloat16 | 64 elem × 2（不量化） |
| **合计** | **656** | | |

详细来源：
```python
# flashmla_sparse.py:140-150
if cache_dtype_str == "fp8_ds_mla":
    return (num_blocks, block_size, 656)  # V3.2 main MLA
```

### 5.3 MLA Forward：选 topk 后 sparse 算

`MultiHeadLatentAttentionWrapper` 内部逻辑（`vllm/v1/attention/backends/mla/utils.py`，未在本仓库文件直接列出，参考 `flashmla_sparse.py:113-114` `SparseMLAAttentionImpl`）：

1. **prefill 阶段**：用 `flash_mla_sparse_fwd` kernel，输入 Q、KV workspace、indices
2. **decode 阶段**：用 `flash_mla_with_kvcache` kernel，输入 Q、paged K/V cache、indices

两个 kernel 接受 `indices: (B, topk) int32`，**对每个 query 只读取 indices 指向的 K/V block**，其余跳过。

### 5.4 Prefill Workspace（`flashmla_sparse.py:234-241`）

```python
def get_prefill_workspace_size(max_model_len: int):
    # 5 * max_model_len * 576 * 2 bytes
    # DeepSeek-V3.2 (max_model_len=163840) → ~900MB
    return max_model_len * 5
```

Pre-allocate workspace 把 paged K cache gather 出来给 `flash_mla_sparse_fwd` 用，避免 kernel 内 gather 开销。

### 5.5 Decode Mixed Batch

`FlashMLASparseImpl._build_fp8_mixed_decode_prefill`（`flashmla_sparse.py:390-402`）支持 decode+prefill 混合：
- prefill 部分用 BF16 kernel（padded heads = 128）
- decode 部分用 FP8 kernel（padded heads = 64/128）

触发条件（`flashmla_sparse.py:60-66`）：
```python
MIN_HEADS_FOR_BF16_PREFILL = 32
# 当 per-rank head count < 32 时用 mixed batch
```

---

## 六、完整数据流（一次 forward）

```
输入: hidden_states (T, H) bf16, qr (T, 1536) bf16 (来自 fused_qkv_a_proj)
      positions (T,) int64, rotary_emb

  ① Indexer forward
     ├─ wq_b(qr)            : (T, 1536) → (T, 64*128) = (T, 8192)
     ├─ wk_weights_proj(x)  : (T, H)    → (T, 128+64)
     ├─ k_norm(k)           : (T, 128)  → (T, 128)
     ├─ rotary_emb(q_pe, k_pe)
     ├─ cat([q_pe, q_nope]) : (T, 64, 128)
     ├─ per_token_group_quant_fp8(q, 128) → q_fp8 (T, 64, 128), q_scale (T, 64, 1)
     ├─ weights 折 q_scale、softmax_scale、1/sqrt(n_head)
     └─ SparseAttnIndexer.forward
        ├─ indexer_k_quant_and_cache(k, slot_mapping) → 写 K cache 132B/token
        ├─ (prefill) cp_gather_indexer_k_quant_cache → 连续 K workspace
        ├─ (prefill) fp8_fp4_mqa_logits → logits (T, max_chunk_seq_len) fp32
        ├─ (decode)  fp8_fp4_paged_mqa_logits → logits (num_decode, max_seq_len) fp32
        ├─ top_k_per_row_prefill / persistent_topk / top_k_per_row_decode
        └─ 写 topk_indices_buffer (T, topk) int32

  ② MLA forward
     ├─ kv_b_proj(kv_lora)  : (T, 512) → (T, n_lh * (nope + v)) = (T, n_lh * 640)
     ├─ MLA absorb (Q × W_UK, KV × W_UK) → Q' (T, n_lh, nope), KV' (T, nope)
     ├─ RoPE (Q, K) + 写 SWA KV cache 656B/token
     ├─ flash_mla_sparse_fwd (prefill) / flash_mla_with_kvcache (decode)
     │      输入: Q, KV, topk_indices
     │      行为: 对每个 query i，只读 topk_indices[i, :] 指向的 K/V block
     └─ o_proj: (T, n_lh * v) → (T, H)

输出: hidden_states (T, H) bf16
```

---

## 七、性能特征

### 7.1 计算量对比（prefill L=128K, topk=2048, n_head=64）

| 操作 | 朴素 attention | V3.2 sparse |
|---|---|---|
| Indexer Q 投影 | 0 | T·H·n_head·head_dim = T·H·8192 GEMM |
| Indexer K 投影 | 0 | T·H·(head_dim+n_head) = T·H·192 GEMM |
| MQA Logits（全量 scan） | 0 | T·L·head_dim = T·L·128 dot products |
| Top-K 选择 | 0 | T·L·log(topk) 排序 |
| MLA QK^T | T·L·head_dim = T·L·576 | T·k·head_dim = T·k·576 |
| MLA Softmax + V | T·L·v_head | T·k·v_head |
| **KV 访问** | **T·L** | **T·k** + T·L（indexer scan） |

**L=128K, k=2048, T=128** 时：
- MLA 部分加速比 ≈ 128K / 2K = **64x**
- Indexer 额外开销 ≈ T·L·128 = 128·128K·128 = 2.1 GFLOPs（相对 MLA 128K 维 14 TFLOPs 可忽略）
- 总加速（相对朴素）≈ **2-3x**（perfill）；decode **8-10x**

### 7.2 内存对比（decode 阶段, B=64, L=128K, topk=2048）

| | 朴素 | V3.2 sparse |
|---|---|---|
| KV cache 读（per token） | 128K × 656B = 80MB | 2K × 656B = 1.3MB |
| Indexer K cache 读 | 0 | 128K × 132B = 16MB（paged, 但 MQA 共享）|
| Indexer 额外写入 | 0 | 132B/token = 8.5KB |

KV 访问带宽减少 **64x**，这是 decode 加速的主要来源。

### 7.3 IndexCache 优化

`use_index_cache=True` + `index_topk_freq=N`：每隔 N 层复用上一层 indexer 的 topk 结果：
- 减少 indexer 计算量 N 倍
- 准确度损失极小（相邻层的 attention 模式相近）

---

## 八、Kernel 选型与平台支持

| Kernel | 平台 | 数据类型 | 何时用 |
|---|---|---|---|
| `fp8_fp4_mqa_logits` | CUDA (DeepGEMM) | FP8 / FP4 | V3.2 / V4 通用 |
| `xpu_fp8_mqa_logits` | Intel XPU | FP8 | XPU 后备 |
| `rocm_aiter_sparse_attn_indexer` | AMD ROCm (AITER) | FP8 | AMD GPU |
| `flash_mla_sparse_fwd` | CUDA (FlashMLA) | bf16 → fp8 cache | Prefill |
| `flash_mla_with_kvcache` | CUDA (FlashMLA) | fp8 | Decode |
| `trtllm_batch_decode_sparse_mla_dsv4` | CUDA (FlashInfer) | fp8 | V4 mixed batch |

**FP4 Indexer cache**（V3.2+ 支持，SM100 only）：
- 每 token 68B（64 packed nibbles + 4 UE8M0 scales），比 FP8 节省 48%
- 需 `attention_config.use_fp4_indexer_cache=True`
- 限制：`next_n ∈ {1, 2}`（其他值降级到 flatten 路径）

---

## 九、与 V4 的关键差异（预告）

V4 在 V3.2 之上引入 **Compressor**：
- 主 K cache 走压缩路径（c_r=4/128）
- Indexer 也配独立 Compressor（`compress_ratio=4`）
- Indexer 的 K 来自 `indexer.compressor.fused_wkv_wgate`（aux stream），而非 `wk_weights_proj` 现算
- `skip_k_cache_insert=True`（K 由 compressor 提前写完）

详细对比见 `Deepseek/dsv4/` 文档（待整理）。

---

## 十、参考

- DeepSeek-V3.2 技术报告：[DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with Sparse Attention](https://api-docs.deepseek.com/news/news251201)
- Lightning Indexer 论文：arxiv:2603.12201 (IndexCache)
- vLLM 源码：`/vllm-workspace/vllm/` (origin/main @ 3da29aa4a)

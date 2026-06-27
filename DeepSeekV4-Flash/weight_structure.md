# DeepSeek-V4-Flash 权重结构详解

> 基于 `/home/weights/DeepSeek-V4-Flash-w8a8-mtp` 的静态分析。
>
> **注意**：本目录的 safetensors 文件**未下载完整**（`._____temp/` 下只有 8/70 个分片，且分片本身也是残缺的），无法直接 `safe_open` 读 header。所以本分析以 `config.json` + `quant_model_description.json` 为准，shape 由 config 推导，量化类型（`W8A8_DYNAMIC` / `FLOAT`）从 description 实读。

---

## 一、模型概况

`config.json` 关键字段：

| 字段 | 值 | 说明 |
|---|---|---|
| `architectures` | `DeepseekV4ForCausalLM` | DeepSeek-V4 系列 |
| `model_type` | `deepseek_v4` | HF 标识 |
| `hidden_size` | 4096 | 隐藏维度 H |
| `num_hidden_layers` | 43 | 主干层数 L |
| `num_nextn_predict_layers` | 1 | MTP（Multi-Token Prediction）额外层数 |
| `num_attention_heads` | 64 | Q 头数 Hq |
| `num_key_value_heads` | 1 | KV 头数 Hk（MLA 风格，只 1 个共享 KV 头） |
| `head_dim` | 512 | 单头维度 D（latent 维度） |
| `qk_rope_head_dim` | 64 | RoPE 维度 Drope |
| `q_lora_rank` | 1024 | Q 压缩秩 |
| `o_lora_rank` | 1024 | O 压缩秩 |
| `o_groups` | 8 | O 端分组数 |
| `n_routed_experts` | 256 | routed expert 总数 E |
| `n_shared_experts` | 1 | shared expert 数（本模型 = 1） |
| `num_experts_per_tok` | 6 | 每 token 激活数 K |
| `moe_intermediate_size` | 2048 | 单 expert 中间维度 I |
| `norm_topk_prob` | true | router 权重归一化 |
| `scoring_func` | `sqrtsoftplus` | 路由打分函数 |
| `topk_method` | `noaux_tc` | topk 选择方法 |
| `routed_scaling_factor` | 1.5 | 路由权重缩放 |
| `swiglu_limit` | 10.0 | SwiGLU clamp 上限 |
| `vocab_size` | 129280 | 词表 V |
| `tie_word_embeddings` | false | lm_head 独立 |
| `torch_dtype` | bfloat16 | 原始 dtype（量化后变 int8） |
| `max_position_embeddings` | 1048576 | 1M 上下文 |
| `rope_theta` | 10000 | RoPE 基频 |
| `rope_scaling` | `{type: yarn, factor: 16, original: 65536, β_fast=32, β_slow=1}` | YaRN 扩展到 1M |
| `sliding_window` | 128 | 滑动窗口大小（部分层使用） |
| `num_hash_layers` | 3 | 哈希路由层数（HC = Hash Competition） |
| `index_n_heads` | 64 | 闪电索引器头数 |
| `index_head_dim` | 128 | 闪电索引器头维度 |
| `index_topk` | 512 | 闪电索引器 topk |
| `rms_norm_eps` | 1e-6 | RMSNorm epsilon |
| `use_cache` | true | 推理开 KV cache |

> **"V4-Flash" 含义**：相比完整 V4，Flash 版把**所有线性层权重量化到 W8A8**（int8 weight + dynamic per-token activation quant），大幅压缩显存；保留 MTP（Multi-Token Prediction）以提升生成速度。

---

## 二、量化方案概览

从 `quant_model_description.json` 实读：

| 量化类型 | 数量 | 说明 |
|---|---|---|
| `W8A8_DYNAMIC` | 102 232 | int8 权重 + 动态 per-token activation 量化（绝大多数 expert / attn 线性层） |
| `FLOAT` | 945 | 保持 bf16 精度（norm γ、attn_sink、HC 张量、indexer 等） |
| `<dict>` | 3 | 元信息（quarot 旋转矩阵等） |
| 其他 | 1 | 模型版本号等元数据 |
| **总计** | **103 181** | tensor 描述项 |

W8A8 量化下，每个被量化的 weight 配套 2 个 FLOAT 张量（`weight_scale` + `weight_offset`），所以 `W8A8_DYNAMIC` 的"实际"int8 权重约 34 077 个（102 232 ÷ 3）。

> 完整文件应该是 70 个 safetensors 分片，本目录只下载了 8 个（且分片本身残缺），所以本节列的 shape 是从 config 推导的，**不是从 safetensors header 实读**。

---

## 三、张量清单（按 N 归并的模板）

### 3.1 Top-level（3 个）

| 路径 | Shape | dtype | 说明 |
|---|---|---|---|
| `embed.weight` | (V, H) = (129280, 4096) | bf16 | token embedding（命名跟 V3 不同！） |
| `norm.weight` | (H,) = (4096,) | bf16 | 末层 RMSNorm |
| `head.weight` | (V, H) = (129280, 4096) | bf16 | lm_head（命名跟 V3 不同！） |

> **注意**：V4 顶层用了 `embed.weight` / `head.weight` / `norm.weight`（**省略 `model.` 前缀**），跟 V3 的 `model.embed_tokens.weight` / `lm_head.weight` / `model.norm.weight` 不一样。

### 3.2 主干每层（×43）

| 模板 | Shape | dtype | 数量 | 说明 |
|---|---|---|---|---|
| `layers.N.attn.attn_sink` | (Hq,) = (64,) | FLOAT | 43 | attention sink（streaming LLM 风格） |
| `layers.N.attn.wq_a.weight` | (qr, H) = (1024, 4096) | W8A8 | 43 | Q 压缩：A → qr |
| `layers.N.attn.wq_b.weight` | (Hq·(D+Drope), qr) = (36864, 1024) | W8A8 | 43 | Q 恢复：qr → Hq·(D+Drope) |
| `layers.N.attn.q_norm.weight` | (D,) = (512,) | FLOAT | 43 | Q latent RMSNorm |
| `layers.N.attn.wkv.weight` | (Hk·(D+Drope), H) = (576, 4096) | W8A8 | 43 | KV 压缩：H → Hk·(D+Drope) |
| `layers.N.attn.kv_norm.weight` | (D,) = (512,) | FLOAT | 43 | KV latent RMSNorm |
| `layers.N.attn.wo_a.weight` | (or_, Hq·D) = (1024, 32768) | W8A8 | 43 | O 压缩：Hq·D → or_ |
| `layers.N.attn.wo_b.weight` | (H, or_·ogroups) = (4096, 8192) | W8A8 | 43 | O 恢复：or_·ogroups → H |
| `layers.N.attn_norm.weight` | (H,) = (4096,) | FLOAT | 43 | attn 前 RMSNorm |
| `layers.N.ffn_norm.weight` | (H,) = (4096,) | FLOAT | 43 | MoE 前 RMSNorm |
| `layers.N.ffn.gate.weight` | (E, H) = (256, 4096) | FLOAT | 43 | router |
| `layers.N.ffn.gate.bias` | (E,) = (256,) | FLOAT | 40 | 部分层带 bias |
| `layers.N.ffn.gate.tid2eid` | (?,) | FLOAT | 3 | token-id → expert-id 映射（HC 路由用） |
| `layers.N.ffn.experts.M.w1.weight` | (I, H) = (2048, 4096) | W8A8 | 11 008 | 256 专家 × SwiGLU gate |
| `layers.N.ffn.experts.M.w2.weight` | (H, I) = (4096, 2048) | W8A8 | 11 008 | 256 专家 × down |
| `layers.N.ffn.experts.M.w3.weight` | (I, H) = (2048, 4096) | W8A8 | 11 008 | 256 专家 × up |
| `layers.N.ffn.shared_experts.w1.weight` | (I, H) = (2048, 4096) | W8A8 | 43 | shared expert gate |
| `layers.N.ffn.shared_experts.w2.weight` | (H, I) = (4096, 2048) | W8A8 | 43 | shared expert down |
| `layers.N.ffn.shared_experts.w3.weight` | (I, H) = (2048, 4096) | W8A8 | 43 | shared expert up |
| `layers.N.hc_attn_fn` | (?) | FLOAT | 43 | HC 路由 attn 系数 |
| `layers.N.hc_ffn_fn` | (?) | FLOAT | 43 | HC 路由 ffn 系数 |
| `layers.N.hc_attn_base` | (?) | FLOAT | 43 | HC 路由 attn 基向量 |
| `layers.N.hc_ffn_base` | (?) | FLOAT | 43 | HC 路由 ffn 基向量 |
| `layers.N.hc_attn_scale` | (?) | FLOAT | 43 | HC 路由 attn 缩放 |
| `layers.N.hc_ffn_scale` | (?) | FLOAT | 43 | HC 路由 ffn 缩放 |

### 3.3 Compressor（×41 层）

| 模板 | Shape | 数量 | 说明 |
|---|---|---|---|
| `layers.N.attn.compressor.ape` | (?) | 41 | RoPE 位置编码（absolute position embedding） |
| `layers.N.attn.compressor.wkv.weight` | (?) | 41 | compressor 的 KV 投影 |
| `layers.N.attn.compressor.wgate.weight` | (?) | 41 | compressor 的 gate |
| `layers.N.attn.compressor.norm.weight` | (?) | 41 | compressor 内部 norm |

> Compressor 是 V4 的新机制：把滑动窗口外的历史 K/V 压缩到低秩表示，节省长上下文 KV cache。

### 3.4 Lightning Indexer（×21 层）

| 模板 | 数量 | 说明 |
|---|---|---|
| `layers.N.attn.indexer.wq_b.weight` | 21 | 索引器 Q 投影 |
| `layers.N.attn.indexer.weights_proj.weight` | 21 | 索引器权重投影 |
| `layers.N.attn.indexer.compressor.*` | 各 21 | 索引器自己的 compressor（4 个） |

> 索引器是 DeepSeek V3.2 风格的稀疏 attention 前置：先快速选 topk 个 token，再对它们做完整 attention。本模型 `index_topk=512`。

### 3.5 MTP 层（×1）

| 模板 | Shape | 说明 |
|---|---|---|
| `mtp.N.enorm.weight` | (H,) | embed-side RMSNorm |
| `mtp.N.hnorm.weight` | (H,) | hidden-side RMSNorm |
| `mtp.N.e_proj.weight` | (H, H) = (4096, 4096) | embed→hidden 投影 |
| `mtp.N.h_proj.weight` | (H, H) = (4096, 4096) | hidden→hidden 投影 |
| `mtp.N.norm.weight` | (H,) | MTP 内 norm |
| `mtp.N.ffn_norm.weight` | (H,) | MTP FFN 前 norm |
| `mtp.N.head.weight` | (V, H) | MTP 自己的 head |
| `mtp.N.emb.tok_emb.weight` | (V, H) | MTP 自己的 embed |
| `mtp.N.attn.*` | 各种 | MTP 自己的 attn（结构跟主干层相同） |
| `mtp.N.ffn.gate.*` | 各种 | MTP 自己的 router |
| `mtp.N.ffn.experts.M.w1/w2/w3.weight` | 256 专家 × 3 = 768 个 | MTP 自己的 256 专家 |
| `mtp.N.ffn.shared_experts.w1/w2/w3.weight` | 3 个 | MTP 自己的 shared expert |
| `mtp.N.hc_*` | 6 个 | MTP 的 HC 路由（+ `hc_head_*` 3 个） |

MTP 总张量数约 1 055 个（按 description 推算）。

### 3.6 顶层 HC

| 模板 | 数量 | 说明 |
|---|---|---|
| `hc_head_fn` | 1 | 顶层 HC head 系数 |
| `hc_head_base` | 1 | 顶层 HC head 基向量 |
| `hc_head_scale` | 1 | 顶层 HC head 缩放 |

### 3.7 元数据

`version`、`model_quant_type`、`metadata`、`group_size`、`optional` 等几个 string/dict 类型的辅助字段。

---

## 四、各 tensor shape 的推导（理论）

### 4.1 MLA（Multi-Latent Attention）— 本模型跟 V2/V3 一样

MLA 的核心是**把 Q/K/V 都先压到低秩 latent 空间**：

```
Q 路径：x → wq_a (qr) → q_latent → wq_b → (Hq, D+Drope)
KV 路径：x → wkv (Hk·(D+Drope)) → kv_latent
解码时：kv_latent ↑ wkv_b → 标准 KV（Hk, D+Drope）
```

推导：

| 权重 | shape | 推导 | 公式 |
|---|---|---|---|
| `wq_a` | (qr, H) = (1024, 4096) | 输入 H，输出 qr | `q_latent = x @ wq_a.T` |
| `wq_b` | (Hq·(D+Drope), qr) = (36864, 1024) | 输入 qr，输出 Hq·(D+Drope) | `q = q_latent @ wq_b.T` |
| `wkv` | (Hk·(D+Drope), H) = (576, 4096) | 输入 H，输出 Hk·(D+Drope) | `kv_latent = x @ wkv.T` |
| `wo_a` | (or_, Hq·D) = (1024, 32768) | 输入 Hq·D，输出 or_ | `o_latent = attn_out @ wo_a.T` |
| `wo_b` | (H, or_·ogroups) = (4096, 8192) | 输入 or_·ogroups，输出 H | `out = o_latent @ wo_b.T` |

> Hk=1（只有 1 个共享 KV 头），kv_latent 维度 = 1·(512+64) = **576**——非常小，KV cache 极省。
> Hq=64，Q 输出维度 = 64·(512+64) = 64·576 = **36 864**。

### 4.2 O 端的 `o_groups=8`

O 输出要过 `wo_b` 时，shape 是 `(or_·ogroups) = 1024·8 = 8192`，说明 V4 把 64 个 Q 头分成 8 组，每组共享一个 wo_b 通道——进一步省参数。

### 4.3 MoE + Shared Expert

跟 V3 一样的 dual-expert 结构：每个 token 同时过 1 个 shared expert + K=6 个 routed expert。

**Routed expert**（256 个）：
| 权重 | shape | 推导 |
|---|---|---|
| `gate.weight` | (E, H) = (256, 4096) | router，输出 256 个 logits |
| `w1.weight` | (I, H) = (2048, 4096) | SwiGLU gate |
| `w2.weight` | (H, I) = (4096, 2048) | down |
| `w3.weight` | (I, H) = (2048, 4096) | SwiGLU up |

**Shared expert**（1 个，每 token 都过）：
| 权重 | shape | 说明 |
|---|---|---|
| `shared_experts.w1/w2/w3` | 同上 | 单独一个 expert，所有 token 共享 |

### 4.4 Norm 层

所有 norm weight 都是 (H,) = (4096,)：
- `layers.N.attn_norm.weight`
- `layers.N.ffn_norm.weight`
- `mtp.N.enorm.weight` / `hnorm.weight` / `norm.weight` / `ffn_norm.weight`
- 顶层 `norm.weight`

> 同 Qwen3：所有 norm 字段名带 `_norm` 但实现是 RMSNorm（因为没有 bias）。

### 4.5 词表

| 权重 | shape | 推导 |
|---|---|---|
| `embed.weight` | (V, H) = (129280, 4096) | 顶层 embed（注意**没有** `model.` 前缀） |
| `head.weight` | (V, H) = (129280, 4096) | 顶层 lm_head（注意**没有** `model.` 前缀） |

Vocab 129280 比 Qwen3 的 151936 略小。

### 4.6 Sanity check

```
总描述项 = 103 181
W8A8_DYNAMIC = 102 232（绝大多数 weight/scale/offset）
FLOAT = 945（norm、router、attn_sink、HC、indexer、compressor 等）
```

按主结构估算：

```
top-level:           3
每层 attn:           15 模板 × 43 = 645
每层 ffn gate:       ~3 模板 × 43 = 129
每层 experts:        9 模板 × 43 × 256 = 99 072
每层 shared:         9 模板 × 43 = 387
每层 HC:             6 模板 × 43 = 258
compressor:          4 模板 × 41 = 164
indexer:             ~7 模板 × 21 = 147
MTP:                ~50 模板 × 1 = 50 + experts 768
顶层 HC:             3
────────────────────────────────
合计: 约 100 853
```

跟 description 的 103 181 大致吻合（差的几百项是 mtp 内部 attn 等细节）。

---

## 五、单层计算图（含 MLA + 双 expert）

```
x ──┬─► attn_norm ──► Q 压缩 (wq_a) → q_latent → wq_b → Q (Hq, D+Drope)
   │                    │                              │
   │                    └► KV 压缩 (wkv) → kv_latent  │
   │                                          │
   │                                          ▼
   │                                  RoPE + qk_norm + attn
   │                                          │
   │                                          ▼
   │                                  O 压缩 (wo_a) → o_latent
   │                                          │
   │                                          ▼
   │                                  O 恢复 (wo_b) → out
   ├──────────────────────────────────────── + ◄───
   │
   ├─► ffn_norm ──► Router (256 logits, top-6, sqrtsoftplus)
   │                │
   │                ├─► Shared Expert: w1/w2/w3 (所有 token 都过)
   │                │
   │                └─► Routed Experts: 6 个 (top-6) → w1/w2/w3 → 加权 sum
   │
   └─► + ◄───
```

MTP 层（额外的 1 层）只用于训练时辅助 loss，推理时一般关闭。

---

## 六、T=100 端到端 shape 一图流

```
input_ids                          (100,)              int64
   │ embed (注意：embed.weight，无 model. 前缀)
   ▼
hidden                             (100, 4096)
   │ × 43 层
   ├─ attn_norm                    (100, 4096)
   ├─ wq_a: (100,4096) @ (1024,4096).T   → (100, 1024)        # q_latent
   ├─ wq_b: (100,1024) @ (36864,1024).T  → (100, 36864)       # Q
   │   reshape → (100, 64, 576)        # (Hq=64, D+Drope=576)
   ├─ wkv:  (100,4096) @ (576,4096).T    → (100, 576)         # kv_latent
   │   reshape → (100, 1, 576)          # (Hk=1, D+Drope=576)
   ├─ RoPE + q_norm (D=512)            shape 不变
   ├─ attn(Q, KV_latent↑)              # MLA 的核心：低秩 KV 注意力
   │   → (100, 64, 512)               # reshape 回 (100, 32768)
   ├─ wo_a: (100,32768) @ (1024,32768).T → (100, 1024)        # o_latent
   ├─ wo_b: (100,1024) @ (4096,8192).T  → (100, 4096)         # out
   ├─ residual +                       (100, 4096)
   │
   ├─ ffn_norm                        (100, 4096)
   ├─ gate: (100,4096) @ (256,4096).T  → (100, 256)           # router logits
   ├─ top-6 (sqrtsoftplus + noaux_tc) → topk_w (100,6) topk_i (100,6)
   │
   ├─ Shared Expert (所有 token 都过):
   │   w1: (100,4096) @ (2048,4096).T  → (100, 2048)
   │   w3: (100,4096) @ (2048,4096).T  → (100, 2048)
   │   silu(w1) * w3                   → (100, 2048)
   │   w2: (100,2048) @ (4096,2048).T  → (100, 4096)
   │
   ├─ Routed Experts (top-6):
   │   dispatch → (600, 4096)
   │   gate+up (600, 4096) @ (4096, 2048) → (600, 2048)       # gate+up 拼接
   │   silu *                                              → (600, 2048)
   │   down (600, 2048) @ (4096, 2048).T                    → (600, 4096)
   │   加权 sum → (100, 6, 4096) → (100, 4096)
   │
   ├─ final = shared + routed × routed_scaling_factor(1.5)
   ├─ residual +                       (100, 4096)
   ▼
... × 43 层 ...
   ▼
norm                                (100, 4096)
head: (100,4096) @ (129280,4096).T  → (100, 129280)         # logits
```

> 注：head.weight 跟 embed.weight 形状完全一样但**权重独立**（`tie_word_embeddings=false`）。

---

## 七、特色机制详解

### 7.1 MLA（Multi-Latent Attention）

参数量对比（Qwen3-30B-A3B 用标准 MHA+GQA）：

| 模型 | 注意力参数量（每层） |
|---|---|
| Qwen3-30B-A3B (GQA, Hq=32, Hk=4) | 18.9 M |
| DeepSeek-V4-Flash (MLA) | q: 1024·4096 + 36864·1024 = 4.2M + 37.7M = 42M；kv: 576·4096 = 2.4M；o: 1024·32768 + 4096·8192 = 33.6M + 33.6M = 67M → 合计约 112M |

MLA 看似参数更多，但**KV cache 极小**（Hk=1，每 token 只需 576 维 latent），换来 1M 上下文能力。

### 7.2 Lightning Indexer（21 层用）

`index_topk=512`：每个 Q token 先通过 indexer 选 top-512 个历史 token，**只对它们做完整 attention**。是 V3.2 Sparse Attention 的核心，节省长上下文 attention 计算量。

### 7.3 Compressor（41 层用）

把滑动窗口外的历史 K/V 压到低秩，节省 KV cache。配合 `sliding_window=128`，前 128 个 token 用完整 K/V，超出的部分用压缩表示。

### 7.4 HC（Hash Competition，3 层用）

通过哈希函数（`hc_*_fn`）+ 基向量（`hc_*_base`）+ 缩放（`hc_*_scale`）实现 token → expert 的快速路由，避免 256 个 expert 全部跑 softmax。3 层使用，其他层还是标准 router。

### 7.5 Shared Expert（V3/V4 标配 vs Qwen3 没有）

| 模型 | shared expert | routed expert |
|---|---|---|
| Qwen3-30B-A3B | ❌ 无 | 128 |
| DeepSeek-V3 | ✅ 1 | 256 |
| DeepSeek-V4-Flash | ✅ 1 | 256 |

Shared expert 是个"通用处理器"，所有 token 都过；routed expert 按路由权重加权求和。最终 `ffn_out = shared_out + routed_scaling_factor · routed_out`。

### 7.6 MTP（Multi-Token Prediction）

主干后挂 1 个 MTP 层，预测第 2 个 token，训练时作为辅助 loss，推理时可以用作 speculative decoding 的 draft 模型。

### 7.7 W8A8 量化

| 量化方案 | 权重大小 | 激活大小 |
|---|---|---|
| 原始 bf16 | 2 字节 | 2 字节 |
| W8A8 | 1 字节 + scale/offset | 1 字节（per-token dynamic quant） |

理论压缩比：~2× 权重 + ~2× 激活 ≈ 4× 总体。但 expert 数量大（256）让总参数继续膨胀，weight_scale 和 weight_offset 也占空间。

### 7.8 YaRN RoPE 扩展

```json
"rope_scaling": {
  "type": "yarn",
  "factor": 16,
  "original_max_position_embeddings": 65536,
  "beta_fast": 32,
  "beta_slow": 1
}
```

训练时 64K 上下文，YaRN 扩展到 1M（16×）。比 Qwen3 的"训练时就训到 128K"激进得多。

---

## 八、与 Qwen3-30B-A3B 的关键差异

| 维度 | Qwen3-30B-A3B | DeepSeek-V4-Flash |
|---|---|---|
| 架构 | dense GQA | MLA |
| 隐藏维度 | 2048 | 4096 |
| 层数 | 48 | 43 + 1 MTP |
| Q 头 / KV 头 | 32 / 4 | 64 / 1（MLA 共享） |
| head_dim | 128 | 512 + 64 RoPE |
| Expert | 128 routed | 256 routed + 1 shared |
| Top-K | 8 | 6 |
| 量化 | 原始 bf16 | W8A8_DYNAMIC |
| MoE 中间维度 | 768 | 2048 |
| 路由 | softmax + top-K | sqrtsoftplus + noaux_tc |
| 上下文 | 128K (rope_theta=1M, 无 scaling) | 1M (YaRN factor=16) |
| 命名风格 | `model.embed_tokens.weight` | `embed.weight`（无 `model.` 前缀） |
| Norm | `input_layernorm` | `attn_norm` / `ffn_norm`（更清晰） |
| 特色机制 | QK-Norm | MLA + Compressor + Indexer + HC + Shared Expert + MTP |

---

## 九、复现脚本

由于本模型权重未完整下载，无法用 `safetensors.safe_open` 读 header。改用 `quant_model_description.json`：

```python
import json, re
from collections import Counter

weights_dir = '/home/weights/DeepSeek-V4-Flash-w8a8-mtp'
desc = json.load(open(f'{weights_dir}/quant_model_description.json'))

# 模板归并
templates = Counter()
quant_types = {}
for k, v in desc.items():
    tmpl = re.sub(r'\.\d+\.', '.N.', k)
    templates[tmpl] += 1
    q = v if isinstance(v, str) else '<dict>'
    quant_types.setdefault(tmpl, Counter())[q] += 1

# 按数量排序
for tmpl, n in sorted(templates.items(), key=lambda x: -x[1])[:30]:
    qs = ', '.join(f'{q}={c}' for q, c in quant_types[tmpl].most_common())
    print(f'  {n:>6}  {tmpl}  ({qs})')
```

## 十、特殊命名 / 检查点

- **顶层命名无 `model.` 前缀**（`embed.weight` / `head.weight` / `norm.weight`），跟 V3 之前的 LLaMA 风格不同。MTP 层用 `mtp.N.*` 前缀。
- **专家命名 `w1/w2/w3`** 对应 V3 时代的 SwiGLU 三件套（HF 风格是 `gate_proj/up_proj/down_proj`，本模型用了 Megatron 风格的 `w1/w2/w3`，其中 `w1=gate, w3=up, w2=down`）。
- **shared expert 和 routed expert 并存**——输出相加而非平均。
- **`num_key_value_heads=1`** 是 MLA 的标志，KV cache 极省但有 trick 风险（KV 完全共享可能限制表达）。
- **`attn_sink`** 是 streaming LLM 的 attention sink 机制，每头一个标量。
- **MTP 层独立于主干**——单独有自己的 embed、head、attn、ffn、HC、indexer，是完整的子网络。
- **W8A8_DYNAMIC 配套 3 个 tensor**：weight (int8) + weight_scale (FLOAT) + weight_offset (FLOAT)，实际 weight ≈ 总量 / 3。
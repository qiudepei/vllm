# Qwen3-30B-A3B 权重结构详解

> 基于本地权重目录 `/home/weights/Qwen3-30B-A3B` 的静态分析。
> 通过遍历 `model-*.safetensors` + `model.safetensors.index.json` 得到 tensor 清单、shape、dtype 与文件分布。
>
> 文档分两部分：
> - **第一部分 · 理论**：从 config 推导每个 tensor 的 shape，解释 GQA、QK-Norm、RoPE、MoE 等机制的工作原理（不绑具体序列长度）。
> - **第二部分 · 实例**：以 **T=100** 的输入串一遍完整 forward，把每一步的 tensor shape、内存布局、代码形态全部落到具体数字上。

---

# 第一部分 · 理论

---

## 一、模型概况

`config.json` 关键字段：

| 字段 | 值 | 说明 |
|---|---|---|
| `architectures` | `Qwen3MoeForCausalLM` | Qwen3 系列 MoE 架构 |
| `model_type` | `qwen3_moe` | HuggingFace 标识 |
| `hidden_size` | 2048 | 隐藏维度 H |
| `intermediate_size` | 6144 | dense MLP 中间维度（本模型 MoE，实际用 `moe_intermediate_size`） |
| `num_hidden_layers` | 48 | 堆叠层数 L |
| `num_attention_heads` | 32 | Q 头数 Hq |
| `num_key_value_heads` | 4 | KV 头数 Hk（GQA ratio=8） |
| `head_dim` | 128 | 单头维度 D |
| `num_experts` | 128 | MoE 专家总数 E |
| `num_experts_per_tok` | 8 | 每 token 激活专家数 K（top-k） |
| `moe_intermediate_size` | 768 | 单专家中间维度 I |
| `norm_topk_prob` | true | router 权重归一化 |
| `vocab_size` | 151936 | 词表大小 V |
| `tie_word_embeddings` | false | `lm_head` 独立于 `embed_tokens` |
| `torch_dtype` | `bfloat16` | 全权重 bf16 |
| `max_position_embeddings` | 131072 | 最大序列长度 |
| `rope_theta` | 1e6 | RoPE 基频 |
| `rms_norm_eps` | 1e-6 | RMSNorm epsilon |
| `decoder_sparse_step` | 1 | 每层都走 MoE（无 dense 间隔层） |

> **A3B 含义**：总参数约 30B，激活参数约 3B。每个 token 只激活 8/128 = 6.25% 的专家。

下文约定：所有 shape 中 `T` 表示当前序列长度（prefill 阶段为整段长度，decode 阶段=1），下标 `e/k/t` 表示 expert/k-th 路由/token 维度。

---

## 二、权重文件分布

- 共 **16** 个 safetensors 文件，每个 ~4 GB，最后一个 ~1 GB。
- 总计 **18 867** 个 tensor，**30 532 122 624** 个元素（bf16 ≈ **61.06 GB**）。
- `model.safetensors.index.json` 的 `metadata.total_size = 61 064 245 248` 字节。

| 文件 | tensor 数 |
|---|---|
| model-00001-of-00016.safetensors | 1 063 |
| model-00002-of-00016.safetensors | 1 262 |
| model-00003-of-00016.safetensors | 1 258 |
| model-00004 ~ 00007 | 1 262 each |
| model-00008 | 1 258 |
| model-00009 ~ 00011 | 1 262 each |
| model-00012 | 1 260 |
| model-00013 | 1 256 |
| model-00014 ~ 00015 | 1 262 each |
| model-00016-of-00016.safetensors | 152（含最后两层的 attn/mlp、`lm_head`、`model.norm`） |

---

## 三、张量清单

### 3.1 Top-level（3 个）

| 路径 | Shape | dtype | 说明 |
|---|---|---|---|
| `model.embed_tokens.weight` | (V, H) = (151936, 2048) | bf16 | token embedding（输入共享） |
| `model.norm.weight` | (H,) = (2048,) | bf16 | 末层 RMSNorm |
| `lm_head.weight` | (V, H) = (151936, 2048) | bf16 | 输出投影（与 embed 不共享） |

### 3.2 每层（×48 层，共 393 个 tensor / 层）

| 路径模板 | Shape | dtype | 数量 | 说明 |
|---|---|---|---|---|
| `layers.N.input_layernorm.weight` | (H,) = (2048,) | bf16 | 48 | attn 前 RMSNorm |
| `layers.N.self_attn.q_proj.weight` | (Hq·D, H) = (4096, 2048) | bf16 | 48 | Q 投影 |
| `layers.N.self_attn.k_proj.weight` | (Hk·D, H) = (512, 2048) | bf16 | 48 | K 投影 |
| `layers.N.self_attn.v_proj.weight` | (Hk·D, H) = (512, 2048) | bf16 | 48 | V 投影 |
| `layers.N.self_attn.o_proj.weight` | (H, Hq·D) = (2048, 4096) | bf16 | 48 | 输出投影 |
| `layers.N.self_attn.q_norm.weight` | (D,) = (128,) | bf16 | 48 | per-head RMSNorm on Q |
| `layers.N.self_attn.k_norm.weight` | (D,) = (128,) | bf16 | 48 | per-head RMSNorm on K |
| `layers.N.post_attention_layernorm.weight` | (H,) = (2048,) | bf16 | 48 | MoE 前 RMSNorm |
| `layers.N.mlp.gate.weight` | (E, H) = (128, 2048) | bf16 | 48 | router：H → E logits |
| `layers.N.mlp.experts.M.gate_proj.weight` | (I, H) = (768, 2048) | bf16 | 6 144 | 128 专家 × SwiGLU gate |
| `layers.N.mlp.experts.M.up_proj.weight` | (I, H) = (768, 2048) | bf16 | 6 144 | 128 专家 × SwiGLU up |
| `layers.N.mlp.experts.M.down_proj.weight` | (H, I) = (2048, 768) | bf16 | 6 144 | 128 专家 × down |

> **GQA**：q_proj 输出 4096 (= 32×128)，k/v_proj 输出 512 (= 4×128)，KV 头被 8 个 Q 头共享。
> **QK-Norm**：per-head RMSNorm，权重 shape = (head_dim,)，与 Qwen3 原仓实现一致。
> **每层无 `shared_expert`**：与 Mixtral 不同，本模型没有共享专家，只有 routed experts。
> **`mlp_only_layers = []`**：所有 48 层均为 MoE 层，无 dense 间隔。

---

## 四、各 tensor shape 的推导（理论）

### 4.1 矩阵乘法约定

HF / Transformers 的 `nn.Linear` 把 `weight` 存成 `[out_features, in_features]`，矩阵乘为 `y = x @ W.T`。
下文推导均按此约定。

### 4.2 注意力

#### `q_proj : (Hq·D, H)`
- `out_features = Hq × D = 32 × 128 = 4096`
- `in_features  = H = 2048`
- 计算：`x (T, H) @ q_proj.T → (T, Hq·D)`，再 reshape 成 `(T, Hq, D)`。

#### `k_proj / v_proj : (Hk·D, H)`
- `out_features = Hk × D = 4 × 128 = 512`
- `in_features  = H = 2048`
- GQA：4 个 KV 头被 32/4 = **8 个 Q 头共享**，输出维度是 Q 的 1/8。reshape 后 `(T, Hk, D)`。

#### `o_proj : (H, Hq·D)`
- `out_features = H = 2048`
- `in_features  = Hq × D = 32 × 128 = 4096`
- 把多头 `(T, Hq, D)` 沿 head 维拼接回 `(T, Hq·D)`，再投回 hidden 维 `(T, H)`。

#### `q_norm / k_norm : (D,)`
- `shape = (head_dim,) = (128,)`
- **per-head RMSNorm**：对每个头内部的 D 维向量单独做 RMSNorm，所有 Q 头（或 K 头）共享同一份 γ 参数。
- 权重只有 D 个标量，而不是 `Hq·D` 个。

### 4.3 MoE

#### `mlp.gate : (E, H)` — router
- `out_features = E = 128`
- `in_features  = H = 2048`
- 每个 token 得到 E 个 logits，做 top-K softmax 得到路由权重。
- `x @ gate.T → (T, E)`。

#### 单个 expert 的三个权重
每个 expert 都是标准 SwiGLU MLP：`H → I → H`。

| 权重 | shape | 推导 | forward 公式 |
|---|---|---|---|
| `gate_proj` | (I, H) | `out = I = 768`, `in = H = 2048` | `g = x @ gate_w.T → (T, I)`，再 `silu` |
| `up_proj`   | (I, H) | 同上 | `u = x @ up_w.T → (T, I)` |
| `down_proj` | (H, I) | `out = H = 2048`, `in = I = 768` | `y = (silu(g)·u) @ down_w.T → (T, H)` |

中间维度用 `moe_intermediate_size=768`（而非 `intermediate_size=6144`）是 MoE 压缩参数的关键设计：
- 若每 expert 用 6144：E × 3 × H × 6144 = 128 × 3 × 2048 × 6144 ≈ **4.8B / 层**
- 缩到 768：E × 3 × I × H = 128 × 3 × 768 × 2048 ≈ **0.6B / 层**

每层 expert 总数：E × 3 = 384；48 层 = 18 432。

### 4.4 Norm 层

| 名称 | shape | 含义 |
|---|---|---|
| `input_layernorm` | (H,) = (2048,) | 进入 attn 之前的 RMSNorm γ |
| `post_attention_layernorm` | (H,) = (2048,) | 进入 MoE 之前的 RMSNorm γ |
| `model.norm` | (H,) = (2048,) | 最后一层 → lm_head 之前的 RMSNorm γ |

RMSNorm 权重长度等于被归一化的向量维度。

### 4.5 词表相关

#### `embed_tokens : (V, H)`
- `shape = (vocab_size, hidden_size)`
- 查表：`token_id → embed_tokens[token_id]` 取出一行 H 维向量。

#### `lm_head : (V, H)`
- `shape = (vocab_size, hidden_size)`
- `logits = hidden @ lm_head.T → (T, V)`
- `tie_word_embeddings=false`：`lm_head` 与 `embed_tokens` 是两份独立权重，各占一份 V×H。

### 4.6 Sanity check：tensor 总数

| 组件 | 数量 |
|---|---|
| top-level（embed_tokens, norm, lm_head） | 3 |
| 每层 norm/router/attn 共 9 个 × 48 | 432 |
| 每层 experts 384 个 × 48 | 18 432 |
| **合计** | **18 867** ✓ |

与 `model.safetensors.index.json` 的 `weight_map` 键数完全一致。

---

## 五、单层计算图

```
x ──┬─► input_layernorm ──► QKV ──► RoPE ──► Attention ──► o_proj ──► + ──► post_layernorm ──► MoE ──► +
    │                                                              ▲                              ▲
    └──────────────────────────────────────────────────────────────┘                              │
                                                                                                   │
x_prev ───────────────────────────────────────────────────────────────────────────────────────────┘
```

整层形状流向：

```
x:                (T, H)
input_layernorm:  (T, H)
q_proj:           (T, Hq·D) ──► reshape ──► (T, Hq, D)
k_proj:           (T, Hk·D) ──► reshape ──► (T, Hk, D)
v_proj:           (T, Hk·D) ──► reshape ──► (T, Hk, D)
RoPE:             shape 不变
q_norm / k_norm:  shape 不变（per-head RMSNorm）
attn(out):        (T, Hq·D)        # 拼接 Hq 个头的 D 维
o_proj:           (T, H)
residual +:       (T, H)
post_attn_norm:   (T, H)
gate (router):    (T, E) ──► top-K ──► (T, K)
对每个被激活的 expert (×K):
  gate_proj:      (T, I)
  up_proj:        (T, I)
  down_proj:      (T, H)
weighted_sum:     (T, H)
residual +:       (T, H)
```

---

## 六、注意力机制理论

### 6.1 GQA（Grouped Query Attention）

- **MHA**：每 Q 头有独立 KV 头 → KV 存储 = T × Hq × D × 2 × 2B（K+V）。
- **MQA**：所有 Q 头共享 1 个 KV 头 → KV 存储极小，但质量有损。
- **GQA**：把 Hq 个 Q 头分组，每组共享 1 个 KV 头（这里 Hq=32, Hk=4 → 8 个 Q 头共享 1 个 KV 头）。
- 实现上，K/V 物理上只存 `Hk × D`，计算时通过 `repeat_interleave(8, dim=head)` 在逻辑上扩展到 Hq 头。

### 6.2 RoPE（Rotary Position Embedding）

- 在 Q/K 上对相邻维度两两分组，按位置 p 旋转角度 `p · θ_i`，`θ_i = base ^ (-2i/D)`。
- θ 越大，对长序列的位置区分越好；`rope_theta=1e6` 是 Qwen3 的设定。
- 旋转后相对位置 m-n 由内积自然得到，KV cache 不需要重算。

### 6.3 QK-Norm（per-head RMSNorm）

- 与 Layernorm 不同，只对最后一维做 `x / RMS(x) · γ`。
- 权重 shape = (D,)，所有 Q 头（或 K 头）共享。
- 在 attention 之前对 Q/K 各做一次，能显著稳定训练，已成 Qwen3 的标配。

### 6.4 Attention 计算

```
Q: (B, T, Hq, D) → (B, Hq, T, D)
K: (B, T, Hk, D) → (B, Hk, T, D)
V: (B, T, Hk, D) → (B, Hk, T, D)

# GQA: K/V 在 head 维复制 (Hq/Hk) 倍
K_gqa: (B, Hq, T, D)
V_gqa: (B, Hq, T, D)

scores = Q @ K_gqa.T / sqrt(D)    # (B, Hq, T, T)
attn   = causal_mask(softmax(scores))
out    = attn @ V_gqa             # (B, Hq, T, D)
out    = out.transpose(1,2).reshape(B, T, Hq·D)
out    = o_proj(out)              # (B, T, H)
```

高效实现（FlashAttention）会把 K/V 的 8× 复制隐式完成，物理上只存 `Hk × D`。

---

## 七、MoE 机制理论

### 7.1 概览

```
hidden ─► Router ─► top-K ──┐
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
          expert 0       expert K-1   ...    共 E 个专家，每个 K 个权重
              │             │
              ▼             ▼
          expert_out_0   expert_out_K-1
              │             │
              └────── 加权 sum ──────┘
```

每层 MoE = `Router + Dispatch + Expert MLPs + Combine`，4 个阶段。

### 7.2 Router

每个 token 独立做一次线性分类，得到 E 个 logits，选 top-K：

```
router_logits = hidden @ gate.T          # (T, E)
scores        = router_logits.softmax(-1) # (T, E) 全分布
topk_w, topk_i = scores.topk(K, dim=-1)   # (T, K), (T, K)
if norm_topk_prob:
    topk_w /= topk_w.sum(-1, keepdim=True) # K 个权重归一化和=1
```

- `norm_topk_prob=true`（本模型）：让 K 个权重和=1，后续直接 `sum` 即可。
- `norm_topk_prob=false`：保留原始 softmax 数值，需要乘回原 logits 再合并。

训练时还引入 `router_aux_loss_coef=0.001` 的负载均衡 loss，避免所有 token 都路由到少数几个专家。

### 7.3 Dispatch：把 token 派发给对应专家

朴素写法对每个专家做一次 mask + gather，要 128 次小操作；工程上用 sort + segment 拍平：

```
flat_i = topk_i.view(-1)                # (T*K,)  展平为 (token, expert) 对
token_ids = arange(T).repeat_interleave(K)  # (T*K,)  每个 token 重复 K 次
sort_perm = flat_i.argsort()            # 按 expert id 排序
counts    = bincount(flat_i, minlength=E) # (E,) 每 expert 分到几个 token
offsets   = cumsum(counts)              # (E,) 前缀和 = 分段端点
expert_input = hidden[token_ids[sort_perm]]  # (T*K, H) 按 expert 连续排列
```

内存布局：

```
expert_input:  (T*K, H)
┌──── expert 0 的 n_0 个 token ────┬─ expert 1 的 n_1 ─┬─ expert 2 的 n_2 ─┬ ...
└──────────────────────────────────┴────────────────────┴────────────────────┘
   offsets[0]=0      offsets[1]=n_0   offsets[2]=n_0+n_1  ...
```

### 7.4 Expert 内部：SwiGLU MLP

每个 expert 是经典 SwiGLU：`H → I → H`：

```
g = silu(x @ gate_w.T)   # (n_e, I)
u = x   @ up_w.T         # (n_e, I)
y = (g * u) @ down_w.T   # (n_e, H)
```

naive 写法对 128 个 expert 各做 3 次小 GEMM，GPU 利用率极低。`fused_moe` 做三件事：

1. **gate + up 拼成一个 GEMM**：把 `(768, H)` 和 `(768, H)` 沿 out 维拼成 `(1536, H)`，一次算两件事。
2. **128 个 expert 的同算子合并为一次 grouped GEMM**（按 counts 分段，segment GEMM / block-sparse GEMM）。
3. **dispatch / combine 与 GEMM overlap**（stream 或把 gather 嵌进 kernel）。

代码参考 `vllm/model_executor/layers/fused_moe/`（`fused_moe.py`、`fused_marlin_moe.py` 等）。

### 7.5 Combine：scatter 回原 token 位置 + 加权求和

```
inv_perm        = sort_perm.argsort()        # 还原 (token, k) 顺序
expert_output   = expert_output[inv_perm]   # (T*K, H)
expert_output  *= flat_topk_w[:, None]      # (T*K, H) 加权
moe_out         = expert_output.view(T, K, H).sum(dim=1)  # (T, H)
```

- `norm_topk_prob=true`：归一化权重和=1，sum 等价于加权平均。
- 若 false：sum 之前要把权重还原成 softmax 后的全分布数值。

### 7.6 关键数字

| 维度 | 值 | 说明 |
|---|---|---|
| E | 128 | 专家总数 |
| K | 8 | 每 token 激活 |
| (T, K) 总派发 | T×K = 100×8 = 800 (T=100 时) | dispatch 后的总行数 |
| 单 expert 平均 | T·K / E = 6.25 | T=100 时的均值（实际长尾） |
| 单 expert 参数量 | 3 × I × H = 3 × 768 × 2048 = 4 718 592 ≈ 4.5M | |
| 单层 expert 总参数量 | E × 4.5M = 603 979 776 ≈ 0.6B | |
| 模型 expert 总参数量 | L × 0.6B ≈ 29B | 占总参数 95% |

---

## 八、参数 / 显存估算（bf16）

| 组件 | 元素数 | 占比 | 字节 (bf16) |
|---|---|---|---|
| embed_tokens | 311 164 928 | 1.02% | 622 MB |
| 每层 attn (q+k+v+o) | 18 874 368 | per-layer | 38 MB |
| 每层 norms + router | ≈ 269 440 | per-layer | 0.5 MB |
| 每层 experts | 603 979 776 | per-layer | 1.21 GB |
| 48 层合计 experts | 28 991 029 248 | 94.95% | 58.0 GB |
| lm_head | 311 164 928 | 1.02% | 622 MB |
| **总计** | **30 532 122 624** | 100% | **~61.0 GB** |

**激活参数估算（per token, per layer）**：

- attn: 4 × Hq·D + 2 × Hk·D + 2 × H·H ≈ 17 M
- MoE: K × (I·H + I·H + H·I) ≈ 12.6 M
- 加上 48 层堆叠 + lm_head，单 token 激活参数 ≈ 0.6–0.7 B（含 embed/head 一次性开销后约 ~3 B），与官方 "A3B" 一致。

---

## 九、特殊命名 / 检查点

- **`model.embed_tokens.weight` 与 `lm_head.weight` 不共享**（`tie_word_embeddings=false`），二者均为 (V, H) = (151936, 2048)。
- **无 `score` / `bias` 等额外权重**——每个专家就是标准的 `gate_proj/up_proj/down_proj` 三件套。
- **`mlp.gate.weight` shape 为 (E, H) = (128, 2048)**，与 `Qwen3MoeSparseMoeBlock.forward` 中的 `hidden_states @ gate_weight.T → (T, E)` 一致。
- **`q_norm` / `k_norm` 是 per-head RMSNorm**，weight shape = (D,) = (128,)；vLLM 仓实现位于 `vllm/model_executor/models/qwen3_moe.py`。
- **`decoder_sparse_step=1`、`mlp_only_layers=[]`**：48 层全是 MoE，无 dense 间隔，也无 shared expert（与 DeepSeek-V3 不同）。

---

## 十、归一化机制详解（LayerNorm vs RMSNorm + 命名约定）

> 补充章节，把"权重里那些 norm 层到底在干什么"讲透。

### 10.1 LayerNorm 公式

对一个向量 `x = (x_1, x_2, ..., x_n)` 在最后一维做归一化：

```
μ    = (1/n) Σ x_i                       # 1. 算均值
σ²   = (1/n) Σ (x_i - μ)²                 # 2. 算方差（要先减均值）
y_i  = γ_i · (x_i - μ) / √(σ² + ε) + β_i # 3. 归一化 + 仿射（2 个参数）
```

每层有 **2 个可学习参数**：缩放 `γ` 和偏移 `β`，shape 都是 `(n,)`。

### 10.2 RMSNorm 公式

```
RMS² = (1/n) Σ x_i²                       # 1. 直接算平方均值（不减均值）
y_i  = γ_i · x_i / √(RMS² + ε)            # 2. 归一化 + 仿射（1 个参数，无 β）
```

每层**只有 1 个可学习参数** `γ`，shape 是 `(n,)`。

### 10.3 核心区别

> **LayerNorm 要先"去中心化"再归一化，RMSNorm 跳过"去中心化"，只归一化尺度。**

| 步骤 | LayerNorm | RMSNorm |
|---|---|---|
| 减均值 | ✅ | ❌ 跳过 |
| 算方差 | ✅ 基于 (x-μ)² | ❌ 直接用 x² |
| 除以 √方差 | ✅ | ✅（用 RMS） |
| 缩放 γ | ✅ | ✅ |
| 偏移 β | ✅ 有 | ❌ 无 |
| 可学习参数数 | **2n** | **n** |
| 输出均值 | 必为 0 | 不一定（没减 μ） |

### 10.4 数值例子（直观感受差异）

设 `x = [1, 2, 3, 4]`，`γ = β = [1,1,1,1]`，`ε ≈ 0`：

**LayerNorm**：
```
μ     = (1+2+3+4)/4 = 2.5
x-μ   = [-1.5, -0.5, 0.5, 1.5]
σ²    = (2.25+0.25+0.25+2.25)/4 = 1.25
y     = (x-μ)/√1.25 ≈ [-1.342, -0.447, 0.447, 1.342]   # 均值=0
```

**RMSNorm**：
```
RMS²  = (1+4+9+16)/4 = 7.5
y     = x/√7.5 ≈ [0.365, 0.730, 1.095, 1.461]          # 均值≠0
```

### 10.5 RMSNorm 权重 shape 的推导规则

**形状 = `(被归一化的那一维的大小,)`**

就这一个规则——因为 RMSNorm 只有 γ 一个参数，weight 就是 1D tensor，长度 = 归一化维度。

本模型的所有 norm 权重：

| 权重 key | shape | 归一化维度 | 数量 |
|---|---|---|---|
| `input_layernorm.weight` | (H,) = (2048,) | hidden_size | 48 |
| `post_attention_layernorm.weight` | (H,) = (2048,) | hidden_size | 48 |
| `model.norm.weight` | (H,) = (2048,) | hidden_size | 1 |
| `self_attn.q_norm.weight` | (D,) = (128,) | head_dim | 48 |
| `self_attn.k_norm.weight` | (D,) = (128,) | head_dim | 48 |

合计 **193 个 norm 权重**：97 个 (2048,) + 96 个 (128,)，与 §四的 sanity check 一致。

### 10.6 两种归一化维度

**(1) 沿 hidden_size 归一化（3 类，97 个）**

```
RMSNorm(hidden_size=2048)     # weight shape = (2048,)
```

每个 token 的 2048 维向量整体归一化，每个维度有独立 γ，**所有 token 共享同一组 γ**。

```
       token 0        token 1     ...    token 99
γ = [γ_1, γ_2, ..., γ_2048]   ← 同一份 γ，对 T 个 token 共享
```

**(2) 沿 head_dim 归一化（2 类，per-head，96 个）**

```
RMSNorm(hidden_size=128)      # weight shape = (128,)
```

每个 head 内部的 128 维向量单独归一化，**所有 Q 头（或 K 头）共享同一组 γ**。

```
       head 0        head 1     ...    head 31
γ = [γ_1, ..., γ_128]   ← 同一份 γ，对 32 个 Q 头（或 4 个 K 头）共享
```

**QK-Norm 为什么是 128 而不是 32×128**：

直觉是 per-head 共享 γ，32 个 Q 头共用一份 128 维 γ，省参数且效果几乎一样。
如果给每个头独立一份 γ，应该是 32 × 128 = 4096 个数（Q 侧）/ 4 × 128 = 512 个数（K 侧），Qwen3 没这么设计。

### 10.7 命名 ≠ 实现：input_layernorm 实际是 RMSNorm

**这是本章最反直觉的一点**：权重 key 名叫 `input_layernorm`，但实现是 RMSNorm。

**证据 1：权重清单里只有 γ、没有 β**

所有 norm 层只有一个 `.weight`，**没有对应的 `.bias`**——这是 RMSNorm 的标志性特征（LayerNorm 应该有两个参数）。

```
input_layernorm.weight          (2048,)    ← 单参数
post_attention_layernorm.weight  (2048,)    ← 单参数
q_norm.weight                    (128,)     ← 单参数
k_norm.weight                    (128,)     ← 单参数
model.norm.weight                (2048,)    ← 单参数
```

**证据 2：vLLM 源码里类名全是 `RMSNorm`**

```python
# vllm/model_executor/models/qwen3_moe.py
from vllm.model_executor.layers.layernorm import RMSNorm

self.q_norm                  = RMSNorm(self.head_dim, eps=rms_norm_eps)              # L340
self.k_norm                  = RMSNorm(self.head_dim, eps=rms_norm_eps)              # L341
self.input_layernorm         = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # L411
self.post_attention_layernorm = RMSNorm(...)                                         # L412
self.norm                    = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # L470
```

5 个 self.xxx **类名全是 RMSNorm**。

**为什么叫 layernorm 这个名字**

这是 **HF 字段命名沿用 LLaMA 历史**：

1. 最早的 `LlamaForCausalLM`（HF Transformers）类里把这层叫 `input_layernorm`，但实际实现是 RMSNorm（Meta LLaMA 论文就是这么设计的）。HF 沿用了 LLaMA 仓库的字段名，没改成 `input_rmsnorm`。
2. Qwen1 / Qwen2 / Qwen3 都从 LLaMA 架构 fork 出来，**继承了字段名 `input_layernorm.weight`**，类内部仍然是 RMSNorm。
3. 权重文件里如果改成 `input_rmsnorm.weight`，就跟所有 LLaMA 系权重不兼容了——Qwen 即使实现换了，**字段名也得保留 `layernorm`**，否则别人加载老权重会报错。

**结论**：

> 看权重时**别被名字骗了**。要判断是 LayerNorm 还是 RMSNorm：
> 1. 看有没有 `.bias`——有 bias 大概率是 LayerNorm，没有就是 RMSNorm。
> 2. 看代码里的类名（`RMSNorm` vs `LayerNorm`）。
> 3. key 名带 `layernorm` 不一定真是 LayerNorm。

### 10.8 总结

| 维度 | LayerNorm | RMSNorm |
|---|---|---|
| 公式 | `(x-μ)/√(σ²+ε) · γ + β` | `x/√(RMS²+ε) · γ` |
| 减均值 | 是 | 否 |
| β 参数 | 有 | 无 |
| 每 norm 层参数 | 2n | n |
| 训练稳定性 | 强 | 接近 LN，实践够用 |
| 计算 / 显存 | 多 | 少 |
| 论文 | Ba 2016 | Zhang & Sennrich 2019 |
| LLaMA / Qwen / Mistral / Gemma | 不用 | 标配 |

---

# 第二部分 · 实例：序列长度 T = 100

> 代入具体数字走一遍 forward，每一步给真实 shape 和内存量级。所有 dtype 默认 bf16，shape 中省略。

---

## 十一、T=100 端到端 shape 一图流

```
input_ids                (100,)            int64
   │ embed_tokens
   ▼
hidden                   (100, 2048)       ←── 进入第 0 层（其余 47 层结构相同）
   │
   ├─ input_layernorm     (100, 2048)       # RMSNorm
   ├─ q_proj              (100, 4096)       # (100,2048) @ (4096,2048).T
   ├─ k_proj              (100, 512)        # (100,2048) @ (512,2048).T
   ├─ v_proj              (100, 512)        # (100,2048) @ (512,2048).T
   ├─ reshape             Q:(100,32,128)  K:(100,4,128)  V:(100,4,128)
   ├─ RoPE                shape 不变
   ├─ q_norm / k_norm     shape 不变（沿 D=128）
   ├─ attn(Q,K,V)         → (1, 32, 100, 100) → (1, 32, 100, 128) → (100, 4096)
   ├─ o_proj              (100, 2048)       # (100,4096) @ (2048,4096).T
   ├─ residual +          (100, 2048)
   ├─ post_attn_norm      (100, 2048)
   │
   ├─ gate (router)       (100, 128)        # (100,2048) @ (128,2048).T
   ├─ top-8 softmax       → topk_w (100,8)  topk_i (100,8)  int64
   │
   ├─ Dispatch
   │     flat_i = topk_i.view(-1)           (800,)         int64
   │     sort_perm = flat_i.argsort()        (800,)         int64
   │     counts = bincount(...)              (128,)         int64, sum=800
   │     offsets = cumsum(counts)            (128,)         int64
   │     expert_input = hidden[token[sort_perm]]   (800, 2048)   bf16
   │
   ├─ Expert MLPs (grouped GEMM)
   │     gate+up  → (800, 1536)
   │     silu(g)*u → (800, 768)
   │     down_proj → (800, 2048)
   │
   ├─ Combine
   │     inv_perm = sort_perm.argsort()      (800,)         int64
   │     expert_output[inv_perm]             (800, 2048)
   │     × flat_topk_w[:, None]              (800, 2048)    加权
   │     .view(100, 8, 2048).sum(dim=1)       (100, 2048)    = moe_out
   │
   └─ residual +          (100, 2048)
   ▼
... × 48 层 ...
   ▼
model.norm                (100, 2048)
lm_head                   (100, 151936)     ←── logits
```

---

## 十二、Embed + Attention 实例

### 11.1 Embedding 查表

```python
input_ids: (100,)   int64
hidden = embed_tokens[input_ids]    # 查表
hidden: (100, 2048)
```

### 11.2 RMSNorm

```python
hidden = (hidden / sqrt(mean(hidden**2, dim=-1, keepdim=True) + eps)) * input_layernorm.weight
# shape 仍为 (100, 2048)
```

### 11.3 QKV 投影

```python
q = hidden @ q_proj.T     # (100, 2048) @ (4096, 2048).T  → (100, 4096)
k = hidden @ k_proj.T     # (100, 2048) @ (512, 2048).T   → (100, 512)
v = hidden @ v_proj.T     # (100, 2048) @ (512, 2048).T   → (100, 512)

# reshape:
q → (100, 32, 128)        # Hq=32, D=128
k → (100,  4, 128)        # Hk=4  (GQA)
v → (100,  4, 128)
```

### 11.4 RoPE + QK-Norm

```python
# RoPE: 沿最后一维两两分组旋转
q = apply_rotary(q, position_ids, rope_theta=1e6)        # (100, 32, 128) 不变
k = apply_rotary(k, position_ids, rope_theta=1e6)        # (100,  4, 128) 不变

# per-head RMSNorm（权重 shape=(D,)）
q = q * q_norm.weight / sqrt(mean(q**2, dim=-1, keepdim=True) + eps)
k = k * k_norm.weight / sqrt(mean(k**2, dim=-1, keepdim=True) + eps)
```

### 11.5 Attention（含 GQA 复制）

```python
Q = q.transpose(0, 1).unsqueeze(0)             # (1, 32, 100, 128)
K = k.transpose(0, 1).unsqueeze(0)             # (1,  4, 100, 128)
V = v.transpose(0, 1).unsqueeze(0)             # (1,  4, 100, 128)

# GQA: K/V 沿 head 维复制 8 倍
K_gqa = K.repeat_interleave(8, dim=1)          # (1, 32, 100, 128)
V_gqa = V.repeat_interleave(8, dim=1)          # (1, 32, 100, 128)

scores = (Q @ K_gqa.transpose(-1, -2)) / sqrt(128)   # (1, 32, 100, 100)
attn   = causal_mask(softmax(scores, dim=-1))        # (1, 32, 100, 100)
out    = attn @ V_gqa                                 # (1, 32, 100, 128)

out = out.transpose(1, 2).reshape(100, 32 * 128)      # (100, 4096)
out = out @ o_proj.T                                   # (100, 4096) @ (2048,4096).T
                                                      # → (100, 2048)
```

> FlashAttention 实现里 K/V 的 8× 复制是**逻辑**上的，物理上 K/V 仍只占 `Hk × D`。

### 11.6 Residual + Norm

```python
hidden = hidden + out                          # (100, 2048) 残差
hidden = post_attention_layernorm(hidden)     # (100, 2048) RMSNorm
```

---

## 十三、MoE 实例（T=100 走完四个阶段）

> 把 `Router → Dispatch → Expert → Combine` 全部落到具体数字。

### 12.1 Router：决定每个 token 去哪 8 个专家

```python
router_logits = hidden @ gate_weight.T         # (100, 2048) @ (128, 2048).T → (100, 128)
routing_weights = router_logits.softmax(dim=-1) # (100, 128) 全分布
topk_weights, topk_idx = routing_weights.topk(8, dim=-1)
# topk_weights: (100, 8)   float32
# topk_idx:     (100, 8)   int64, 每个值 ∈ [0, 127]

if norm_topk_prob:
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    # 现在 (100, 8) 每行和=1
```

示例：第 0 个 token 路由到 expert `[3, 17, 42, 88, 5, 91, 27, 64]`，权重 `[0.21, 0.18, 0.14, 0.12, 0.10, 0.09, 0.08, 0.08]`。

形状变化：

```
hidden          (100, 2048)
router_logits   (100, 128)
topk_weights    (100, 8)    float32
topk_idx        (100, 8)    int64
```

### 12.2 Dispatch：把 token 派发给对应专家

```python
flat_i   = topk_idx.view(-1)                     # (800,)  int64
flat_w   = topk_weights.view(-1)                 # (800,)  float32
token_ids = torch.arange(100).repeat_interleave(8)   # (800,)  int64

sort_perm  = flat_i.argsort()                    # (800,)  按专家 id 排序
expert_ids = flat_i[sort_perm]                   # (800,)  排序后的专家 id
sorted_tok = token_ids[sort_perm]                # (800,)  排序后的 token id

counts  = torch.bincount(expert_ids, minlength=128)   # (128,) int64, sum=800
offsets = torch.cumsum(counts, dim=0)                 # (128,) int64

expert_input = hidden[sorted_tok]                # (800, 2048) bf16
```

**T=100 时的内存布局示意**（假设 counts = [6,4,5,0,3,…]，仅示意）：

```
expert_input:  (800, 2048)
┌──── expert 0 的 6 个 token ────┬─ expert 1 的 4 个 ─┬─ expert 2 的 5 个 ─┬ ...
│ t17  t3  t92  t44  t0  t55    │ t8 t12 t67 t99     │ t1 t3 t5 t77 t88  │
└────────────────────────────────┴─────────────────────┴───────────────────┘
   offsets[0]=0      offsets[1]=6   offsets[2]=10  offsets[3]=15 ...
```

每个 expert 平均 `100 × 8 / 128 = 6.25` 个 token；实际是长尾分布，热门 expert 可能吃 20+，冷门可能 0。

形状汇总：

```
hidden          (100, 2048)
flat_i/w        (800,) int64 / float32
sort_perm       (800,) int64
counts          (128,) int64 (sum=800)
offsets         (128,) int64
expert_input    (800, 2048) bf16
```

### 12.3 Expert MLPs

**Naive 版（理解用）**：

```python
for e in range(128):
    s, t = offsets[e-1] if e>0 else 0, offsets[e]
    x_e = expert_input[s:t]                  # (n_e, 2048)
    g   = silu(x_e @ gate_proj_w[e].T)       # (n_e, 768)
    u   = x_e @ up_proj_w[e].T               # (n_e, 768)
    y_e = (g * u) @ down_proj_w[e].T         # (n_e, 2048)
    expert_output[s:t] = y_e
```

形状：

```
expert_input     (800, 2048)
gate_proj out    (800, 768)
up_proj out      (800, 768)
silu(g)*u        (800, 768)
down_proj out    (800, 2048)     ← expert_output
```

**Fused 版（vLLM `fused_moe` 思路）**：

```python
# gate+up 沿 out 维拼起来：(800, 2048) @ (1536, 2048) → (800, 1536)
gu = grouped_gemm(expert_input, concat_gate_up_weight, offsets)
g, u = gu.chunk(2, dim=-1)
y    = grouped_gemm(silu(g) * u, down_weight, offsets)   # (800, 2048)
```

每个 expert 的参数量：

```
gate_proj: 768 × 2048 = 1 572 864
up_proj:   768 × 2048 = 1 572 864
down_proj: 2048 × 768 = 1 572 864
单 expert 合计        = 4 718 592   ≈ 4.5 M
单层 ×128             = 603 979 776 ≈ 0.6 B
模型 ×48              = 28 991 029 248 ≈ 29 B
```

### 12.4 Combine：scatter + 加权

```python
inv_perm      = sort_perm.argsort()              # (800,) 还原 (token, k) 顺序
expert_output = expert_output[inv_perm]         # (800, 2048)
expert_output = expert_output * flat_w[:, None] # (800, 2048) 元素乘 = 加权
moe_out       = expert_output.view(100, 8, 2048).sum(dim=1)  # (100, 2048)
```

形状变化：

```
expert_output  (800, 2048)
× flat_w       (800, 2048)            # 元素乘
view           (100, 8, 2048)
sum(dim=1)     (100, 2048)            = moe_out
```

`norm_topk_prob=true` 时 8 个权重和=1，sum 即加权平均。

### 12.5 Residual

```python
hidden = hidden + moe_out            # (100, 2048) + (100, 2048)
```

### 12.6 完整 MoE forward 伪代码

```python
def moe_forward(hidden, gate_w, gate_proj_w, up_proj_w, down_proj_w,
                num_experts=128, top_k=8):
    T, H = hidden.shape

    # 1. Router
    router_logits = hidden @ gate_w.T                       # (T, E)
    scores        = router_logits.softmax(dim=-1)
    topk_w, topk_i = scores.topk(top_k, dim=-1)
    if norm_topk_prob:
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

    # 2. Dispatch
    flat_i    = topk_i.view(-1)                              # (T*K,)
    flat_w    = topk_w.view(-1)                              # (T*K,)
    token_ids = torch.arange(T).repeat_interleave(top_k)     # (T*K,)
    sort_perm = flat_i.argsort()
    expert_ids = flat_i[sort_perm]
    sorted_tok = token_ids[sort_perm]
    expert_input = hidden[sorted_tok]                        # (T*K, H)

    counts  = torch.bincount(expert_ids, minlength=num_experts)
    offsets = torch.cumsum(counts, dim=0)

    # 3. Expert MLPs (grouped GEMM)
    expert_output = grouped_swiglu(expert_input, expert_ids, offsets,
                                   gate_proj_w, up_proj_w, down_proj_w)

    # 4. Combine
    inv_perm      = sort_perm.argsort()
    expert_output = expert_output[inv_perm]
    expert_output = expert_output * flat_w[:, None]
    moe_out       = expert_output.view(T, top_k, H).sum(dim=1)

    return moe_out
```

### 12.7 T=100 时各 step 的形状汇总

| Step | Shape | 含义 |
|---|---|---|
| hidden | (100, 2048) | RMSNorm 后输入 |
| router_logits | (100, 128) | router 输出 |
| topk_w / topk_i | (100, 8) | top-8 选择 |
| flat_i, flat_w | (800,) | 展平 |
| sort_perm | (800,) int64 | 按 expert id 排序 |
| expert_input | (800, 2048) | dispatch 后 |
| counts | (128,) int64 | 每 expert 的 token 数（和=800） |
| offsets | (128,) int64 | 前缀和 |
| expert_output | (800, 2048) | 全部 expert 输出 |
| × flat_w | (800, 2048) | 路由权重 |
| view → sum | (100, 8, 2048) → (100, 2048) | combine |
| moe_out | (100, 2048) | MoE 输出 |

---

## 十四、量级直觉 + 单 token 解码差异

### 13.1 T=100 时的显存与计算量

| 项目 | T=100 时的量 |
|---|---|
| KV cache（每层）| 2 × 100 × 4 × 128 × 2B = **200 KB** |
| KV cache（48 层合计）| ≈ **10 MB** |
| attn 分数矩阵（每层）| 32 × 100 × 100 × 2B = **640 KB** |
| attn 分数矩阵（48 层峰值）| ≈ **30 MB** |
| MoE 总派发 token 数 | 100 × 8 = **800**（单层） |
| 单 expert 平均 token | 800 / 128 = **6.25** |
| 一次 forward 激活显存峰值 | 几百 MB（远小于权重 ~61 GB） |

### 13.2 单 token 解码（T_q=1）的差异

推理时除 prefill 走上面完整流程，后续每生成一个新 token 时：

- **prefill**：上面 T=100 的流程一次性跑完，建好 KV cache。
- **decode（每步 T_q=1）**：
  - q_proj 后 shape `(1, 4096)` → `(1, 32, 128)`
  - k_proj/v_proj 后 shape `(1, 512)` → `(1, 4, 128)`，**只算当前 token 这 1 步的 K/V**，append 到 KV cache
  - attn：`Q (1, 32, 1, 128) @ K_cached (1, 32, L, 128).T → (1, 32, 1, L)`，L 是当前已缓存的序列长度
  - MoE router `(1, 128)` → top-8 → 只 dispatch **1×8=8 个 (token, expert) 对**到 expert（极端稀疏，counts 多为 0）

> decode 阶段 GPU 利用率天然低（attention 是 1×L 小矩阵，MoE 大部分 expert 闲置），所以推理优化重点是 **KV cache 压缩、continuous batching、speculative decoding、expert parallel**。

---

## 十五、复现脚本

```python
from safetensors import safe_open
import os, re
from collections import Counter

weights_dir = '/home/weights/Qwen3-30B-A3B'
files = sorted(f for f in os.listdir(weights_dir) if f.endswith('.safetensors'))

patterns = Counter()
total = 0
for f in files:
    with safe_open(os.path.join(weights_dir, f), framework='pt') as st:
        for k in st.keys():
            t = st.get_tensor(k)
            total += t.numel()
            tmpl = re.sub(r'\.\d+\.', '.N.', k)
            patterns[(tuple(t.shape), tmpl)] += 1

for (shp, tmpl), n in sorted(patterns.items(), key=lambda x: (x[0][1], -x[1])):
    print(f'{str(shp):<25} x{n:<6} {tmpl}')
print('total elements:', total, '| bf16 GB:', total * 2 / 1e9)
```
# Qwen3-30B-A3B 权重结构详解

> 基于本地权重目录 `/home/weights/Qwen3-30B-A3B` 的静态分析。  
> 通过遍历 `model-*.safetensors` + `model.safetensors.index.json` 得到 tensor 清单、shape、dtype 与文件分布。

---

## 一、模型概况

`config.json` 关键字段：

| 字段 | 值 | 说明 |
|---|---|---|
| `architectures` | `Qwen3MoeForCausalLM` | Qwen3 系列 MoE 架构 |
| `model_type` | `qwen3_moe` | HuggingFace 标识 |
| `hidden_size` | 2048 | 隐藏维度 |
| `intermediate_size` | 6144 | dense MLP 中间维度（本模型 MoE，实际用 `moe_intermediate_size`） |
| `num_hidden_layers` | 48 | 堆叠层数 |
| `num_attention_heads` | 32 | Q 头数 |
| `num_key_value_heads` | 4 | KV 头数（GQA，ratio=8） |
| `head_dim` | 128 | 单头维度 |
| `num_experts` | 128 | MoE 专家总数 |
| `num_experts_per_tok` | 8 | 每 token 激活专家数（top-k） |
| `moe_intermediate_size` | 768 | 单专家中间维度 |
| `norm_topk_prob` | true | router 权重归一化 |
| `vocab_size` | 151936 | 词表大小 |
| `tie_word_embeddings` | false | `lm_head` 独立于 `embed_tokens` |
| `torch_dtype` | `bfloat16` | 全权重 bf16 |
| `max_position_embeddings` | 131072 | 最大序列长度 |
| `rope_theta` | 1e6 | RoPE 基频 |
| `rms_norm_eps` | 1e-6 | RMSNorm epsilon |
| `decoder_sparse_step` | 1 | 每层都走 MoE（无 dense 间隔层） |

> **A3B 含义**：总参数约 30B，激活参数约 3B。每个 token 只激活 8/128 = 6.25% 的专家。

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

## 三、Top-level 张量

| 路径 | Shape | dtype | 说明 |
|---|---|---|---|
| `model.embed_tokens.weight` | (151936, 2048) | bf16 | token embedding（输入共享） |
| `model.norm.weight` | (2048,) | bf16 | 末层 RMSNorm |
| `lm_head.weight` | (151936, 2048) | bf16 | 输出投影（与 embed 不共享） |

---

## 四、每层张量（×48 层）

每层共 **393** 个 tensor，模板分布如下：

| 路径模板 | Shape | dtype | 数量 | 说明 |
|---|---|---|---|---|
| `layers.N.input_layernorm.weight` | (2048,) | bf16 | 48 | attn 前 RMSNorm |
| `layers.N.self_attn.q_proj.weight` | (4096, 2048) | bf16 | 48 | Q 投影 (32 heads × 128 dim) |
| `layers.N.self_attn.k_proj.weight` | (512, 2048) | bf16 | 48 | K 投影 (4 heads × 128 dim) |
| `layers.N.self_attn.v_proj.weight` | (512, 2048) | bf16 | 48 | V 投影 (4 heads × 128 dim) |
| `layers.N.self_attn.o_proj.weight` | (2048, 4096) | bf16 | 48 | 输出投影 |
| `layers.N.self_attn.q_norm.weight` | (128,) | bf16 | 48 | per-head RMSNorm on Q |
| `layers.N.self_attn.k_norm.weight` | (128,) | bf16 | 48 | per-head RMSNorm on K |
| `layers.N.post_attention_layernorm.weight` | (2048,) | bf16 | 48 | MoE 前 RMSNorm |
| `layers.N.mlp.gate.weight` | (128, 2048) | bf16 | 48 | router：hidden → 128 logits |
| `layers.N.mlp.experts.M.gate_proj.weight` | (768, 2048) | bf16 | 6 144 | 128 专家 × SwiGLU gate |
| `layers.N.mlp.experts.M.up_proj.weight` | (768, 2048) | bf16 | 6 144 | 128 专家 × SwiGLU up |
| `layers.N.mlp.experts.M.down_proj.weight` | (2048, 768) | bf16 | 6 144 | 128 专家 × down |

> **GQA**：q_proj 输出 4096 (= 32×128)，k/v_proj 输出 512 (= 4×128)，KV 头被 8 个 Q 头共享。  
> **QK-Norm**：每个 head 维度上单独做 RMSNorm，权重 shape = (head_dim,) = (128,)，与 Qwen3 原仓实现一致。  
> **每层无 `shared_expert`**：与 Mixtral 不同，本模型没有共享专家，只有 routed experts。  
> **`mlp_only_layers = []`**：所有 48 层均为 MoE 层，无 dense 间隔。

---

## 五、单层计算图

```
x ──┬─► input_layernorm ──► QKV ──► RoPE ──► Attention ──► o_proj ──► + ──► post_layernorm ──► MoE ──► +
    │                                                              ▲                              ▲
    └──────────────────────────────────────────────────────────────┘                              │
                                                                                                  │
x_prev ───────────────────────────────────────────────────────────────────────────────────────────┘
```

**MoE 子图**：

```
h ──► gate (128,2048) ──► top-8 softmax ──► 路由权重 w_i, 索引 idx_i
                                       │
                  ┌────────────────────┼────────────────────┐
                  ▼                    ▼                    ▼
       experts[idx_0] ...      experts[idx_7]
       ┌──────────────┐
       │ gate_proj    │ (768,2048)
       │ up_proj      │ (768,2048)
       │ silu(g)*u    │ → (768,)
       │ down_proj    │ (2048,768)
       └──────┬───────┘
              ▼
       Σ w_i · out_i  (加权求和)
```

**单专家 forward**：

```python
def expert(x, gate_w, up_w, down_w):
    # x: (T, 2048)
    g = silu(x @ gate_w.T)   # (T, 768)
    u = x @ up_w.T           # (T, 768)
    y = (g * u) @ down_w.T   # (T, 2048)
    return y
```

> 注意 HF/Transformers 的 `gate_proj/up_proj/down_proj.weight` 形状约定为 `[out_features, in_features]`，因此矩阵乘时需要 `.T`。

---

## 六、参数 / 显存估算

按 bf16 (2 字节) 估算：

| 组件 | 元素数 | 占比 | 字节 (bf16) |
|---|---|---|---|
| embed_tokens | 311 164 928 | 1.02% | 622 MB |
| 每层 attn (q+k+v+o) | 4096·2048 + 512·2048·2 + 2048·4096 = 18 874 368 | per-layer | 38 MB |
| 每层 norms + router | 2048·4 + 128 + 128 + 128·2048 ≈ 269 440 | per-layer | 0.5 MB |
| 每层 experts | 128 × (768·2048 + 768·2048 + 2048·768) = 603 979 776 | per-layer | 1.21 GB |
| 48 层合计 experts | 28 991 029 248 | 94.95% | 58.0 GB |
| lm_head | 311 164 928 | 1.02% | 622 MB |
| **总计** | **30 532 122 624** | 100% | **~61.0 GB** |

**激活参数估算（per token, per layer）**：

- attn: 4×2048·128 + 2×2048·128 + 2·2048·2048 ≈ 17 M FLOPs/elem (近似)
- MoE: 8 × (768·2048·2 + 2048·768) ≈ 12.6 M 参数
- 加上 48 层堆叠 + lm_head，单 token 激活参数 ≈ 0.6–0.7 B（含 embed/head 一次性开销后约 ~3 B），与官方 "A3B" 一致。

---

## 七、各 tensor shape 的推导与计算过程

> 约定：HF `nn.Linear` 的权重 shape 是 `[out_features, in_features]`，所以 `y = x @ W.T`。  
> GQA：多个 Q 头共享同一对 K/V 头，所以 K/V 投影的输出维度比 Q 小。

### 7.1 注意力部分

#### `q_proj : (4096, 2048)`

```
out_features = num_attention_heads × head_dim = 32 × 128 = 4096
in_features  = hidden_size                    = 2048
```

`x (T, 2048) @ q_proj.T → (T, 4096)`，再 reshape 成 `(T, 32, 128)` —— 32 个 Q 头，每头 128 维。

#### `k_proj / v_proj : (512, 2048)`

```
out_features = num_key_value_heads × head_dim = 4 × 128 = 512
in_features  = hidden_size                    = 2048
```

GQA 让 4 个 KV 头被 32 / 4 = **8 个 Q 头共享**，输出维度只有 Q 的 1/8。reshape 后 `(T, 4, 128)`。

#### `o_proj : (2048, 4096)`

```
out_features = hidden_size                    = 2048
in_features  = num_attention_heads × head_dim = 32 × 128 = 4096
```

把多头 `(T, 32, 128)` 沿 head 维拼接回 `(T, 4096)`，再投回 hidden 维 `(T, 2048)`。

#### `q_norm / k_norm : (128,)`

```
shape = (head_dim,) = (128,)
```

**per-head RMSNorm**：对每个头内部的 128 维向量单独做 RMSNorm，所有 Q 头（或 K 头）共享同一份 γ 参数。所以权重只有 128 个标量（一组），而不是 `num_heads × head_dim` 个。

---

### 7.2 MoE 部分

#### `mlp.gate : (128, 2048)` — router

```
out_features = num_experts = 128
in_features  = hidden_size = 2048
```

每个 token 过线性层得到 128 个 logits，做 top-8 softmax 得到路由权重。  
`x @ gate.T → (T, 128)`，对应 `[Qwen3MoeSparseMoeBlock.forward]` 中 `router_logits = hidden_states @ gate_weight.T`。

#### 单个 expert 的三个权重

每个 expert 都是标准 SwiGLU MLP：`hidden(2048) → intermediate(768) → hidden(2048)`。

| 权重 | shape | 推导 | forward 公式 |
|---|---|---|---|
| `gate_proj` | (768, 2048) | `out = moe_intermediate_size = 768`, `in = hidden_size = 2048` | `g = x @ gate_w.T → (T, 768)`，再 `silu` |
| `up_proj`   | (768, 2048) | 同上 | `u = x @ up_w.T → (T, 768)` |
| `down_proj` | (2048, 768) | `out = hidden_size = 2048`, `in = moe_intermediate_size = 768` | `y = (silu(g)·u) @ down_w.T → (T, 2048)` |

中间维度用 `moe_intermediate_size=768`（不是 `intermediate_size=6144`），是 MoE 压缩参数的关键设计：
- 若每个 expert 用 6144：128 × 3 × 2048 × 6144 ≈ **4.8B** / 层
- 缩到 768：128 × 3 × 768 × 2048 ≈ **0.6B** / 层

每层 expert 总数：128 experts × 3 weights = **384 个 expert 张量**；48 层 = 18 432 个。

---

### 7.3 Norm 层

#### `input_layernorm / post_attention_layernorm : (2048,)`

```
shape = (hidden_size,) = (2048,)
```

RMSNorm 的 γ 参数，长度等于被归一化的向量维度。一层 Transformer 里有两次 RMSNorm：
- `input_layernorm`：进入 attn 之前
- `post_attention_layernorm`：进入 MoE 之前

#### `model.norm : (2048,)`

最后一层输出到 `lm_head` 之前的 RMSNorm，shape 同上。

---

### 7.4 词表相关

#### `embed_tokens : (151936, 2048)`

```
shape = (vocab_size, hidden_size)
```

查表：`token_id → embed_tokens[token_id]` 取出一行 2048 维向量。

#### `lm_head : (151936, 2048)`

```
shape = (vocab_size, hidden_size)
```

把最后一层 2048 维隐向量映射到 151936 个 logits：`logits = hidden @ lm_head.T → (T, 151936)`。

本模型 `tie_word_embeddings=false`，所以 `lm_head` 与 `embed_tokens` 是**两份独立权重**，各占一份 151936×2048。

---

### 7.5 单层 forward 的 shape 变化（一图流）

```
x:                (T, 2048)
input_layernorm:  (T, 2048)                                # RMSNorm，无 shape 变化
q_proj:           (T, 2048) @ (4096,2048).T → (T, 4096)    # reshape → (T, 32, 128)
k_proj:           (T, 2048) @ (512,2048).T  → (T, 512)     # reshape → (T, 4,  128)
v_proj:           (T, 2048) @ (512,2048).T  → (T, 512)     # reshape → (T, 4,  128)
q_norm / k_norm:  (T, 32, 128) / (T, 4, 128) RMSNorm       # shape 不变
attn(out):        (T, 32, 128) → 拼接 → (T, 4096)
o_proj:           (T, 4096) @ (2048,4096).T → (T, 2048)
residual +:       (T, 2048)
post_attn_norm:   (T, 2048)
gate (router):    (T, 2048) @ (128,2048).T  → (T, 128)     # top-8 softmax
对每个被激活的 expert (×8):
  gate_proj:      (T, 2048) @ (768,2048).T  → (T, 768)
  up_proj:        (T, 2048) @ (768,2048).T  → (T, 768)
  down_proj:      (T, 768)  @ (2048,768).T  → (T, 2048)
weighted_sum:     (T, 2048)
residual +:       (T, 2048)
```

---

### 7.6 Sanity check：权重总数

| 组件 | tensor 数 |
|---|---|
| top-level（embed_tokens, norm, lm_head） | 3 |
| 每层 norm/router/attn 共 9 个 × 48 | 432 |
| 每层 experts 384 个 × 48 | 18 432 |
| **合计** | **18 867** ✓ |

与 `model.safetensors.index.json` 的 `weight_map` 键数完全一致。

---

## 八、特殊命名/检查点

- **`model.embed_tokens.weight` 与 `lm_head.weight` 不共享**（`tie_word_embeddings=false`），二者均为 (151936, 2048)。
- **无 `score` / `bias` 等额外权重**——每个专家就是标准的 `gate_proj/up_proj/down_proj` 三件套。
- **`mlp.gate.weight` shape 为 (128, 2048)**，即 `[num_experts, hidden_size]`，与 `Qwen3MoeSparseMoeBlock.forward` 中的 `hidden_states @ gate_weight.T → (T, num_experts)` 一致。
- **`q_norm` / `k_norm` 是 per-head RMSNorm**，weight shape = (head_dim,) = (128,)；在 vLLM 仓实现中位于 `vllm/model_executor/models/qwen3_moe.py`（具体行号见代码）。

---

## 九、MoE 层详解（以 T=100 为实例）

> 目标：把 §7.2 里 `(T, 128) → top-8 → (T, 8) → expert → (T, 2048)` 这条流水线拆到每一步的 tensor 形状、内存布局与代码形态。
>
> 配置：`num_experts=128`、`num_experts_per_tok=8`、`hidden_size=2048`、`moe_intermediate_size=768`、`norm_topk_prob=true`。

### 9.1 全流程概览

```
hidden  (100, 2048)
   │
   ├─► Router ─────────────────────────────┐
   │                                        ▼
   │                              ┌──────────────────────┐
   │                              │ router_logits (100,128)│
   │                              │ top-8 softmax        │
   │                              │ → topk_w (100,8)     │
   │                              │ → topk_i (100,8)     │
   │                              └──────────────────────┘
   │                                        │
   │                                        ▼
   ├─► Dispatch ─────────►  按专家 id 排序/分段
   │                                        │
   │                                        ▼
   │                              expert_inputs (N, 2048)
   │                              其中 N = 100 × 8 = 800
   │                                        │
   │                                        ▼
   ├─► 每个被激活的 expert ──►  SwiGLU MLP
   │                                        │
   │                                        ▼
   │                              expert_outputs (N, 2048)
   │                                        │
   │                                        ▼
   └─► Combine ───────────►  scatter 回 (100, 8, 2048) → 加权 sum → (100, 2048)
```

下面逐步展开。

---

### 9.2 Router：决定每个 token 去哪 8 个专家

```python
# 代码骨架
router_logits = torch.nn.functional.linear(hidden, gate_weight)
#              = hidden @ gate_weight.T
#              = (100, 2048) @ (128, 2048).T  → (100, 128)
```

每个 token 独立得到 128 个标量 logits，**还没有归一化**，数值差异可能很大。

```python
# top-8 routing
routing_weights = torch.softmax(router_logits, dim=-1)        # (100, 128) 全分布
topk_weights, topk_idx = routing_weights.topk(k=8, dim=-1)    # 各取前 8
# topk_weights: (100, 8)   —— 选中的 8 个权重（已 softmax 后）
# topk_idx:     (100, 8)   int64，0~127 的专家 id
```

> 因为 `norm_topk_prob=true`，vLLM 内部还会再做一次 `topk_weights / topk_weights.sum(-1, keepdim=True)`，让 8 个权重和为 1（不是用 128 个全分布）。两种归一化在数值上略有差别。

形状变化：

```
hidden          (100, 2048)
router_logits   (100, 128)
topk_weights    (100, 8)    float32
topk_idx        (100, 8)    int64
```

**这一步的关键问题**：现在每个 token 知道该去 8 个专家那里，但 128 个专家分布在 GPU 显存里，怎么高效把它们喂进去？

---

### 9.3 Dispatch：把 token 派发给对应专家

最朴素的写法（理解用，但很慢）：

```python
# naive dispatch：对 (token, expert) 双重循环
expert_inputs = []   # list of 128 个 tensor，每个形状 (n_i, 2048)
for e in range(128):
    mask = (topk_idx == e).any(dim=-1)            # (100,) bool，哪些 token 选了这个专家
    idx  = mask.nonzero(as_tuple=True)[0]         # 这些 token 的位置
    expert_inputs.append(hidden[idx])             # 取出对应的 hidden
```

naive 写法的两个痛点：
1. **128 次小矩阵乘**——每个 expert 处理的 token 数 N_e 是个位数到几十，GPU 利用率极低。
2. **每个 token 去 8 个专家** → 一个 token 的 hidden 会被复制到 8 份输入里，重复读取。

工程上用 **permute + sort + segment** 把这步拍平：

```python
# 1. 展平 (100, 8) → (800,)
flat_topk_idx = topk_idx.view(-1)                  # (800,) int64，值 0~127
flat_topk_w   = topk_weights.view(-1)              # (800,) float32

# 2. 按 expert id 排序：让同一个 expert 的 token 在内存里连续
sorted_idx = flat_topk_idx.argsort()               # (800,) int64，排序后位置
expert_id_sorted = flat_topk_idx[sorted_idx]       # (800,) 排好序的专家 id

# 3. 找出每个 expert 在排序数组里的起止位置
#    expert_offsets[e] = 第 e 个 expert 在排序数组中的 [start, end)
counts      = torch.bincount(expert_id_sorted, minlength=128)   # (128,) 每个 expert 的 token 数
offsets     = torch.cumsum(counts, dim=0)                       # (128,) 前缀和
# 比如 counts=[7,3,0,12,...]，offsets=[7,10,10,22,...]

# 4. 取出所有要送进 expert 的 hidden（按排序后顺序）
token_ids    = torch.arange(100).repeat_interleave(8)           # (800,) 第几个 token
sorted_token_ids = token_ids[sorted_idx]                        # (800,) 排序后的 token id
expert_inputs    = hidden[sorted_token_ids]                     # (800, 2048)  拍平、按 expert 连续
```

dispatch 后的内存布局（`counts` 假设是这样分布：[6,4,5,0,3,...]，仅示意）：

```
expert_inputs:  (800, 2048)
┌──── expert 0 的 6 个 token ────┬─ expert 1 的 4 个 ─┬─ expert 2 的 5 个 ─┬ ...
│ t17  t3  t92  t44  t0  t55    │ t8 t12 t67 t99     │ t1 t3 t5 t77 t88  │
└────────────────────────────────┴─────────────────────┴───────────────────┘
   offsets[0]=0      offsets[1]=6   offsets[2]=10  offsets[3]=15 ...
```

形状变化：

```
hidden          (100, 2048)
topk_idx/view   (800,) int64
sorted_idx      (800,) int64
counts          (128,) int64
expert_inputs   (800, 2048) bf16
```

---

### 9.4 专家内部：SwiGLU MLP

128 个 expert 各做三次 matmul（gate / up / down）。**Naive**：

```python
for e in range(128):
    s, t = offsets[e-1] if e>0 else 0, offsets[e]
    x_e = expert_inputs[s:t]                          # (n_e, 2048)
    g   = silu(x_e @ gate_w[e].T)                     # (n_e, 768)
    u   = x_e @ up_w[e].T                             # (n_e, 768)
    y_e = (g * u) @ down_w[e].T                       # (n_e, 2048)
    expert_outputs[s:t] = y_e
```

**Fused 版（vLLM `fused_moe` 的思路）**：

把每个 expert 的 `gate_proj` 和 `up_proj` 沿输出维拼起来，一次 GEMM 算两件事：

```python
# 把 128 个 expert 的 gate_w / up_w 各堆成 (128, 768, 2048)，
# 但只取 counts>0 的有效 expert（counts 累加为 800 个 token）
# 思路：每个 expert 处理不同数量的 token，用 grouped GEMM (grouped_matmul) 算一次
#
# gate+up 拼起来：(800, 2048) @ (E_eff, 2*768, 2048) → (800, 2*768)
# 然后 silu(g) * u
# 再 (800, 768) @ (E_eff, 2048, 768) → (800, 2048)
```

这一步的形状：

```
expert_inputs   (800, 2048)
gate_proj out   (800, 768)
up_proj out     (800, 768)
silu(g)*u       (800, 768)
down_proj out   (800, 2048)   ← expert_outputs
```

**每个 expert 内部的参数量**：

```
gate_proj:  768 × 2048  = 1 572 864
up_proj:    768 × 2048  = 1 572 864
down_proj:  2048 × 768  = 1 572 864
合计 / expert:                  4 718 592  (≈ 4.5 M)
合计 / 层 (×128):               603 979 776 (≈ 0.6 B)
合计 / 模型 (×48):            28 991 029 248 (≈ 29 B)
```

注意 `expert_outputs` 顺序仍然按 `sorted_idx` 排，不是原 token 顺序，下一步要 inverse。

---

### 9.5 Combine：scatter 回原 token 位置 + 加权求和

```python
# 1. 把 expert_outputs 用 sorted_idx 的逆排列还原回 (token, k) 顺序
inv_sort = sorted_idx.argsort()                       # (800,) 把"排序后位置"映回"原 (t,k) 位置"
expert_outputs = expert_outputs[inv_sort]              # (800, 2048)

# 2. 加权：每个 (token, expert) 对有一份路由权重
expert_outputs = expert_outputs * flat_topk_w[:, None] # (800, 2048) * (800,1) → 加权
                                                     # flat_topk_w 是 (800,) float32

# 3. reshape 回 (100, 8, 2048) 再在 k 维 sum
expert_outputs = expert_outputs.view(100, 8, 2048)    # (100, 8, 2048)
moe_out        = expert_outputs.sum(dim=1)            # (100, 2048)  把 8 个专家结果相加
```

形状变化：

```
expert_outputs  (800, 2048)
× flat_topk_w   (800, 2048)     # 元素乘
view            (100, 8, 2048)
sum(dim=1)      (100, 2048)     # moe_out
```

加权方式小结：
- **本模型 `norm_topk_prob=true`**：已经归一化为 8 个权重和=1，所以直接 `sum` 等价于加权平均。  
- 若 `norm_topk_prob=false`：需要保留 softmax 前的 logits，把 `topk_weights` 重新映射为 sum-to-1 之前的值再用。

---

### 9.6 完整 MoE forward 代码（伪 PyTorch）

```python
def moe_forward(hidden, gate_w, gate_proj_w, up_proj_w, down_proj_w,
                num_experts=128, top_k=8):
    """
    hidden:              (T, 2048)
    gate_w:              (128, 2048)           # router
    *_proj_w:            (128, 768, 2048) or (128, 2048, 768)
                          ↑ 128 个 expert 的权重堆叠
    return:              (T, 2048)
    """
    T, H = hidden.shape

    # ---- 1. Router ----
    router_logits = hidden @ gate_w.T                       # (T, 128)
    scores        = router_logits.softmax(dim=-1)           # (T, 128)  全分布
    topk_w, topk_i = scores.topk(top_k, dim=-1)             # (T, 8), (T, 8)
    if norm_topk_prob:
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)  # 归一化和=1

    # ---- 2. Dispatch ----
    flat_i = topk_i.view(-1)                                # (T*8,)
    flat_w = topk_w.view(-1)                                # (T*8,)
    token_ids = torch.arange(T).repeat_interleave(top_k)    # (T*8,)

    sort_perm    = flat_i.argsort()                         # 按专家 id 排序
    expert_ids   = flat_i[sort_perm]
    sorted_tok   = token_ids[sort_perm]
    expert_input = hidden[sorted_tok]                       # (T*8, 2048)

    counts       = torch.bincount(expert_ids, minlength=num_experts)  # (128,)
    offsets      = torch.cumsum(counts, dim=0)                         # (128,)

    # ---- 3. Expert MLPs (grouped GEMM) ----
    expert_output = grouped_swiglu(expert_input, expert_ids, offsets,
                                   gate_proj_w, up_proj_w, down_proj_w)
    # expert_output: (T*8, 2048)

    # ---- 4. Combine ----
    inv_perm      = sort_perm.argsort()                     # 还原 (t, k) 顺序
    expert_output = expert_output[inv_perm]
    expert_output = expert_output * flat_w[:, None]          # (T*8, 2048) 加权
    moe_out       = expert_output.view(T, top_k, H).sum(dim=1)   # (T, 2048)

    return moe_out
```

---

### 9.7 T=100 时各 step 的内存布局示意

```
Step              Shape              含义
─────────────────────────────────────────────────────────────
hidden            (100, 2048)        RMSNorm 后输入
router_logits     (100, 128)         router 输出
topk_w / topk_i   (100, 8)           top-8 选择
flat_i, flat_w    (800,)             展平
sort_perm         (800,) int64       按 expert id 排序的位置
expert_input      (800, 2048)        dispatch 后的输入
counts            (128,) int64       每个 expert 的 token 数（和=800）
offsets           (128,) int64       前缀和
expert_output     (800, 2048)        全部 expert 输出
加权 × flat_w     (800, 2048)        路由权重
view → sum        (100, 8, 2048) → (100, 2048)
moe_out           (100, 2048)        MoE 输出
─────────────────────────────────────────────────────────────
```

**T=100 时的负载估算（理想均衡）**：

```
每个 expert 平均处理 token 数 = 100 × 8 / 128 = 6.25
expert 输入矩阵       :  6.25 × 2048   ≈ 12.8 K 元素
gate+up GEMM 输出     :  6.25 × 1536   ≈ 9.6 K
down GEMM 输出        :  6.25 × 2048   ≈ 12.8 K
```

实际是**长尾分布**——热门 expert 可能吃 20+ token，冷门 expert 可能 0 token。Router 训练时加了 `router_aux_loss_coef=0.001` 的负载均衡 loss 来缓解。

---

### 9.8 为什么要 fused_moe

naive 写法在 128 个 expert 上要做 128 × 3 = 384 次小 GEMM，GPU 利用率极低。`fused_moe` 把：

1. **gate + up 拼成一个 GEMM**（一次算两件事）
2. **128 个 expert 的相同算子合并为一次 grouped GEMM**（按 counts 分段，segment GEMM / block-sparse GEMM）
3. **dispatch / combine 与 GEMM overlap**（用 stream 或把 gather 嵌进 kernel）

收益：吞吐通常能提升 3–10×，是 vLLM 跑 MoE 模型的关键优化。代码参考 `vllm/model_executor/layers/fused_moe/` 下的 `fused_moe.py`、`fused_marlin_moe.py` 等多种后端实现（CPU/GPU、bf16/int4/int8）。

---

### 9.9 MoE 层的总 FLOPs / 激活参数（per token）

```
激活参数量 / token / 层  =  8 × (768·2048 + 768·2048 + 2048·768)  / 2048
                          ≈ 8 × 1.5 M
                          ≈ 12.6 M 参数

× 48 层 + lm_head ≈ 0.6 B  → "A3B" 中的 ~3B 激活
```

---

## 十、复现脚本

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
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

## 九、复现脚本

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
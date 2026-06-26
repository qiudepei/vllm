# RMSNorm vs LayerNorm（兼 Qwen3 里的命名约定）

> 从 `weight_structure.md` 的「十、归一化机制详解」独立出来，作为专题文档。

---

## 一、LayerNorm 公式

对一个向量 `x = (x_1, x_2, ..., x_n)` 在最后一维做归一化：

```
μ    = (1/n) Σ x_i                       # 1. 算均值
σ²   = (1/n) Σ (x_i - μ)²                 # 2. 算方差（要先减均值）
y_i  = γ_i · (x_i - μ) / √(σ² + ε) + β_i # 3. 归一化 + 仿射（2 个参数）
```

每层有 **2 个可学习参数**：缩放 `γ` 和偏移 `β`，shape 都是 `(n,)`。

特点：减去均值让数据中心化，方差让数据归一化到单位方差。

---

## 二、RMSNorm 公式

```
RMS² = (1/n) Σ x_i²                       # 1. 直接算平方均值（不减均值）
y_i  = γ_i · x_i / √(RMS² + ε)            # 2. 归一化 + 仿射（1 个参数，无 β）
```

每层**只有 1 个可学习参数** `γ`，shape 是 `(n,)`。

---

## 三、核心区别（一句话）

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

直觉：LayerNorm 假设数据有"位置"和"尺度"两个自由度需要处理；RMSNorm 假设只有"尺度"一个自由度，"位置"不重要。

---

## 四、数值例子

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

注意 RMSNorm 输出**不一定是零均值**——因为没减 μ。

---

## 五、对比表

| 维度 | LayerNorm | RMSNorm |
|---|---|---|
| 公式 | `(x-μ)/√(σ²+ε) · γ + β` | `x/√(RMS²+ε) · γ` |
| 减均值 | 是 | 否 |
| β 参数 | 有 | 无 |
| 每 norm 层参数 | 2n | n |
| 训练稳定性 | 强 | 接近 LN，实践够用 |
| 计算 / 显存 | 多 | 少（少读少写一半） |
| 论文 | Ba 2016 | Zhang & Sennrich 2019 |
| LLaMA / Qwen / Mistral / Gemma | 不用 | 标配 |

为什么 LLM 现在都用 RMSNorm：质量跟 LN 几乎一样，**省一次均值计算 + 省一半参数 + 训练更稳定**，基本是白捡的优化。

---

## 六、RMSNorm 权重 shape 的推导规则

**形状 = `(被归一化的那一维的大小,)`**

就这一个规则——因为 RMSNorm 只有 γ 一个参数，weight 就是 1D tensor，长度 = 归一化维度。

### 6.1 Qwen3-30B-A3B 的所有 RMSNorm 权重

| 权重 key | shape | 归一化维度 | 数量 |
|---|---|---|---|
| `input_layernorm.weight` | (H,) = (2048,) | hidden_size | 48 |
| `post_attention_layernorm.weight` | (H,) = (2048,) | hidden_size | 48 |
| `model.norm.weight` | (H,) = (2048,) | hidden_size | 1 |
| `self_attn.q_norm.weight` | (D,) = (128,) | head_dim | 48 |
| `self_attn.k_norm.weight` | (D,) = (128,) | head_dim | 48 |

合计 **193 个 norm 权重**：97 个 (2048,) + 96 个 (128,)，与 `weight_structure.md` §四的 sanity check 一致。

### 6.2 公式表达

```
RMSNorm 权重 shape  = (归一化维度,)

归一化维度 = {
    hidden_size       # 沿 token 最后一维归一化 → (H,) = (2048,)
    head_dim          # 沿 head 最后一维归一化 → (D,) = (128,)
}
```

### 6.3 两种归一化维度

**(1) 沿 hidden_size 归一化（3 类，97 个）**

```python
RMSNorm(hidden_size=2048)        # weight shape = (2048,)
```

每个 token 的 2048 维向量整体归一化，每个维度有独立 γ，**所有 token 共享同一组 γ**。

```
       token 0        token 1     ...    token 99
γ = [γ_1, γ_2, ..., γ_2048]   ← 同一份 γ，对 T 个 token 共享
```

**(2) 沿 head_dim 归一化（2 类，per-head，96 个）**

```python
RMSNorm(hidden_size=128)         # weight shape = (128,)
```

每个 head 内部的 128 维向量单独归一化，**所有 Q 头（或 K 头）共享同一组 γ**。

```
       head 0        head 1     ...    head 31
γ = [γ_1, ..., γ_128]   ← 同一份 γ，对 32 个 Q 头（或 4 个 K 头）共享
```

**QK-Norm 为什么是 128 而不是 32×128**

直觉是 per-head 共享 γ，32 个 Q 头共用一份 128 维 γ，省参数且效果几乎一样。
如果给每个头独立一份 γ，应该是 32 × 128 = 4096 个数（Q 侧）/ 4 × 128 = 512 个数（K 侧），Qwen3 没这么设计。

---

## 七、命名 ≠ 实现：input_layernorm 实际是 RMSNorm

**这是本章最反直觉的一点**：权重 key 名叫 `input_layernorm`，但实现是 RMSNorm。

### 7.1 证据 1：权重清单里只有 γ、没有 β

所有 norm 层只有一个 `.weight`，**没有对应的 `.bias`**——这是 RMSNorm 的标志性特征（LayerNorm 应该有两个参数）。

```
input_layernorm.weight          (2048,)    ← 单参数
post_attention_layernorm.weight  (2048,)    ← 单参数
q_norm.weight                    (128,)     ← 单参数
k_norm.weight                    (128,)     ← 单参数
model.norm.weight                (2048,)    ← 单参数
```

### 7.2 证据 2：vLLM 源码里类名全是 `RMSNorm`

```python
# vllm/model_executor/models/qwen3_moe.py
from vllm.model_executor.layers.layernorm import RMSNorm

self.q_norm                   = RMSNorm(self.head_dim, eps=rms_norm_eps)              # L340
self.k_norm                   = RMSNorm(self.head_dim, eps=rms_norm_eps)              # L341
self.input_layernorm          = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # L411
self.post_attention_layernorm = RMSNorm(...)                                          # L412
self.norm                     = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # L470
```

5 个 `self.xxx` **类名全是 RMSNorm**。

### 7.3 为什么叫 layernorm 这个名字

这是 **HF 字段命名沿用 LLaMA 历史**：

1. 最早的 `LlamaForCausalLM`（HF Transformers）类里把这层叫 `input_layernorm`，但实际实现是 RMSNorm（Meta LLaMA 论文就是这么设计的）。HF 沿用了 LLaMA 仓库的字段名，没改成 `input_rmsnorm`。
2. Qwen1 / Qwen2 / Qwen3 都从 LLaMA 架构 fork 出来，**继承了字段名 `input_layernorm.weight`**，类内部仍然是 RMSNorm。
3. 权重文件里如果改成 `input_rmsnorm.weight`，就跟所有 LLaMA 系权重不兼容了——Qwen 即使实现换了，**字段名也得保留 `layernorm`**，否则别人加载老权重会报错。

### 7.4 结论

> 看权重时**别被名字骗了**。要判断是 LayerNorm 还是 RMSNorm：
> 1. 看有没有 `.bias`——有 bias 大概率是 LayerNorm，没有就是 RMSNorm。
> 2. 看代码里的类名（`RMSNorm` vs `LayerNorm`）。
> 3. key 名带 `layernorm` 不一定真是 LayerNorm。

---

## 八、回到 Qwen3

| 字段 | 值 | 含义 |
|---|---|---|
| `rms_norm_eps` | 1e-6 | RMSNorm 公式里的 ε，防止除零 |

config 里**没有** `rms_norm_eps` 之外的 norm 相关字段，没有 `layer_norm_eps`、没有 `qkv_bias`、没有 `qk_norm_eps`——所有 norm 共享同一个 `rms_norm_eps`。